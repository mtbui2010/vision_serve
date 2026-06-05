"""Concurrency + open-loop constant-rate load sweep (W1 + W4/W4b).

Drives an open-loop, coordinated-omission-correct generator (wrk2 or vegeta) against a system
that exposes VisionServe's ``POST /api/predict`` (VisionServe itself, or the FastAPI baselines),
sweeps concurrency C in {1,2,4,8,16,32,64}, optionally runs a fixed-QPS open-loop point, parses
CO-correct p50/p95/p99/p99.9 (the runners surface HdrHistogram percentiles from wrk2), and writes
a CSV + a provenance ``.meta.json``. RPS is the **measured** completed-request rate, never
``1000/mean_latency``.

SessionPool-size sweep (W4): VisionServe's pool size is hardcoded per model in Go today
(no runtime env var). So ``--pool-sizes`` only does something if you also pass ``--server-cmd``
with a ``{pool}`` placeholder; the harness then restarts the server once per pool size before
each concurrency sweep. Without ``--server-cmd`` the pool dimension is recorded as the server's
built-in size (and a single sweep is run). See eval/README.md section 3.

Example:

    python -m eval.loadgen.sweep \
        --target http://localhost:11435 --model mobilenet-v3 \
        --image test/testdata/sample.jpg \
        --concurrency 1,2,4,8,16,32,64 \
        --open-loop-rate 200 --duration 30 \
        --tool wrk2 --out eval/results/w1_mobilenet.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict
from typing import Optional

from eval.common.api import HEALTH_PATH, build_json_request, encode_image
from eval.common.provenance import build_meta, write_meta
from eval.loadgen import runners


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _write_payload(model: str, image: str, prompt: Optional[str], box: Optional[str]) -> str:
    import json

    body = build_json_request(model, encode_image(image), prompt=prompt, box=box)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="vs_payload_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(body, f)
    return path


def _wait_health(target: str, timeout_s: float = 60.0) -> bool:
    """Poll the server health endpoint until it responds or times out."""
    url = target.rstrip("/") + HEALTH_PATH
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310 - localhost only
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def _read_device(target: str, payload_path: str) -> str:
    """One probe request to read back the EP the server actually used (Result.device)."""
    try:
        with open(payload_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(  # noqa: S310 - localhost only
            target.rstrip("/") + "/api/predict",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            import json

            return json.loads(r.read()).get("device", "")
    except Exception:  # noqa: BLE001
        return ""


def _run_one(tool: str, url: str, payload: str, rate: int, conns: int,
             duration: int) -> runners.RunOutcome:
    if tool == "wrk2":
        return runners.run_wrk2(url, payload, rate=rate, connections=conns, duration_s=duration)
    if tool == "vegeta":
        return runners.run_vegeta(url, payload, rate=rate, duration_s=duration, connections=conns)
    raise ValueError(f"unknown tool {tool!r}")


class ServerController:
    """Optionally (re)start the SUT for a given pool size via a user --server-cmd template."""

    def __init__(self, target: str, server_cmd: Optional[str]) -> None:
        self.target = target
        self.server_cmd = server_cmd
        self._proc: Optional[subprocess.Popen] = None

    def restart_for_pool(self, pool: int) -> None:
        if not self.server_cmd:
            return  # caller relies on an already-running server at its built-in pool size
        self.stop()
        cmd = self.server_cmd.format(pool=pool)
        # shell=True so the template can include env assignments / pipes the user wants.
        self._proc = subprocess.Popen(cmd, shell=True)  # noqa: S602 - user-supplied template
        if not _wait_health(self.target):
            raise RuntimeError(f"server did not become healthy after restart for pool={pool}")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="http://localhost:11435",
                    help="base URL of the system under test")
    ap.add_argument("--path", default="/api/predict", help="predict endpoint path")
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True, help="image to replay (identical bytes per request)")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--box", default=None)
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64",
                    help="comma-separated connection counts C")
    ap.add_argument("--open-loop-rate", type=int, default=0,
                    help="constant arrival rate (req/s) for the open-loop point; 0 = derive per-C")
    ap.add_argument("--duration", type=int, default=30, help="seconds per cell")
    ap.add_argument("--pool-sizes", default="",
                    help="comma-separated SessionPool sizes (needs --server-cmd to take effect)")
    ap.add_argument("--server-cmd", default=None,
                    help="shell template to (re)start the SUT; must contain '{pool}'")
    ap.add_argument("--tool", default=None, choices=[None, "wrk2", "vegeta"],
                    help="load generator (auto-detected if omitted)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args(argv)

    tool = runners.autodetect_tool(args.tool)
    url = args.target.rstrip("/") + args.path
    concurrency = _parse_int_list(args.concurrency)
    pool_sizes = _parse_int_list(args.pool_sizes) if args.pool_sizes else [0]  # 0 = built-in
    payload = _write_payload(args.model, args.image, args.prompt, args.box)

    if args.pool_sizes and not args.server_cmd:
        print("WARNING: --pool-sizes given without --server-cmd; pool size cannot be changed "
              "at runtime (it is hardcoded in Go). Recording pool='builtin'. See README sec 3.",
              file=sys.stderr)

    controller = ServerController(args.target, args.server_cmd)
    rows: list[dict[str, object]] = []
    device_reported = ""
    total_requests = 0

    try:
        for pool in pool_sizes:
            controller.restart_for_pool(pool)
            if not _wait_health(args.target):
                raise RuntimeError(f"target {args.target} is not healthy; start the server first")
            if not device_reported:
                device_reported = _read_device(args.target, payload)

            for c in concurrency:
                # open-loop rate: explicit, else a high rate that saturates so the queue forms.
                rate = args.open_loop_rate if args.open_loop_rate > 0 else max(1, c) * 1000
                print(f"[sweep] pool={pool or 'builtin'} C={c} rate={rate} "
                      f"dur={args.duration}s tool={tool}", file=sys.stderr)
                outcome = _run_one(tool, url, payload, rate, c, args.duration)
                row = {
                    "model": args.model,
                    "pool_size": pool if pool else "builtin",
                    "concurrency": c,
                    "offered_rate": rate,
                    "mode": "open-loop",
                }
                row.update(outcome.as_row())
                rows.append(row)
                total_requests += outcome.requests
    finally:
        controller.stop()
        try:
            os.unlink(payload)
        except OSError:
            pass

    _write_csv(args.out, rows)
    meta = build_meta(
        experiment="loadgen-sweep (W1/W4/W4b)",
        target=args.target,
        model=args.model,
        device_reported=device_reported,
        n_requests=total_requests,
        loadgen_tool=tool,
        loadgen_version=runners.get_tool_version(tool),
        dataset=args.image,
        extra={"concurrency": concurrency, "pool_sizes": pool_sizes,
               "open_loop_rate": args.open_loop_rate, "duration_s": args.duration},
    )
    meta_path = write_meta(args.out, meta)
    print(f"wrote {args.out} ({len(rows)} rows) + {meta_path}")
    return 0


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # union of keys preserves any tool-specific columns
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# keep asdict imported for users who post-process RunOutcome programmatically
_ = asdict

if __name__ == "__main__":
    raise SystemExit(main())
