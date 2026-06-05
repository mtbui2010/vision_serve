"""Pure-Python concurrency + open-loop sweep (W1/W4) — no wrk2/vegeta required.

This is the dependency-light alternative to ``sweep.py`` for hosts without the wrk2/vegeta
binaries. It drives the SAME ``POST /api/predict`` endpoint (VisionServe or the FastAPI
baselines) with identical request bytes, sweeps concurrency C, and writes a CSV +
provenance ``.meta.json``.

Two modes (pick with ``--mode``):

* ``closed`` (default): a true *concurrency* sweep — C workers each send back-to-back for
  ``--duration`` seconds. Answers "throughput & latency vs concurrency" (W4) directly.
  ``rps_measured`` is the **completed** request rate (never ``1000/mean``).
* ``open``: a coordinated-omission-correct constant-rate point via
  ``openloop_client.run_open_loop`` — latency is measured from each request's *scheduled*
  arrival time, so tail latency is not hidden behind a stalled client (W4b).

The latency backend (HdrHistogram vs numpy fallback) is recorded per row in the ``backend``
column, so a reviewer always knows how the percentiles were computed. Nothing is fabricated:
the harness only emits what it measured against a real running server.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.request
from typing import Optional

from eval.common.api import HEALTH_PATH, PREDICT_PATH, build_json_request, encode_image
from eval.common.percentiles import LatencyRecorder
from eval.common.provenance import build_meta, write_meta
from eval.loadgen.openloop_client import run_open_loop


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _wait_health(target: str, timeout_s: float = 60.0) -> bool:
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


def _post(url: str, body: bytes, timeout_s: float) -> tuple[bool, Optional[dict]]:
    req = urllib.request.Request(  # noqa: S310 - localhost benchmarking
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
            payload = r.read()
            ok = r.status == 200
    except Exception:  # noqa: BLE001
        return False, None
    try:
        return ok, json.loads(payload)
    except Exception:  # noqa: BLE001
        return ok, None


def _read_device(url: str, body: bytes) -> str:
    ok, obj = _post(url, body, timeout_s=60.0)
    if ok and obj:
        return str(obj.get("device", ""))
    return ""


def _closed_loop(url: str, body: bytes, conns: int, duration_s: float,
                 timeout_s: float) -> dict[str, object]:
    """C workers each send requests back-to-back until the deadline (true concurrency sweep)."""
    rec = LatencyRecorder()
    lock = threading.Lock()
    counters = {"completed": 0, "errors": 0}
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            ok, _ = _post(url, body, timeout_s)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            with lock:
                counters["completed"] += 1
                if not ok:
                    counters["errors"] += 1
                rec.record_ms(lat_ms)

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, conns))]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    time.sleep(duration_s)
    stop.set()
    for w in workers:
        w.join(timeout=timeout_s + 1)
    wall = time.perf_counter() - t0

    summary = rec.summary()
    row: dict[str, object] = {
        "mode": "closed-loop",
        "completed": counters["completed"],
        "errors": counters["errors"],
        "wall_s": round(wall, 3),
        "rps_measured": round(counters["completed"] / wall, 3) if wall > 0 else 0.0,
    }
    row.update(summary.as_row())
    return row


def _open_loop_point(url: str, body: bytes, rate: float, conns: int, duration_s: float,
                     timeout_s: float) -> dict[str, object]:
    res = run_open_loop(url, body, rate=rate, duration_s=duration_s,
                        max_inflight=conns, timeout_s=timeout_s)
    row: dict[str, object] = {
        "mode": "open-loop",
        "offered_rate": rate,
        "sent": res.sent,
        "completed": res.completed,
        "errors": res.errors,
        "wall_s": round(res.wall_s, 3),
        "rps_measured": round(res.rps_measured, 3),
    }
    row.update(res.summary.as_row())
    return row


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="http://localhost:11435")
    ap.add_argument("--path", default=PREDICT_PATH)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True, help="image replayed identically on every request")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--box", default=None)
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per concurrency cell")
    ap.add_argument("--warmup", type=int, default=20, help="warmup requests before timing")
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--open-loop-rate", type=float, default=0.0,
                    help="constant arrival rate (req/s) for --mode open; 0 = derive C*200")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s)")
    ap.add_argument("--label", default="", help="system label recorded in each row (e.g. visionserve)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args(argv)

    url = args.target.rstrip("/") + args.path
    body = json.dumps(build_json_request(args.model, encode_image(args.image),
                                         prompt=args.prompt, box=args.box)).encode()
    concurrency = _parse_int_list(args.concurrency)

    if not _wait_health(args.target):
        print(f"ERROR: {args.target} not healthy — start the server first", file=sys.stderr)
        return 2

    device = _read_device(url, body)
    print(f"[sweep] target={args.target} model={args.model} device={device!r} "
          f"mode={args.mode}", file=sys.stderr)

    # warmup (covers cold-start / first-CUDA-launch so it does not pollute the C=1 cell)
    for _ in range(max(0, args.warmup)):
        _post(url, body, args.timeout)

    rows: list[dict[str, object]] = []
    total = 0
    for c in concurrency:
        print(f"[sweep] C={c} dur={args.duration}s mode={args.mode}", file=sys.stderr)
        if args.mode == "closed":
            cell = _closed_loop(url, body, c, args.duration, args.timeout)
        else:
            rate = args.open_loop_rate if args.open_loop_rate > 0 else c * 200.0
            cell = _open_loop_point(url, body, rate, c, args.duration, args.timeout)
        row: dict[str, object] = {"system": args.label or args.target,
                                  "model": args.model, "concurrency": c}
        row.update(cell)
        rows.append(row)
        total += int(cell.get("completed", 0) or 0)
        print(f"[sweep]   -> rps={cell.get('rps_measured')} "
              f"p50={cell.get('p50.0_ms')} p99={cell.get('p99.0_ms')} "
              f"errors={cell.get('errors')} backend={cell.get('backend')}", file=sys.stderr)

    _write_csv(args.out, rows)
    meta = build_meta(
        experiment=f"sweep_openloop ({args.mode}) W1/W4",
        target=args.target, model=args.model, device_reported=device,
        n_requests=total, loadgen_tool="sweep_openloop.py", loadgen_version="builtin",
        dataset=args.image,
        extra={"concurrency": concurrency, "duration_s": args.duration,
               "mode": args.mode, "label": args.label},
    )
    meta_path = write_meta(args.out, meta)
    print(f"wrote {args.out} ({len(rows)} rows) + {meta_path}")
    return 0


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
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


if __name__ == "__main__":
    raise SystemExit(main())
