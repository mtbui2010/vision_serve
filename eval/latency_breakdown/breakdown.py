"""Decompose end-to-end request latency into stages (W8 characterization study).

Target model: ``L = T_network + T_decode + T_preprocess + T_ORT_Run + T_postprocess + T_encode``.
The revision plan wants a stacked-bar breakdown across model-size x concurrency x hardware, to fit
a crossover curve for when non-inference overhead dominates.

What this harness can measure WITHOUT touching core:

* ``T_total_client``  — full wall time of the HTTP round trip (client-side).
* ``T_server_reported`` — the server's own ``Result.duration_ms`` (VisionServe sets this around
  the inference path in ``internal/lifecycle/session.go``). Treat it as the server-side compute
  component; it is NOT a full decode/pre/ORT/post split today.
* ``T_network_overhead = T_total_client - T_server_reported`` — HTTP + serialization + queueing.

What needs a server-side timing hook (TODO, owned by another stream — NOT edited here):

* The fine split decode / preprocess / ORT Run / postprocess / encode requires the server to
  emit per-stage timings (e.g. extra fields on ``Result`` or a ``Server-Timing`` response header,
  or Go ``pprof``/``runtime/trace`` on the server process). When such a hook exists, set
  ``--server-timing-header`` and this harness will parse the ``Server-Timing`` header. Until then
  we report the coarse client/server/network split and mark the fine stages as ``unavailable``
  rather than inventing them.

A purely client-side *lower bound* on decode+preprocess+encode can also be obtained by replaying
the SAME preprocessing locally (eval/baselines/preprocess.py) and timing it; we include that as a
reference column labeled ``client_side_preprocess_ref`` so reviewers see it is a client estimate,
not the server's actual time.

Usage:
    python -m eval.latency_breakdown.breakdown \
        --target http://localhost:11435 --model mobilenet-v3 \
        --image test/testdata/sample.jpg --n 500 --out eval/results/w8_mobilenet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from eval.common.api import build_json_request, encode_image, parse_result
from eval.common.percentiles import LatencyRecorder
from eval.common.provenance import build_meta, write_meta


@dataclass
class StageSamples:
    """Recorders per stage (ms)."""

    total_client: LatencyRecorder = field(default_factory=LatencyRecorder)
    server_reported: LatencyRecorder = field(default_factory=LatencyRecorder)
    network_overhead: LatencyRecorder = field(default_factory=LatencyRecorder)
    # Fine stages parsed from a Server-Timing header IF the server emits one.
    server_timing: dict[str, LatencyRecorder] = field(default_factory=dict)


def _parse_server_timing(header: str) -> dict[str, float]:
    """Parse an HTTP ``Server-Timing`` header: 'decode;dur=1.2, ort;dur=8.7' -> {name: ms}."""
    out: dict[str, float] = {}
    for part in header.split(","):
        part = part.strip()
        if not part:
            continue
        name = part.split(";", 1)[0].strip()
        dur = None
        for attr in part.split(";")[1:]:
            attr = attr.strip()
            if attr.startswith("dur="):
                try:
                    dur = float(attr[4:])
                except ValueError:
                    dur = None
        if dur is not None:
            out[name] = dur
    return out


def run(
    target: str,
    model: str,
    image: str,
    n: int,
    warmup: int = 5,
    server_timing: bool = False,
    prompt: Optional[str] = None,
    box: Optional[str] = None,
) -> tuple[StageSamples, str]:
    """Issue ``n`` sequential requests and record the coarse stage split."""
    body = json.dumps(build_json_request(model, encode_image(image),
                                         prompt=prompt, box=box)).encode()
    url = target.rstrip("/") + "/api/predict"
    samples = StageSamples()
    device = ""

    def one() -> Optional[tuple[float, float, dict[str, float]]]:
        req = urllib.request.Request(  # noqa: S310 - localhost benchmarking
            url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            raw = r.read()
            st_header = r.headers.get("Server-Timing", "") if server_timing else ""
        total_ms = (time.perf_counter() - t0) * 1000.0
        res = parse_result(raw)
        fine = _parse_server_timing(st_header) if st_header else {}
        return total_ms, res.duration_ms, fine, res.device  # type: ignore[return-value]

    for _ in range(max(0, warmup)):
        one()  # discard warm-up

    for _ in range(n):
        total_ms, server_ms, fine, dev = one()  # type: ignore[misc]
        device = device or dev
        samples.total_client.record_ms(total_ms)
        samples.server_reported.record_ms(server_ms)
        samples.network_overhead.record_ms(max(0.0, total_ms - server_ms))
        for name, ms in fine.items():
            samples.server_timing.setdefault(name, LatencyRecorder()).record_ms(ms)
    return samples, device


def to_rows(model: str, n: int, samples: StageSamples) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(stage: str, rec: LatencyRecorder, available: bool = True) -> None:
        s = rec.summary().as_row()
        s.update({"model": model, "stage": stage, "n": n, "available": available})
        rows.append(s)

    add("total_client", samples.total_client)
    add("server_reported", samples.server_reported)
    add("network_overhead", samples.network_overhead)
    if samples.server_timing:
        for name, rec in samples.server_timing.items():
            add(f"server_timing:{name}", rec)
    else:
        # Be explicit that the fine split is not available without a server-side hook.
        rows.append({
            "model": model, "stage": "fine_split", "n": n, "available": False,
            "note": "decode/preprocess/ORT/postprocess/encode split requires a server-side "
                    "timing hook (Server-Timing header or pprof); not fabricated.",
        })
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="http://localhost:11435")
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--box", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--server-timing-header", action="store_true",
                    help="parse a Server-Timing response header if the server emits one")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args(argv)

    samples, device = run(args.target, args.model, args.image, args.n,
                          warmup=args.warmup, server_timing=args.server_timing_header,
                          prompt=args.prompt, box=args.box)
    rows = to_rows(args.model, args.n, samples)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    write_meta(args.out, build_meta(
        experiment="latency-breakdown (W8)", target=args.target, model=args.model,
        device_reported=device, n_requests=args.n, dataset=args.image,
    ))
    print(f"wrote {args.out} ({len(rows)} stage rows); device={device or 'unknown'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
