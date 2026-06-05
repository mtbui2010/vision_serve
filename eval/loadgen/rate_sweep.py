"""Open-loop constant-rate sweep (W4b / Pareto): sweep OFFERED RATE, measure goodput + tail.

Unlike sweep_openloop's concurrency sweep, this drives a fixed arrival RATE (open loop, CO-correct
via openloop_client.run_open_loop) at increasing offered rates, and records the MEASURED completed
rate (goodput) and p50/p95/p99 at each. Plotting measured-rps vs p99 gives the throughput-latency
Pareto frontier; plotting p99 vs offered-rate shows where each system's tail hockey-sticks (its real
ceiling) — replacing the bogus RPS=1000/mean Table.

Run:
    python -m eval.loadgen.rate_sweep --target http://localhost:11436 --label visionserve \
        --model mobilenet-v3 --image test/testdata/sample.jpg \
        --rates 25,50,100,200,400,600,800,1200 --duration 15 --max-inflight 256 \
        --out eval/results/pareto_mobilenet_visionserve.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request

from eval.common.api import HEALTH_PATH, PREDICT_PATH, build_json_request, encode_image
from eval.common.provenance import build_meta, write_meta
from eval.loadgen.openloop_client import run_open_loop


def _wait_health(target: str, timeout_s: float = 60.0) -> bool:
    url = target.rstrip("/") + HEALTH_PATH
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def _read_device(url: str, body: bytes) -> str:
    try:
        req = urllib.request.Request(url, data=body,  # noqa: S310
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            return str(json.loads(r.read()).get("device", ""))
    except Exception:  # noqa: BLE001
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="http://localhost:11436")
    ap.add_argument("--path", default=PREDICT_PATH)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--rates", default="25,50,100,200,400,600,800,1200")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--max-inflight", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    url = args.target.rstrip("/") + args.path
    body = json.dumps(build_json_request(args.model, encode_image(args.image))).encode()
    rates = [float(x) for x in args.rates.split(",") if x.strip()]

    if not _wait_health(args.target):
        print(f"ERROR: {args.target} not healthy", file=sys.stderr)
        return 2
    device = _read_device(url, body)
    # warmup
    for _ in range(20):
        try:
            urllib.request.urlopen(urllib.request.Request(  # noqa: S310
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"),
                timeout=30).read()
        except Exception:  # noqa: BLE001
            pass

    rows = []
    total = 0
    for rate in rates:
        res = run_open_loop(url, body, rate=rate, duration_s=args.duration,
                            max_inflight=args.max_inflight, timeout_s=30.0)
        # goodput_ok excludes errored requests: past saturation a server may fast-FAIL
        # connections, which would otherwise inflate "completed/wall" into a meaningless number.
        ok = max(0, res.completed - res.errors)
        goodput_ok = round(ok / res.wall_s, 2) if res.wall_s > 0 else 0.0
        row = {"system": args.label or args.target, "model": args.model,
               "offered_rate": rate, "goodput_rps": round(res.rps_measured, 2),
               "goodput_ok_rps": goodput_ok,
               "sent": res.sent, "completed": res.completed, "errors": res.errors}
        row.update(res.summary.as_row())
        rows.append(row)
        total += res.completed
        print(f"[rate_sweep] offered={rate} goodput={row['goodput_rps']} "
              f"p50={row.get('p50.0_ms')} p99={row.get('p99.0_ms')} errors={res.errors}",
              file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    meta = build_meta(experiment="rate_sweep open-loop Pareto (W4b)", target=args.target,
                      model=args.model, device_reported=device, n_requests=total,
                      loadgen_tool="rate_sweep.py", dataset=args.image,
                      extra={"rates": rates, "duration_s": args.duration,
                             "max_inflight": args.max_inflight, "label": args.label})
    write_meta(args.out, meta)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
