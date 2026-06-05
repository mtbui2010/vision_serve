"""VisionServe tensor-in benchmark (no server-side decode) — closed-loop concurrency sweep.

Hits VisionServe's new POST /api/infer_tensor?model=&shape= endpoint with a raw little-endian
float32 NCHW tensor, so the server skips JPEG/PNG decode + preprocess and only runs ORT +
postprocess. This is the apples-to-apples counterpart to Triton's tensor-in benchmark
(eval/baselines/triton_bench.py): same "no decode" regime, so the two ceilings are comparable.

Run:
    python -m eval.baselines.vs_tensor_bench --url http://localhost:11436 --model mobilenet-v3 \
        --shape 1,3,224,224 --concurrency 1,2,4,8,16,32,64 --duration 20 \
        --out eval/results/vs_tensor_mobilenet-v3.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
import urllib.request

import numpy as np

from eval.common.percentiles import LatencyRecorder


def _bench_one(url: str, body: bytes, conns: int, duration_s: float) -> dict:
    rec = LatencyRecorder()
    lock = threading.Lock()
    counters = {"completed": 0, "errors": 0}
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(url, data=body,  # noqa: S310
                                             headers={"Content-Type": "application/octet-stream"},
                                             method="POST")
                urllib.request.urlopen(req, timeout=30).read()  # noqa: S310
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
            lat = (time.perf_counter() - t0) * 1000.0
            with lock:
                counters["completed"] += 1
                if not ok:
                    counters["errors"] += 1
                rec.record_ms(lat)

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, conns))]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    time.sleep(duration_s)
    stop.set()
    for w in workers:
        w.join(timeout=30)
    wall = time.perf_counter() - t0
    row = {"system": "visionserve(tensor-in)", "concurrency": conns, "mode": "closed-loop",
           "completed": counters["completed"], "errors": counters["errors"],
           "wall_s": round(wall, 3),
           "rps_measured": round(counters["completed"] / wall, 2) if wall > 0 else 0.0}
    row.update(rec.summary().as_row())
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:11436")
    ap.add_argument("--model", default="mobilenet-v3")
    ap.add_argument("--shape", default="1,3,224,224")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    shape = tuple(int(x) for x in args.shape.split(","))
    endpoint = (args.url.rstrip("/") +
                f"/api/infer_tensor?model={args.model}&shape={','.join(map(str, shape))}")
    rng = np.random.default_rng(0)
    body = rng.standard_normal(shape).astype("<f4").tobytes()  # little-endian float32 NCHW

    # smoke check
    try:
        req = urllib.request.Request(endpoint, data=body,  # noqa: S310
                                     headers={"Content-Type": "application/octet-stream"},
                                     method="POST")
        import json
        r = json.loads(urllib.request.urlopen(req, timeout=30).read())  # noqa: S310
        print(f"[smoke] device={r.get('device')} task={r.get('task')} "
              f"top={ (r.get('classifications') or [{}])[0].get('class') }")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: smoke request failed: {e}")
        return 2

    rows = []
    for c in (int(x) for x in args.concurrency.split(",")):
        row = _bench_one(endpoint, body, c, args.duration)
        rows.append(row)
        print(f"[vs-tensor] C={c} rps={row['rps_measured']} p50={row.get('p50.0_ms')} "
              f"p99={row.get('p99.0_ms')} errors={row['errors']}")

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
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
