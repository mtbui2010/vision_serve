"""Pure-Python open-loop, coordinated-omission-correct load client (fallback / portable).

wrk2/vegeta are preferred (lower client overhead, battle-tested CO correctness). This module is
a dependency-light alternative for hosts where those binaries are unavailable, and for the
SessionPool sweep where we want one process that also reads back ``Result.device``.

Coordinated-omission correctness: requests are *scheduled* at fixed inter-arrival times derived
from a target rate (open loop). If a request's send time has already passed when a worker becomes
free, we still measure latency from the *scheduled* send time, not the actual send time — this is
exactly what wrk2 does to avoid hiding tail latency behind a stalled client.

This client is honest about its limits: at very high rates the Python/GIL client itself can become
the bottleneck. For headline p99.9 numbers prefer wrk2; use this for portability / smoke tests.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional

from eval.common.percentiles import LatencyRecorder, LatencySummary


@dataclass
class OpenLoopResult:
    summary: LatencySummary
    sent: int
    completed: int
    errors: int
    wall_s: float

    @property
    def rps_measured(self) -> float:
        return self.completed / self.wall_s if self.wall_s > 0 else 0.0


def _post(url: str, body: bytes, timeout_s: float) -> int:
    req = urllib.request.Request(  # noqa: S310 - localhost benchmarking
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
        r.read()
        return r.status


def run_open_loop(
    url: str,
    body: bytes,
    rate: float,
    duration_s: float,
    max_inflight: int,
    timeout_s: float = 10.0,
) -> OpenLoopResult:
    """Issue requests at a constant ``rate`` (req/s) for ``duration_s`` with <= ``max_inflight``
    concurrent in-flight requests. Latency is measured from the *scheduled* arrival time."""
    rec = LatencyRecorder()
    interval = 1.0 / rate if rate > 0 else 0.0
    schedule: "Queue[float]" = Queue()
    lock = threading.Lock()
    counters = {"sent": 0, "completed": 0, "errors": 0}
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            try:
                scheduled_at = schedule.get(timeout=0.1)
            except Empty:
                continue
            try:
                _post(url, body, timeout_s)
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
            done = time.perf_counter()
            # CO-correct: latency from the SCHEDULED send time, not the dequeue time.
            latency_ms = (done - scheduled_at) * 1000.0
            with lock:
                counters["completed"] += 1
                if not ok:
                    counters["errors"] += 1
                rec.record_ms(latency_ms)

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, max_inflight))]
    for w in workers:
        w.start()

    t0 = time.perf_counter()
    next_send = t0
    end = t0 + duration_s
    while True:
        now = time.perf_counter()
        if now >= end:
            break
        if now >= next_send:
            schedule.put(next_send)  # enqueue the SCHEDULED arrival timestamp
            with lock:
                counters["sent"] += 1
            next_send += interval
        else:
            time.sleep(min(next_send - now, 0.001))

    # drain outstanding, then stop
    while not schedule.empty():
        time.sleep(0.01)
    stop.set()
    for w in workers:
        w.join(timeout=timeout_s + 1)
    wall = time.perf_counter() - t0

    return OpenLoopResult(
        summary=rec.summary(),
        sent=counters["sent"],
        completed=counters["completed"],
        errors=counters["errors"],
        wall_s=wall,
    )


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    from eval.common.api import build_json_request, encode_image

    ap = argparse.ArgumentParser(description="pure-Python open-loop CO-correct load client")
    ap.add_argument("--target", default="http://localhost:11435")
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--rate", type=float, required=True, help="constant arrival rate (req/s)")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--max-inflight", type=int, default=64)
    args = ap.parse_args(argv)

    body = json.dumps(build_json_request(args.model, encode_image(args.image))).encode()
    res = run_open_loop(
        args.target.rstrip("/") + "/api/predict",
        body, args.rate, args.duration, args.max_inflight,
    )
    print(json.dumps({
        "sent": res.sent, "completed": res.completed, "errors": res.errors,
        "rps_measured": round(res.rps_measured, 3),
        **res.summary.as_row(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
