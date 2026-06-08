#!/usr/bin/env python3
"""Edge EP×device benchmark cell: closed-loop sweep + Jetson tegrastats power.

Measures ONE (model, EP) point against an already-running VisionServe server, while
sampling tegrastats power rails in parallel, and appends a row per concurrency level to
a CSV. The EP is chosen by how the server was launched (VISIONSERVE_EP=cpu|cuda|tensorrt);
this script reads the actual EP back from the server's Result.device field — never guesses.

Energy/inference = (VDD_GPU_SOC + VDD_CPU_CV) mW * wall_seconds / n_requests   [mJ/inf]
(AGX Orin compute rails; VIN_SYS_5V0 captured for reference but not summed.)

No fabricated numbers: every row is produced by a real run on the target host. A sidecar
<out>.meta.json records git commit, host, nvpmodel mode, ORT lib, and a UTC timestamp.

Usage:
  python edge_bench.py --target http://localhost:11435 --model mobilenet-v3 \
      --images test/testdata/sample.jpg,demo/images/000000039769.jpg \
      --concurrency 1,8 --duration 10 --out eval/results/e1_matrix.csv
  # segmentation (PipelineModel) needs a prompt:
  python edge_bench.py ... --model mobile-sam --box 200,200,400,400
"""
import argparse, base64, csv, json, os, re, statistics, subprocess, sys, threading, time
import datetime, urllib.request

RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


def load_images(spec):
    paths = [p.strip() for p in spec.split(",") if p.strip()]
    bodies = []
    for p in paths:
        with open(p, "rb") as f:
            bodies.append(base64.b64encode(f.read()).decode())
    if not bodies:
        sys.exit("no images loaded")
    return paths, bodies


def make_body(model, b64, box):
    d = {"model": model, "image_base64": b64}
    if box:
        d["box"] = box
    return json.dumps(d).encode()


def one_call(url, body):
    t = time.perf_counter()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        r.read()
    return (time.perf_counter() - t) * 1000.0


class Tegra:
    """Sample tegrastats in the background; report mean rail power over a time window."""
    def __init__(self, interval_ms=200):
        self.interval = interval_ms
        self.rows = []          # (epoch, {rail: mW})
        self.proc = None
        self.thr = None

    def _run(self):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in self.proc.stdout:
            m = re.match(r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
            if not m:
                continue
            ep = time.mktime(datetime.datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").timetuple())
            vals = {}
            for rail in RAILS:
                g = re.search(rail + r" (\d+)mW", line)
                if g:
                    vals[rail] = int(g.group(1))
            self.rows.append((ep, vals))

    def start(self):
        self.thr = threading.Thread(target=self._run, daemon=True)
        self.thr.start()

    def stop(self):
        if self.proc:
            self.proc.terminate()

    def window(self, start, end):
        sub = [r for r in self.rows if start - 0.5 <= r[0] <= end + 0.5] or self.rows
        out = {}
        for rail in RAILS:
            xs = [r[1].get(rail) for r in sub if r[1].get(rail) is not None]
            out[rail] = statistics.mean(xs) if xs else 0.0
        return out


def read_device(url, body):
    """Single call to read back the server's actual EP (Result.device)."""
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read()).get("device", "?")


def server_rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None


def sweep_cell(url, model, bodies, C, duration, rss_pid):
    lat, cnt, lk = [], [0], threading.Lock()
    peak_rss = [0.0]
    start_epoch = time.time()
    stop_at = time.perf_counter() + duration
    idx = [0]

    def worker():
        while time.perf_counter() < stop_at:
            with lk:
                b = bodies[idx[0] % len(bodies)]; idx[0] += 1
            ms = one_call(url, b)
            with lk:
                lat.append(ms); cnt[0] += 1

    def rss_sampler():
        while time.perf_counter() < stop_at:
            if rss_pid:
                v = server_rss_mb(rss_pid)
                if v and v > peak_rss[0]:
                    peak_rss[0] = v
            time.sleep(0.2)

    ts = [threading.Thread(target=worker) for _ in range(C)]
    rs = threading.Thread(target=rss_sampler, daemon=True)
    t0 = time.perf_counter()
    rs.start()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - t0
    end_epoch = time.time()
    lat.sort()
    pct = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]
    return dict(C=C, n=cnt[0], wall=wall, rps=cnt[0] / wall,
                p50=pct(0.50), p95=pct(0.95), p99=pct(0.99), min=lat[0],
                start=start_epoch, end=end_epoch, peak_rss_mb=peak_rss[0])


def nvpmodel_mode():
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
        m = re.search(r"NV Power Mode:\s*(\S+)", out)
        return m.group(1) if m else "?"
    except Exception:
        return "?"


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://localhost:11435")
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="test/testdata/sample.jpg")
    ap.add_argument("--box", default="")
    ap.add_argument("--concurrency", default="1,8")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--server-pid", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    url = args.target.rstrip("/") + "/api/predict"
    paths, bodies = load_images(args.images)
    bodies = [make_body(args.model, b, args.box) for b in bodies]
    Cs = [int(x) for x in args.concurrency.split(",")]

    # cold-start: time the first call (TRT builds its engine here), then warm up.
    t_cold = time.perf_counter()
    one_call(url, bodies[0])
    cold_ms = (time.perf_counter() - t_cold) * 1000.0
    for _ in range(args.warmup):
        one_call(url, bodies[0])
    device = read_device(url, bodies[0])

    tg = Tegra(); tg.start(); time.sleep(0.6)
    rows = []
    for C in Cs:
        r = sweep_cell(url, args.model, bodies, C, args.duration, args.server_pid)
        w = tg.window(r["start"], r["end"])
        gpu, cpu, v5 = w["VDD_GPU_SOC"], w["VDD_CPU_CV"], w["VIN_SYS_5V0"]
        compute_mw = gpu + cpu
        r.update(model=args.model, device=device, cold_ms=round(cold_ms, 1),
                 gpu_soc_mw=round(gpu), cpu_mw=round(cpu), vin5v_mw=round(v5),
                 compute_w=round(compute_mw / 1000.0, 2),
                 mj_per_inf=round(compute_mw * r["wall"] / r["n"], 1))
        rows.append(r)
        print(f"{args.model:<16} {device:<10} C={C:<3} rps={r['rps']:6.1f} "
              f"p50={r['p50']:6.1f} p95={r['p95']:6.1f} p99={r['p99']:6.1f} "
              f"pow={r['compute_w']:4.1f}W {r['mj_per_inf']:6.1f}mJ/inf "
              f"rss={r['peak_rss_mb']:.0f}MB cold={cold_ms:.0f}ms")
    tg.stop()

    cols = ["model", "device", "C", "n", "rps", "p50", "p95", "p99", "min",
            "compute_w", "gpu_soc_mw", "cpu_mw", "vin5v_mw", "mj_per_inf",
            "peak_rss_mb", "cold_ms"]
    new = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(r[k], 1) if isinstance(r.get(k), float) else r.get(k)) for k in cols})

    meta = dict(git_commit=git_commit(), host=os.uname().nodename,
                nvpmodel=nvpmodel_mode(), ort_dylib=os.getenv("ORT_DYLIB_PATH", "?"),
                visionserve_ep=os.getenv("VISIONSERVE_EP", "(manifest default)"),
                images=paths, box=args.box, duration_s=args.duration,
                utc=datetime.datetime.utcnow().isoformat() + "Z")
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
