"""
VisionServe benchmark harness.
Measures latency / throughput / memory for multiple baselines on RF-DETR
(detection) and GroundingDINO (open-vocab) — the two models with real weights.

Baselines:
  1. visionserve_go_http   - Go binary HTTP server (port 11435)
  2. python_ort_direct     - Python onnxruntime, no HTTP
  3. python_fastapi_http   - FastAPI + onnxruntime server (port 11436)
  4. cpp_ort_direct        - C++ ORT binary (pure engine, no preprocess)

Run as:
  python3 benchmarks/run_all.py [--gpu] [--cpu] [--quick]
"""
import argparse, json, os, signal, statistics, subprocess, sys, time, threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "benchmarks" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PY   = "/home/trung/miniconda3/envs/label/bin/python3"
BIN  = str(ROOT / "bin" / "visionserve")
IMG  = "/tmp/cats.jpg"
CATS_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

RF_REAL  = str(ROOT / "models/rf-detr/rf-detr-base-real.onnx")
GDINO    = str(ROOT / "models/grounding-dino/model.onnx")
SAM_ENC  = str(ROOT / "models/mobile-sam/mobile_sam_encoder.onnx")
SAM_DEC  = str(ROOT / "models/mobile-sam/mobile_sam_decoder_single.onnx")
VOCAB    = str(ROOT / "models/grounding-dino/vocab.txt")
MODELS   = str(ROOT / "models")

N_WARMUP = 5
N_RUNS   = 30

# ── helpers ──────────────────────────────────────────────────────────────────

def ensure_image():
    if not os.path.exists(IMG) or os.path.getsize(IMG) < 10000:
        subprocess.run(["curl", "-sL", CATS_URL, "-o", IMG], check=True)

def pstats(times):
    s = sorted(times)
    n = len(s)
    return {
        "p50_ms": round(statistics.median(s), 1),
        "p95_ms": round(s[int(n * 0.95)], 1),
        "p99_ms": round(s[min(int(n * 0.99), n-1)], 1),
        "mean_ms": round(statistics.mean(s), 1),
        "throughput_rps": round(1000 / statistics.mean(s), 2),
        "n_runs": n,
    }

def get_rss(pid=None):
    try:
        import psutil
        p = psutil.Process(pid or os.getpid())
        return p.memory_info().rss // (1024*1024)
    except Exception:
        try:
            path = f"/proc/{pid or os.getpid()}/status"
            for line in open(path):
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
        except Exception:
            return None

def get_vram(pid=None):
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().splitlines():
            p, mem = line.split(",")
            if pid is None or int(p.strip()) == pid:
                return int(mem.strip().split()[0])
    except Exception:
        pass
    return None

def gpu_env():
    """Return environment dict for GPU inference."""
    import glob
    nv_base = "/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia"
    libs = ":".join(glob.glob(f"{nv_base}/*/lib"))
    ort_gpu = ("/home/trung/trung_workdir/vision_forge/frontend/"
               "node_modules/onnxruntime-node/bin/napi-v6/linux/x64")
    env = os.environ.copy()
    env["ORT_DYLIB_PATH"] = f"{ort_gpu}/libonnxruntime.so.1"
    env["LD_LIBRARY_PATH"] = f"{ort_gpu}:{libs}:{env.get('LD_LIBRARY_PATH','')}"
    return env

def cpu_env():
    env = os.environ.copy()
    ort_cpu = ("/home/trung/miniconda3/envs/label/lib/python3.12/"
               "site-packages/onnxruntime/capi/libonnxruntime.so.1.26.0")
    env["ORT_DYLIB_PATH"] = ort_cpu
    env.pop("LD_LIBRARY_PATH", None)
    return env

def hardware_info():
    gpu = "N/A"
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        gpu = r.stdout.strip().splitlines()[0] if r.returncode == 0 else "N/A"
    except Exception:
        pass
    try:
        cores = int(subprocess.check_output(["nproc"]).strip())
        ram = int(subprocess.check_output("free -m | awk '/Mem/{print $2}'", shell=True).strip()) // 1024
    except Exception:
        cores, ram = None, None
    return {"gpu": gpu, "cpu_cores": cores, "ram_gb": ram}


# ── Baseline 1: VisionServe Go HTTP ──────────────────────────────────────────

def bench_visionserve_go(device="cpu", quick=False):
    import urllib.request, urllib.parse
    n = 10 if quick else N_RUNS
    env = gpu_env() if device == "gpu" else cpu_env()
    results = {}

    for model, prompt in [("rf-detr", None), ("grounding-dino", "cat. remote.")]:
        print(f"  go_http {model} {device} ...", flush=True)
        # cold start: launch server, wait for first response
        t0 = time.perf_counter()
        srv = subprocess.Popen(
            [BIN, "serve", "--models", MODELS, "--addr", ":11435"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # poll health
        for _ in range(60):
            try:
                urllib.request.urlopen("http://localhost:11435/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        # first predict (triggers model load)
        parts = [("model", model), ("image", open(IMG, "rb").read())]
        if prompt:
            parts.append(("prompt", prompt))
        _do_predict_multipart("http://localhost:11435/api/predict", model, IMG, prompt)
        cold_ms = (time.perf_counter() - t0) * 1000

        # warmup
        for _ in range(N_WARMUP):
            _do_predict_multipart("http://localhost:11435/api/predict", model, IMG, prompt)

        # timed runs
        times, server_times = [], []
        rss = get_rss(srv.pid)
        vram = get_vram(srv.pid) if device == "gpu" else None
        for _ in range(n):
            t = time.perf_counter()
            r = _do_predict_multipart("http://localhost:11435/api/predict", model, IMG, prompt)
            times.append((time.perf_counter() - t) * 1000)
            if r and "duration_ms" in r:
                server_times.append(r["duration_ms"])

        srv.terminate(); srv.wait(5)
        time.sleep(1)

        res = pstats(times)
        res["cold_start_ms"] = round(cold_ms, 1)
        res["rss_mb"] = rss
        res["vram_mb"] = vram
        if server_times:
            res["server_p50_ms"] = round(statistics.median(sorted(server_times)), 1)
        results[model] = res

    return results


def _do_predict_multipart(url, model, img_path, prompt=None):
    import urllib.request, mimetypes
    boundary = "BENCH_BOUNDARY_XYZ"
    body = b""
    for name, val in [("model", model)] + ([("prompt", prompt)] if prompt else []):
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n".encode()
    with open(img_path, "rb") as f:
        img_data = f.read()
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"img.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode()
    body += img_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ── Baseline 2: Python raw ORT ───────────────────────────────────────────────

PYTHON_ORT_SCRIPT = """
import time, json, statistics, sys, os
import numpy as np
import onnxruntime as ort
from PIL import Image

model_name = sys.argv[1]  # rf-detr or grounding-dino
device = sys.argv[2]      # cpu or gpu
n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 30

providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if device=="gpu" else ["CPUExecutionProvider"]

IMG = "/tmp/cats.jpg"
img = Image.open(IMG).convert("RGB")
W, H = img.size

def preprocess_rfdetr(img):
    import numpy as np
    from PIL import Image
    img_r = img.resize((560,560), Image.BILINEAR)
    x = np.array(img_r, dtype=np.float32)/255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    x = (x - mean) / std
    return x.transpose(2,0,1)[None]  # [1,3,560,560]

def preprocess_gdino(img):
    img_r = img.resize((800,800), Image.BILINEAR)
    x = np.array(img_r, dtype=np.float32)/255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    x = ((x - mean)/std).transpose(2,0,1)[None].astype(np.float32)
    pixel_mask = np.ones((1,800,800), dtype=np.int64)
    # simple tokenizer: cat.=>[101,4937,1012,102]  remote.=>[101,6556,1012,102]
    ids = np.array([[101,4937,1012,6556,1012,102]], dtype=np.int64)
    mask = np.ones_like(ids)
    types = np.zeros_like(ids)
    return x, pixel_mask, ids, mask, types

if model_name == "rf-detr":
    model_path = "models/rf-detr/rf-detr-base-real.onnx"
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, providers=providers)
    inp = preprocess_rfdetr(img)
    _ = sess.run(None, {"input": inp})
    cold_ms = (time.perf_counter()-t0)*1000
    for _ in range(5): sess.run(None, {"input": inp})
    times=[]
    for _ in range(n_runs):
        t=time.perf_counter(); sess.run(None,{"input":inp}); times.append((time.perf_counter()-t)*1000)
elif model_name == "grounding-dino":
    model_path = "models/grounding-dino/model.onnx"
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, providers=providers)
    pv,pm,ids,msk,typ = preprocess_gdino(img)
    _ = sess.run(None,{"pixel_values":pv,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ})
    cold_ms = (time.perf_counter()-t0)*1000
    for _ in range(5): sess.run(None,{"pixel_values":pv,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ})
    times=[]
    for _ in range(n_runs):
        t=time.perf_counter()
        sess.run(None,{"pixel_values":pv,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ})
        times.append((time.perf_counter()-t)*1000)

import psutil
rss = psutil.Process().memory_info().rss//(1024*1024) if 'psutil' in sys.modules else None
try: import psutil; rss=psutil.Process().memory_info().rss//(1024*1024)
except: pass
times.sort()
n=len(times)
res={"cold_start_ms":round(cold_ms,1),"p50_ms":round(statistics.median(times),1),
     "p95_ms":round(times[int(n*0.95)],1),"p99_ms":round(times[min(int(n*0.99),n-1)],1),
     "mean_ms":round(statistics.mean(times),1),"throughput_rps":round(1000/statistics.mean(times),2),
     "rss_mb":rss,"n_runs":n,"providers_used":ort.get_available_providers()}
print(json.dumps(res))
"""

def bench_python_ort(device="cpu", quick=False):
    n = 10 if quick else N_RUNS
    script = "/tmp/bench_ort.py"
    with open(script, "w") as f:
        f.write(PYTHON_ORT_SCRIPT)
    results = {}
    env = os.environ.copy()
    if device == "gpu":
        import glob
        nv_base = "/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia"
        libs = ":".join(glob.glob(f"{nv_base}/*/lib"))
        env["LD_LIBRARY_PATH"] = libs + ":" + env.get("LD_LIBRARY_PATH", "")
    for model in ["rf-detr", "grounding-dino"]:
        print(f"  python_ort {model} {device} ...", flush=True)
        try:
            r = subprocess.run([PY, script, model, device, str(n)],
                capture_output=True, text=True, timeout=300, cwd=str(ROOT), env=env)
            results[model] = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:
            results[model] = {"error": str(e)}
    return results


# ── Baseline 3: FastAPI + ORT ─────────────────────────────────────────────────

FASTAPI_SERVER = """
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time, numpy as np, onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
import io

app = FastAPI()
sessions = {}

MODELS = {
    "rf-detr": "models/rf-detr/rf-detr-base-real.onnx",
    "grounding-dino": "models/grounding-dino/model.onnx",
}
PROVIDERS = ["CUDAExecutionProvider","CPUExecutionProvider"] if os.environ.get("USE_CUDA") else ["CPUExecutionProvider"]

def get_session(model):
    if model not in sessions:
        sessions[model] = ort.InferenceSession(MODELS[model], providers=PROVIDERS)
    return sessions[model]

@app.post("/predict")
async def predict(model: str = Form(...), image: UploadFile = File(...), prompt: str = Form(None)):
    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    sess = get_session(model)
    if model == "rf-detr":
        x = np.array(img.resize((560,560)))/255.0
        mean,std = [0.485,0.456,0.406],[0.229,0.224,0.225]
        x = ((x-mean)/std).transpose(2,0,1)[None].astype(np.float32)
        outs = sess.run(None, {"input": x})
    else:
        x = np.array(img.resize((800,800)))/255.0
        mean,std=[0.485,0.456,0.406],[0.229,0.224,0.225]
        x=((x-mean)/std).transpose(2,0,1)[None].astype(np.float32)
        pm=np.ones((1,800,800),dtype=np.int64)
        ids=np.array([[101,4937,1012,6556,1012,102]],dtype=np.int64)
        msk=np.ones_like(ids); typ=np.zeros_like(ids)
        outs=sess.run(None,{"pixel_values":x,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ})
    return {"duration_ms": round((time.perf_counter()-t0)*1000, 1)}
"""

def bench_fastapi(device="cpu", quick=False):
    n = 10 if quick else N_RUNS
    # write server
    srv_path = "/tmp/fastapi_bench_server.py"
    with open(srv_path, "w") as f:
        f.write(FASTAPI_SERVER)
    env = os.environ.copy()
    if device == "gpu":
        import glob
        nv_base="/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia"
        libs=":".join(glob.glob(f"{nv_base}/*/lib"))
        env["LD_LIBRARY_PATH"]=libs+":"+env.get("LD_LIBRARY_PATH","")
        env["USE_CUDA"]="1"
    else:
        env.pop("USE_CUDA", None)

    results = {}
    for model, prompt in [("rf-detr",None),("grounding-dino","cat. remote.")]:
        print(f"  fastapi {model} {device} ...", flush=True)
        # check uvicorn available
        r = subprocess.run([PY,"-c","import uvicorn"], capture_output=True)
        if r.returncode != 0:
            results[model] = {"error": "uvicorn not installed"}
            continue
        srv = subprocess.Popen(
            [PY,"-m","uvicorn","fastapi_bench_server:app","--port","11436","--workers","1","--log-level","error"],
            cwd="/tmp", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import urllib.request
        for _ in range(30):
            try: urllib.request.urlopen("http://localhost:11436/predict",timeout=1); break
            except: time.sleep(0.5)
        # first request = cold
        t0=time.perf_counter()
        _do_predict_multipart("http://localhost:11436/predict",model,IMG,prompt)
        cold_ms=(time.perf_counter()-t0)*1000
        for _ in range(N_WARMUP):
            _do_predict_multipart("http://localhost:11436/predict",model,IMG,prompt)
        times,srv_times=[],[]
        rss=get_rss(srv.pid); vram=get_vram(srv.pid) if device=="gpu" else None
        for _ in range(n):
            t=time.perf_counter()
            r=_do_predict_multipart("http://localhost:11436/predict",model,IMG,prompt)
            times.append((time.perf_counter()-t)*1000)
            if r and "duration_ms" in r: srv_times.append(r["duration_ms"])
        srv.terminate(); srv.wait(5); time.sleep(1)
        res=pstats(times)
        res["cold_start_ms"]=round(cold_ms,1)
        res["rss_mb"]=rss; res["vram_mb"]=vram
        if srv_times: res["server_p50_ms"]=round(statistics.median(sorted(srv_times)),1)
        results[model]=res
    return results


# ── Baseline 4: C++ raw ORT ───────────────────────────────────────────────────

def bench_cpp_ort():
    cpp_bin = "/tmp/bench_cpp_ort"
    ort_lib = "/tmp/ort_cpp/lib"
    if not os.path.exists(cpp_bin):
        return {"error": "C++ binary not compiled"}
    results = {}
    for device in ["cpu", "gpu"]:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ort_lib
        if device == "gpu":
            nv = "/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia/cudnn/lib"
            env["LD_LIBRARY_PATH"] += ":" + nv
        try:
            r = subprocess.run(
                [cpp_bin, RF_REAL, device if device=="cuda" else ""],
                capture_output=True, text=True, timeout=120, env=env)
            if r.returncode == 0:
                line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")][-1]
                d = json.loads(line)
                results[device] = d
            else:
                results[device] = {"error": f"exit {r.returncode}: {r.stderr[:200]}"}
        except Exception as e:
            results[device] = {"error": str(e)}
    return {"rf-detr": results,
            "note": "pure ORT engine only — random float32 input, no preprocess/postprocess"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--quick", action="store_true", help="10 runs instead of 30")
    parser.add_argument("--skip", nargs="*", default=[], help="baselines to skip: go ort fastapi cpp")
    args = parser.parse_args()

    ensure_image()
    hw = hardware_info()
    print(f"Hardware: {hw['gpu']}, {hw['cpu_cores']} cores, {hw['ram_gb']}GB RAM")
    print(f"Quick mode: {args.quick}, Devices: {'GPU' if args.gpu else ''} {'CPU' if args.cpu else ''}")
    print()

    output = {"hardware": hw, "baselines": {}}

    devices = []
    if args.gpu: devices.append("gpu")
    if args.cpu: devices.append("cpu")

    # VisionServe Go HTTP
    if "go" not in args.skip:
        print("[1/4] VisionServe Go HTTP")
        vs = {}
        for dev in devices:
            print(f"  device={dev}")
            vs[dev] = bench_visionserve_go(dev, args.quick)
        output["baselines"]["visionserve_go_http"] = vs

    # Python raw ORT
    if "ort" not in args.skip:
        print("[2/4] Python raw ORT")
        port = {}
        for dev in devices:
            print(f"  device={dev}")
            port[dev] = bench_python_ort(dev, args.quick)
        output["baselines"]["python_ort_direct"] = port

    # FastAPI + ORT
    if "fastapi" not in args.skip:
        print("[3/4] Python FastAPI + ORT")
        fa = {}
        for dev in devices:
            print(f"  device={dev}")
            fa[dev] = bench_fastapi(dev, args.quick)
        output["baselines"]["python_fastapi_http"] = fa

    # C++ ORT
    if "cpp" not in args.skip:
        print("[4/4] C++ raw ORT")
        output["baselines"]["cpp_ort_direct"] = bench_cpp_ort()

    # save
    out_path = RESULTS / "all_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    return output


if __name__ == "__main__":
    main()
