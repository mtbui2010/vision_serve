"""
VisionServe benchmark — measures real inference latency across baselines.
Run: python3 benchmarks/bench.py
Results: benchmarks/results/all_benchmarks.json
"""
import json, os, signal, statistics, subprocess, sys, time, threading, glob
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "benchmarks" / "results"
OUT.mkdir(parents=True, exist_ok=True)

PY    = "/home/trung/miniconda3/envs/label/bin/python3"
BIN   = str(ROOT / "bin" / "visionserve")
IMG   = "/tmp/cats.jpg"
MODELS_DIR = str(ROOT / "models")

RF_REAL = str(ROOT / "models/rf-detr/rf-detr-base-real.onnx")
GDINO   = str(ROOT / "models/grounding-dino/model.onnx")
VOCAB   = str(ROOT / "models/grounding-dino/vocab.txt")

N_WARM = 5
N_RUNS = 30

# ─── helpers ────────────────────────────────────────────────────────────────

def ensure_image():
    if not os.path.exists(IMG) or os.path.getsize(IMG) < 10000:
        subprocess.run(["curl","-sL",
            "http://images.cocodataset.org/val2017/000000039769.jpg","-o",IMG], check=True)

def p(times):
    s = sorted(times); n = len(s)
    return dict(p50=round(statistics.median(s),1), p95=round(s[int(n*.95)],1),
                p99=round(s[min(int(n*.99),n-1)],1), mean=round(statistics.mean(s),1),
                rps=round(1000/statistics.mean(s),2), n=n)

def rss(pid=None):
    try:
        import psutil; return psutil.Process(pid).memory_info().rss//(1024*1024)
    except Exception: return None

def vram(pid=None):
    try:
        r=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory",
            "--format=csv,noheader"],capture_output=True,text=True,timeout=5)
        for line in r.stdout.strip().splitlines():
            p2,m=line.split(",")
            if pid is None or int(p2.strip())==pid:
                return int(m.strip().split()[0])
    except Exception: return None
    return None

def gpu_ld():
    """LD_LIBRARY_PATH for CUDA EP (cuDNN from label env wheels + node ORT)."""
    ort_dir="/home/trung/trung_workdir/vision_forge/frontend/node_modules/onnxruntime-node/bin/napi-v6/linux/x64"
    nv=":".join(glob.glob("/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia/*/lib"))
    return f"{ort_dir}:{nv}"

def gpu_ort():
    return ("/home/trung/trung_workdir/vision_forge/frontend/node_modules/"
            "onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime.so.1")

def cpu_ort():
    return ("/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/"
            "onnxruntime/capi/libonnxruntime.so.1.26.0")

def hw():
    try:
        g=subprocess.check_output(["nvidia-smi","-L"],text=True).strip().splitlines()[0]
    except Exception: g="N/A"
    try: cores=int(subprocess.check_output(["nproc"]).strip())
    except Exception: cores=None
    try: ram=int(subprocess.check_output("free -m|awk '/Mem/{print $2}'",shell=True).strip())//1024
    except Exception: ram=None
    return dict(gpu=g,cpu_cores=cores,ram_gb=ram)

# ─── HTTP helpers ────────────────────────────────────────────────────────────

def health_wait(url, timeout=60):
    import urllib.request
    for _ in range(timeout*2):
        try:
            urllib.request.urlopen(url, timeout=1); return True
        except Exception: time.sleep(0.5)
    return False

def predict_http(url, model, img_path, prompt=None, timeout=300):
    import urllib.request
    boundary="BENCHBND"
    body=b""
    for name,val in ([("model",model)]+([("prompt",prompt)] if prompt else [])):
        body+=f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n".encode()
    with open(img_path,"rb") as f: img=f.read()
    body+=(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"i.jpg\"\r\n"
           "Content-Type: image/jpeg\r\n\r\n").encode()+img+f"\r\n--{boundary}--\r\n".encode()
    req=urllib.request.Request(url,data=body,
        headers={"Content-Type":f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return json.loads(r.read())

# ─── Baseline 1: VisionServe Go HTTP ────────────────────────────────────────

def bench_go(device):
    env = os.environ.copy()
    if device=="gpu":
        env["ORT_DYLIB_PATH"]=gpu_ort(); env["LD_LIBRARY_PATH"]=gpu_ld()
    else:
        env["ORT_DYLIB_PATH"]=cpu_ort(); env.pop("LD_LIBRARY_PATH",None)

    res={}
    for model,prompt in [("rf-detr",None),("grounding-dino","cat. remote.")]:
        print(f"  go/{device}/{model}...", flush=True)
        # cold start: server launch → first predict response
        t0=time.perf_counter()
        srv=subprocess.Popen([BIN,"serve","--models",MODELS_DIR,"--addr",":11435"],
            env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if not health_wait("http://localhost:11435/api/health"):
            srv.terminate(); res[model]={"error":"server didn't start"}; continue
        try: predict_http("http://localhost:11435/api/predict",model,IMG,prompt,timeout=300)
        except Exception as e: srv.terminate(); res[model]={"error":str(e)}; continue
        cold_ms=(time.perf_counter()-t0)*1000

        for _ in range(N_WARM):
            try: predict_http("http://localhost:11435/api/predict",model,IMG,prompt,timeout=120)
            except Exception: pass

        pid=srv.pid; rss_mb=rss(pid); vram_mb=vram(pid) if device=="gpu" else None
        times=[]; srv_times=[]
        for _ in range(N_RUNS):
            t=time.perf_counter()
            r=predict_http("http://localhost:11435/api/predict",model,IMG,prompt,timeout=120)
            times.append((time.perf_counter()-t)*1000)
            if "duration_ms" in r: srv_times.append(r["duration_ms"])

        srv.terminate(); srv.wait(5); time.sleep(1)
        stat=p(times); stat.update(cold_ms=round(cold_ms,1),rss_mb=rss_mb,vram_mb=vram_mb)
        if srv_times: stat["srv_p50"]=round(statistics.median(sorted(srv_times)),1)
        res[model]=stat
    return res

# ─── Baseline 2: Python raw ORT ─────────────────────────────────────────────

PY_ORT_CODE=r"""
import sys,time,json,statistics,os,glob
import numpy as np
import onnxruntime as ort

model=sys.argv[1]; device=sys.argv[2]; n=int(sys.argv[3] if len(sys.argv)>3 else 30)
img_path="/tmp/cats.jpg"

if device=="gpu":
    providers=["CUDAExecutionProvider","CPUExecutionProvider"]
else:
    providers=["CPUExecutionProvider"]

from PIL import Image
img=Image.open(img_path).convert("RGB")

def pre_rfdetr(img):
    x=np.array(img.resize((560,560)),dtype=np.float32)/255.0
    x=(x-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    return x.transpose(2,0,1)[None]

def pre_gdino(img):
    x=np.array(img.resize((800,800)),dtype=np.float32)/255.0
    x=((x-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
    pm=np.ones((1,800,800),dtype=np.int64)
    ids=np.array([[101,4937,1012,6556,1012,102]],dtype=np.int64)
    msk=np.ones_like(ids); typ=np.zeros_like(ids)
    return x,pm,ids,msk,typ

t0=time.perf_counter()
if model=="rf-detr":
    sess=ort.InferenceSession("models/rf-detr/rf-detr-base-real.onnx",providers=providers)
    inp=pre_rfdetr(img)
    _=sess.run(None,{"input":inp})
    cold=(time.perf_counter()-t0)*1000
    for _ in range(5): sess.run(None,{"input":inp})
    times=[]
    for _ in range(n):
        t=time.perf_counter(); sess.run(None,{"input":inp}); times.append((time.perf_counter()-t)*1000)
elif model=="grounding-dino":
    sess=ort.InferenceSession("models/grounding-dino/model.onnx",providers=providers)
    pv,pm,ids,msk,typ=pre_gdino(img)
    feed={"pixel_values":pv,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ}
    _=sess.run(None,feed)
    cold=(time.perf_counter()-t0)*1000
    for _ in range(5): sess.run(None,feed)
    times=[]
    for _ in range(n):
        t=time.perf_counter(); sess.run(None,feed); times.append((time.perf_counter()-t)*1000)
else:
    print(json.dumps({"error":"unknown model"})); sys.exit(0)

times.sort(); ns=len(times)
try: import psutil; rss2=psutil.Process().memory_info().rss//(1024*1024)
except: rss2=None
avail=ort.get_available_providers()
print(json.dumps({"cold_ms":round(cold,1),"p50":round(statistics.median(times),1),
    "p95":round(times[int(ns*.95)],1),"p99":round(times[min(int(ns*.99),ns-1)],1),
    "mean":round(statistics.mean(times),1),"rps":round(1000/statistics.mean(times),2),
    "rss_mb":rss2,"providers":avail,"n":ns}))
"""

def bench_python_ort(device):
    script="/tmp/_bench_ort.py"
    with open(script,"w") as f: f.write(PY_ORT_CODE)
    env=os.environ.copy()
    if device=="gpu":
        nv=":".join(glob.glob("/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia/*/lib"))
        env["LD_LIBRARY_PATH"]=nv+":"+env.get("LD_LIBRARY_PATH","")
    res={}
    for model in ["rf-detr","grounding-dino"]:
        print(f"  pyort/{device}/{model}...", flush=True)
        try:
            r=subprocess.run([PY,script,model,device,str(N_RUNS)],
                capture_output=True,text=True,timeout=600,
                cwd=str(ROOT),env=env)
            lines=[l for l in r.stdout.strip().splitlines() if l.startswith("{")]
            if not lines: res[model]={"error":r.stderr[-300:] or "no output"}
            else: res[model]=json.loads(lines[-1])
        except Exception as e: res[model]={"error":str(e)}
    return res

# ─── Baseline 3: FastAPI + ORT ───────────────────────────────────────────────

FA_SERVER=r"""
import sys,os,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+"/..")
import numpy as np,onnxruntime as ort
from fastapi import FastAPI,UploadFile,File,Form
from PIL import Image
import io

app=FastAPI()
_sessions={}
PROVIDERS=["CUDAExecutionProvider","CPUExecutionProvider"] if os.environ.get("USE_CUDA") else ["CPUExecutionProvider"]

def sess(model):
    if model not in _sessions:
        paths={"rf-detr":"models/rf-detr/rf-detr-base-real.onnx",
               "grounding-dino":"models/grounding-dino/model.onnx"}
        _sessions[model]=ort.InferenceSession(paths[model],providers=PROVIDERS)
    return _sessions[model]

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict")
async def predict(model:str=Form(...),image:UploadFile=File(...),prompt:str=Form(None)):
    t0=time.perf_counter()
    img=Image.open(io.BytesIO(await image.read())).convert("RGB")
    s=sess(model)
    if model=="rf-detr":
        x=np.array(img.resize((560,560)),dtype=np.float32)/255.0
        x=((x-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None]
        s.run(None,{"input":x})
    else:
        x=np.array(img.resize((800,800)),dtype=np.float32)/255.0
        x=((x-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
        pm=np.ones((1,800,800),dtype=np.int64)
        ids=np.array([[101,4937,1012,6556,1012,102]],dtype=np.int64)
        msk=np.ones_like(ids);typ=np.zeros_like(ids)
        s.run(None,{"pixel_values":x,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ})
    return {"duration_ms":round((time.perf_counter()-t0)*1000,1)}
"""

def bench_fastapi(device):
    srv_path=str(ROOT/"benchmarks"/"_fa_server.py")
    with open(srv_path,"w") as f: f.write(FA_SERVER)
    env=os.environ.copy()
    if device=="gpu":
        nv=":".join(glob.glob("/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia/*/lib"))
        env["LD_LIBRARY_PATH"]=nv+":"+env.get("LD_LIBRARY_PATH","")
        env["USE_CUDA"]="1"
    else: env.pop("USE_CUDA",None)

    res={}
    for model,prompt in [("rf-detr",None),("grounding-dino","cat. remote.")]:
        print(f"  fastapi/{device}/{model}...", flush=True)
        srv=subprocess.Popen(
            [PY,"-m","uvicorn","benchmarks._fa_server:app","--port","11436","--workers","1","--log-level","error"],
            cwd=str(ROOT),env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # wait for health (not /predict which needs POST)
        if not health_wait("http://localhost:11436/health", timeout=30):
            srv.terminate(); res[model]={"error":"uvicorn didn't start"}; continue

        # cold start: first request includes model load
        t0=time.perf_counter()
        try: predict_http("http://localhost:11436/predict",model,IMG,prompt,timeout=600)
        except Exception as e: srv.terminate(); res[model]={"error":f"cold: {e}"}; continue
        cold_ms=(time.perf_counter()-t0)*1000

        for _ in range(N_WARM):
            try: predict_http("http://localhost:11436/predict",model,IMG,prompt,timeout=120)
            except Exception: pass

        pid=srv.pid; rss_mb=rss(pid); vram_mb=vram(pid) if device=="gpu" else None
        times=[]; srv_times=[]
        for _ in range(N_RUNS):
            t=time.perf_counter()
            try:
                r=predict_http("http://localhost:11436/predict",model,IMG,prompt,timeout=120)
                times.append((time.perf_counter()-t)*1000)
                if "duration_ms" in r: srv_times.append(r["duration_ms"])
            except Exception: pass

        srv.terminate(); srv.wait(5); time.sleep(1)
        if not times: res[model]={"error":"no successful runs"}; continue
        stat=p(times); stat.update(cold_ms=round(cold_ms,1),rss_mb=rss_mb,vram_mb=vram_mb)
        if srv_times: stat["srv_p50"]=round(statistics.median(sorted(srv_times)),1)
        res[model]=stat
    return res

# ─── Baseline 4: C++ raw ORT ────────────────────────────────────────────────

def bench_cpp():
    cpp_bin="/tmp/bench_cpp_ort"
    ort_lib="/tmp/ort_cpp/lib"
    if not os.path.exists(cpp_bin):
        return {"error":"not compiled","note":"run: make in benchmarks/"}
    res={}
    for device in ["cpu","gpu"]:
        env=os.environ.copy()
        env["LD_LIBRARY_PATH"]=ort_lib
        if device=="gpu":
            nv="/home/trung/miniconda3/envs/label/lib/python3.12/site-packages/nvidia"
            env["LD_LIBRARY_PATH"]+=":"+":".join(glob.glob(f"{nv}/*/lib"))
        try:
            r=subprocess.run([cpp_bin, RF_REAL, "cuda" if device=="gpu" else ""],
                capture_output=True,text=True,timeout=120,env=env)
            lines=[l for l in r.stdout.strip().splitlines() if l.startswith("{")]
            if r.returncode==0 and lines:
                d=json.loads(lines[-1])
                d["note"]="pure ORT engine, random input, no preprocess"
                res[device]=d
            else: res[device]={"error":f"exit {r.returncode}: {r.stderr[:200]}"}
        except Exception as e: res[device]={"error":str(e)}
    return {"rf-detr":res,"note":"ceiling reference — ORT engine only, no preprocess/postprocess"}

# ─── main ────────────────────────────────────────────────────────────────────

if __name__=="__main__":
    ensure_image()
    hardware=hw()
    print(f"Hardware: {hardware['gpu']}, {hardware['cpu_cores']} cores, {hardware['ram_gb']}GB RAM\n")

    results={"hardware":hardware,"baselines":{}}

    for device in ["gpu","cpu"]:
        print(f"\n{'='*50}\nDevice: {device.upper()}\n{'='*50}")

        print("\n[1/3] VisionServe Go HTTP")
        results["baselines"].setdefault("visionserve_go_http",{})[device]=bench_go(device)

        print("\n[2/3] Python raw ORT")
        results["baselines"].setdefault("python_ort_direct",{})[device]=bench_python_ort(device)

        print("\n[3/3] Python FastAPI + ORT")
        results["baselines"].setdefault("python_fastapi_http",{})[device]=bench_fastapi(device)

    print("\n[4] C++ raw ORT (engine only)")
    results["baselines"]["cpp_ort_direct"]=bench_cpp()

    out=OUT/"all_benchmarks.json"
    with open(out,"w") as f: json.dump(results,f,indent=2)
    print(f"\nSaved → {out}")

    # print summary table
    print("\n"+("="*70))
    print(f"{'Model':<20} {'Baseline':<25} {'Device':<6} {'p50ms':>7} {'RPS':>8} {'RSS MB':>8}")
    print("-"*70)
    for bl,devs in results["baselines"].items():
        if bl=="cpp_ort_direct":
            for dev,stat in devs.get("rf-detr",{}).items():
                if isinstance(stat,dict) and "p50_ms" in stat:
                    print(f"{'rf-detr(engine)':<20} {bl:<25} {dev:<6} {stat.get('p50_ms','?'):>7} {stat.get('throughput_rps','?'):>8} {'N/A':>8}")
        else:
            for dev,models in devs.items():
                for model,stat in models.items():
                    if isinstance(stat,dict) and "p50" in stat:
                        print(f"{model:<20} {bl:<25} {dev:<6} {stat['p50']:>7} {stat['rps']:>8} {stat.get('rss_mb','?'):>8}")
    print("="*70)
