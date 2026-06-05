"""Task 2: ndarray round-trip latency through the server, PNG (old client) vs JPEG (new client).
Shows the win from encoding a numpy frame as JPEG instead of PNG before sending."""
import base64, io, json, statistics, sys, time, urllib.request
import numpy as np
from PIL import Image

URL = "http://localhost:11436/api/predict"
MODEL = "mobilenet-v3"
def post(b64):
    body = json.dumps({"model": MODEL, "image_base64": b64}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter(); r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return (time.perf_counter()-t0)*1000.0, r
# load a real photographic frame as ndarray (what a camera pipeline would have in RAM)
arr = np.asarray(Image.open("test/testdata/sample.jpg").convert("RGB"))
def enc(fmt, **kw):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format=fmt, **kw); return buf.getvalue()
out = {}
for name, raw in [("PNG (old client)", enc("PNG")), ("JPEG-q92 (new client)", enc("JPEG", quality=92))]:
    b64 = base64.b64encode(raw).decode()
    for _ in range(10): post(b64)  # warmup
    lats = [post(b64)[0] for _ in range(60)]
    out[name] = {"payload_kb": round(len(raw)/1024,1), "p50_ms": round(statistics.median(lats),2),
                 "mean_ms": round(statistics.mean(lats),2), "min_ms": round(min(lats),2)}
    print(f"{name:24} payload={out[name]['payload_kb']:>7} KB  p50={out[name]['p50_ms']:>6} ms  min={out[name]['min_ms']:>6} ms")
json.dump(out, open("eval/results/client_format_bench.json","w"), indent=2)
print("wrote eval/results/client_format_bench.json")
