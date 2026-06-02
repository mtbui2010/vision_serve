
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
        mean=np.array([0.485,0.456,0.406],dtype=np.float32); std=np.array([0.229,0.224,0.225],dtype=np.float32); x=((x-mean)/std).transpose(2,0,1)[None]
        s.run(None,{"input":x})
    else:
        x=np.array(img.resize((800,800)),dtype=np.float32)/255.0
        mean=np.array([0.485,0.456,0.406],dtype=np.float32); std=np.array([0.229,0.224,0.225],dtype=np.float32); x=((x-mean)/std).transpose(2,0,1)[None].astype(np.float32)
        pm=np.ones((1,800,800),dtype=np.int64)
        ids=np.array([[101,4937,1012,6556,1012,102]],dtype=np.int64)
        msk=np.ones_like(ids);typ=np.zeros_like(ids)
        s.run(None,{"pixel_values":x,"pixel_mask":pm,"input_ids":ids,"attention_mask":msk,"token_type_ids":typ})
    return {"duration_ms":round((time.perf_counter()-t0)*1000,1)}
