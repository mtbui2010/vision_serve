"""FastAPI + onnxruntime-python baseline (W1 cleanest engine control).

Loads the SAME ONNX file VisionServe serves, with the CUDA EP, and exposes
``POST /api/predict`` in VisionServe's request/response shape. Because the runtime (ONNX
Runtime) is held constant, any latency/throughput delta vs VisionServe-Go is attributable to
the *serving layer* (Python/uvicorn + GIL vs Go goroutines), which is exactly the W1 claim.

Config via environment variables (so it composes with uvicorn/gunicorn):

    ONNX_PATH      (required) path to the .onnx file (same one VisionServe loads)
    MODEL_NAME     reported back in Result.model (default: basename of ONNX dir)
    TASK           one of classification/detection/segmentation/... (default: classification)
    INPUT_W/INPUT_H/LETTERBOX  preprocess overrides (default 224/224/false)
    LABELS_PATH    optional newline-delimited label file (e.g. imagenet1k.txt) for classification
    EP             onnxruntime providers, comma-separated
                   (default: CUDAExecutionProvider,CPUExecutionProvider)
    INTRA_OP_THREADS / SESSION_POOL  see notes below

Run:

    ONNX_PATH=~/.visionserve/models/mobilenet-v3/model.onnx MODEL_NAME=mobilenet-v3 \
        uvicorn eval.baselines.fastapi_ort:app --host 0.0.0.0 --port 8001

Concurrency note (relevant to W4): a single ``onnxruntime.InferenceSession`` is internally
thread-safe but a CUDA session still serializes the GPU launch; to mirror VisionServe's
``SessionPool`` we keep a small pool of sessions guarded by a queue (SESSION_POOL, default 1).
This makes the GIL-vs-goroutine comparison apples-to-apples (both have a pool knob).
"""

from __future__ import annotations

import base64
import os
import queue
import time
from typing import Any, Optional

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "onnxruntime is required for fastapi_ort.py; `pip install onnxruntime-gpu==1.20.1`"
    ) from exc

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from eval.baselines.preprocess import PreprocessConfig, preprocess


def _load_labels(path: Optional[str]) -> Optional[list[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


class OrtPool:
    """A bounded pool of identical InferenceSessions, mirroring VisionServe's SessionPool."""

    def __init__(self, onnx_path: str, providers: list[str], pool_size: int,
                 intra_op_threads: int) -> None:
        so = ort.SessionOptions()
        if intra_op_threads > 0:
            so.intra_op_num_threads = intra_op_threads
        self._sessions: "queue.Queue[ort.InferenceSession]" = queue.Queue()
        for _ in range(max(1, pool_size)):
            self._sessions.put(ort.InferenceSession(onnx_path, sess_options=so,
                                                    providers=providers))
        probe = self._sessions.queue[0]
        self.input_name = probe.get_inputs()[0].name
        self.output_names = [o.name for o in probe.get_outputs()]
        self.active_providers = probe.get_providers()

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        sess = self._sessions.get()
        try:
            return sess.run(self.output_names, feed)
        finally:
            self._sessions.put(sess)


# ---- App setup (read config once at import) -----------------------------------------------

ONNX_PATH = os.environ.get("ONNX_PATH")
if not ONNX_PATH:
    raise RuntimeError("set ONNX_PATH to the .onnx file (the same one VisionServe serves)")

MODEL_NAME = os.environ.get("MODEL_NAME") or os.path.basename(os.path.dirname(ONNX_PATH))
TASK = os.environ.get("TASK", "classification")
EP = os.environ.get("EP", "CUDAExecutionProvider,CPUExecutionProvider").split(",")
SESSION_POOL = int(os.environ.get("SESSION_POOL", "1"))
INTRA_OP_THREADS = int(os.environ.get("INTRA_OP_THREADS", "0"))
LABELS = _load_labels(os.environ.get("LABELS_PATH"))

_CFG = PreprocessConfig(
    width=int(os.environ.get("INPUT_W", "224")),
    height=int(os.environ.get("INPUT_H", "224")),
    letterbox=os.environ.get("LETTERBOX", "false").lower() in ("1", "true", "yes"),
)

_POOL = OrtPool(ONNX_PATH, EP, SESSION_POOL, INTRA_OP_THREADS)

app = FastAPI(title="VisionServe baseline: FastAPI + onnxruntime")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _postprocess(outputs: list[np.ndarray], top_k: int = 5) -> dict[str, Any]:
    """Minimal, task-aware postprocess.

    Classification is implemented (sufficient for the mobilenet-v3 W1 cell and ImageNet Top-1).
    Detection/segmentation decode is model-specific and is intentionally NOT reimplemented here
    — for those tasks the W1 latency cell still works (we time the full request path), but the
    detection payload is left empty with a TODO so we never emit fabricated boxes.
    """
    if TASK == "classification":
        logits = np.asarray(outputs[0]).reshape(-1)
        probs = _softmax(logits)
        idx = np.argsort(-probs)[:top_k]
        cls = []
        for i in idx:
            label = LABELS[i] if LABELS and i < len(LABELS) else str(int(i))
            cls.append({"class": label, "conf": float(probs[i])})
        return {"classifications": cls}
    # TODO: requires per-model decode (RF-DETR NMS-free query decode, MobileSAM RLE, GDINO
    # token->label). Mirror internal/models/<name> postprocess before reporting boxes/masks.
    return {"detections": [], "masks": []}


def _predict(image_bytes: bytes) -> dict[str, Any]:
    tensor, _meta = preprocess(image_bytes, _CFG)
    t0 = time.perf_counter()
    outputs = _POOL.run({_POOL.input_name: tensor})
    infer_ms = (time.perf_counter() - t0) * 1000.0
    result: dict[str, Any] = {
        "task": TASK,
        "model": MODEL_NAME,
        # report the EP actually in use (first provider) for provenance parity with VisionServe
        "device": _POOL.active_providers[0] if _POOL.active_providers else "",
        "duration_ms": infer_ms,
    }
    result.update(_postprocess(outputs))
    return result


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(
    request: Request,
    model: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
) -> dict[str, Any]:
    """Accept both the JSON (image_base64) and multipart (image file) branches."""
    ct = request.headers.get("content-type", "")
    if ct.startswith("application/json"):
        body = await request.json()
        b64 = body.get("image_base64")
        if not b64:
            raise HTTPException(status_code=400, detail="image_base64 required")
        try:
            raw = base64.b64decode(b64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid image_base64: {exc}") from exc
        return _predict(raw)

    if image is None:
        raise HTTPException(status_code=400, detail="missing 'image' file or image_base64")
    raw = await image.read()
    return _predict(raw)
