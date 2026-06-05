"""FastAPI + PyTorch baseline skeleton (W1: the common real-world stack).

This represents the "FastAPI + model.forward()" stack many teams actually deploy. Unlike
``fastapi_ort.py`` it does NOT share the ONNX runtime, so it answers a different W1 question:
"how does VisionServe compare to a typical PyTorch serving stack" (vs the engine-controlled
"same ONNX" comparison). Keep both; the paper should label which baseline shares the ONNX file.

The model load is deliberately left as a TODO because the correct checkpoint + preprocessing is
model-specific and must be supplied per model — fabricating it would produce meaningless numbers.

Run (after filling the TODO):

    MODEL_NAME=mobilenet-v3 TORCH_CKPT=/path/to/ckpt.pth \
        uvicorn eval.baselines.fastapi_torch:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

try:
    import torch  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("torch is required for fastapi_torch.py; `pip install torch`") from exc

MODEL_NAME = os.environ.get("MODEL_NAME", "unknown")
TASK = os.environ.get("TASK", "classification")
TORCH_CKPT = os.environ.get("TORCH_CKPT", "")
DEVICE = os.environ.get("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="VisionServe baseline: FastAPI + PyTorch")


def _load_model() -> Any:
    """Load and return an eval-mode torch model on ``DEVICE``.

    TODO: requires the original PyTorch checkpoint + architecture for ``MODEL_NAME``.
    Example shape (mobilenet-v3):

        import torchvision
        m = torchvision.models.mobilenet_v3_large(weights=None)
        m.load_state_dict(torch.load(TORCH_CKPT, map_location=DEVICE))
        return m.eval().to(DEVICE)

    Each model needs its own loader (RF-DETR, MobileSAM, GroundingDINO are not in torchvision).
    Do NOT substitute a random-weight model — that would yield fabricated accuracy/latency.
    """
    raise NotImplementedError(
        "fastapi_torch._load_model is a TODO: supply the real checkpoint/loader for "
        f"MODEL_NAME={MODEL_NAME!r} (TORCH_CKPT={TORCH_CKPT!r})."
    )


def _preprocess_torch(image_bytes: bytes):
    """TODO: requires the model's torchvision transform (resize/normalize) matching its export."""
    raise NotImplementedError(
        "fastapi_torch._preprocess_torch is a TODO: implement the model-specific transform."
    )


def _postprocess_torch(output: Any) -> dict[str, Any]:
    """TODO: requires per-task decode (classification softmax / detection / segmentation)."""
    raise NotImplementedError(
        "fastapi_torch._postprocess_torch is a TODO: implement the per-task decode."
    )


# Lazy singleton so the module imports (and `py_compile`/lint) even before the TODO is filled.
_MODEL: Optional[Any] = None


def _model() -> Any:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _predict(image_bytes: bytes) -> dict[str, Any]:
    inp = _preprocess_torch(image_bytes)
    if hasattr(inp, "to"):
        inp = inp.to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = _model()(inp)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0
    result: dict[str, Any] = {
        "task": TASK,
        "model": MODEL_NAME,
        "device": DEVICE,
        "duration_ms": infer_ms,
    }
    result.update(_postprocess_torch(out))
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
    ct = request.headers.get("content-type", "")
    if ct.startswith("application/json"):
        body = await request.json()
        b64 = body.get("image_base64")
        if not b64:
            raise HTTPException(status_code=400, detail="image_base64 required")
        raw = base64.b64decode(b64)
        return _predict(raw)
    if image is None:
        raise HTTPException(status_code=400, detail="missing 'image' file or image_base64")
    raw = await image.read()
    return _predict(raw)
