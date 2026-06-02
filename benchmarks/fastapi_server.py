#!/usr/bin/env python3
"""
benchmarks/fastapi_server.py — FastAPI + ONNX Runtime inference server.

Equivalent to VisionServe (Go HTTP) but implemented in Python/FastAPI.
Used as a benchmark baseline for HTTP overhead comparison.

Supported models:
  - rf-detr          (detection, Apache-2.0)
  - grounding-dino   (open-vocab detection, Apache-2.0)
  - grounded-sam     (open-vocab detection + segmentation, Apache-2.0)

POST /predict   multipart/form-data
  model  : str   — "rf-detr" | "grounding-dino" | "grounded-sam"
  image  : file  — JPEG/PNG
  prompt : str   — required for grounding-dino and grounded-sam

Response JSON:
  {"detections": [...], "masks": [...], "duration_ms": N}

Lazy model load: first request for a model triggers session creation.
Sessions are cached for the lifetime of the process.

Port: 11436 (do not use 11435 — VisionServe uses that)
Run:
  uvicorn benchmarks.fastapi_server:app --port 11436 --workers 1
  # or from vision_serve root:
  /home/trung/miniconda3/envs/label/bin/python3 -m uvicorn benchmarks.fastapi_server:app --port 11436 --workers 1
"""

import io
import os
import time
import threading
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    "rf-detr": os.path.join(ROOT, "models/rf-detr/rf-detr-base-real.onnx"),
    "grounding-dino": os.path.join(ROOT, "models/grounding-dino/model.onnx"),
    # grounded-sam uses two separate sessions (gdino + sam encoder + sam decoder)
    "grounded-sam-gdino": os.path.join(ROOT, "models/grounding-dino/model.onnx"),
    "grounded-sam-enc":   os.path.join(ROOT, "models/mobile-sam/mobile_sam_encoder.onnx"),
    "grounded-sam-dec":   os.path.join(ROOT, "models/mobile-sam/mobile_sam_decoder_single.onnx"),
}

VOCAB_PATH = os.path.join(ROOT, "models/grounding-dino/vocab.txt")
LABELS_PATH = os.path.join(ROOT, "models/rf-detr/coco91.txt")

# Normalization constants (ImageNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ORT providers: use CUDA if requested via USE_CUDA env var
_USE_CUDA = os.environ.get("USE_CUDA", "0") in ("1", "true", "True", "yes")
ORT_PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if _USE_CUDA
    else ["CPUExecutionProvider"]
)

# ── Global session cache (lazy, thread-safe) ──────────────────────────────────

_sessions: dict = {}
_session_lock = threading.Lock()


def _get_session(key: str) -> ort.InferenceSession:
    """Return a cached ORT session for `key`, creating it on first call."""
    if key not in _sessions:
        with _session_lock:
            if key not in _sessions:  # double-checked locking
                path = MODEL_PATHS.get(key)
                if path is None or not os.path.exists(path):
                    raise RuntimeError(f"Model not found: {key!r} → {path}")
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                _sessions[key] = ort.InferenceSession(
                    path, sess_options=sess_opts, providers=ORT_PROVIDERS
                )
    return _sessions[key]


# ── COCO-91 label loading ─────────────────────────────────────────────────────

_coco91_labels: list[str] = []


def _get_coco91() -> list[str]:
    global _coco91_labels
    if not _coco91_labels:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH) as f:
                _coco91_labels = [l.strip() for l in f.readlines()]
        else:
            _coco91_labels = [str(i) for i in range(91)]
    return _coco91_labels


# ── WordPiece tokenizer (matches verify_gdino.py) ─────────────────────────────

_vocab: dict[str, int] = {}


def _get_vocab() -> dict[str, int]:
    global _vocab
    if not _vocab:
        with open(VOCAB_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f):
                _vocab[line.rstrip("\n")] = i
    return _vocab


CLS_ID, SEP_ID, UNK_ID, PAD_ID = 101, 102, 100, 0


def _basic_tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    tokens, cur = [], ""
    for ch in text:
        if ch.isspace():
            if cur:
                tokens.append(cur)
                cur = ""
        elif not ch.isalnum():
            if cur:
                tokens.append(cur)
                cur = ""
            tokens.append(ch)
        else:
            cur += ch
    if cur:
        tokens.append(cur)
    return tokens


def _wordpiece(token: str, vocab: dict, max_chars: int = 100) -> list[str]:
    if len(token) > max_chars:
        return ["[UNK]"]
    if token in vocab:
        return [token]
    sub, start = [], 0
    while start < len(token):
        end, found = len(token), None
        while start < end:
            piece = token[start:end]
            if start > 0:
                piece = "##" + piece
            if piece in vocab:
                found = piece
                break
            end -= 1
        if found is None:
            return ["[UNK]"]
        sub.append(found)
        start = end
    return sub


def _encode(text: str) -> tuple[list[int], list[str]]:
    vocab = _get_vocab()
    ids, pieces = [CLS_ID], ["[CLS]"]
    for tok in _basic_tokenize(text):
        for wp in _wordpiece(tok, vocab):
            pieces.append(wp)
            ids.append(vocab.get(wp, UNK_ID))
    ids.append(SEP_ID)
    pieces.append("[SEP]")
    return ids, pieces


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_rfdetr(img: Image.Image) -> np.ndarray:
    """Resize to 560×560, normalize, return [1,3,560,560] float32."""
    img_r = img.resize((560, 560), Image.BILINEAR)
    x = np.asarray(img_r, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # NCHW


def _preprocess_gdino(img: Image.Image, prompt: str):
    """
    Resize to 800×800, normalize.
    Returns (pixel_values, pixel_mask, input_ids, attention_mask, token_type_ids, pieces).
    """
    img_r = img.resize((800, 800), Image.BILINEAR)
    pv = np.asarray(img_r, dtype=np.float32) / 255.0
    pv = (pv - IMAGENET_MEAN) / IMAGENET_STD
    pv = pv.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    pm = np.ones((1, 800, 800), dtype=np.int64)
    ids, pieces = _encode(prompt)
    input_ids    = np.array([ids], dtype=np.int64)
    attn_mask    = np.ones_like(input_ids)
    type_ids     = np.zeros_like(input_ids)
    return pv, pm, input_ids, attn_mask, type_ids, ids, pieces


def _preprocess_sam_encoder(img: Image.Image) -> tuple[np.ndarray, float]:
    """
    Resize long side to 1024 (SAM ResizeLongestSide).
    Returns (hwc_float32 input, scale_factor).
    Encoder does normalize+pad internally.
    """
    orig_w, orig_h = img.size
    scale = 1024.0 / max(orig_h, orig_w)
    new_w = int(orig_w * scale + 0.5)
    new_h = int(orig_h * scale + 0.5)
    img_r = img.resize((new_w, new_h), Image.BILINEAR)
    enc_in = np.asarray(img_r, dtype=np.float32)  # HWC, 0..255
    return enc_in, scale


# ── Postprocessing ────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _postprocess_rfdetr(
    logits: np.ndarray,
    boxes: np.ndarray,
    orig_w: int,
    orig_h: int,
    conf_threshold: float = 0.5,
) -> list[dict]:
    """
    RF-DETR postprocess (NMS-free):
      logits: [1, Q, 91] float32
      boxes:  [1, Q, 4]  float32  cxcywh normalized [0,1]
    Returns list of {label, score, bbox:[x0,y0,x1,y1] in orig pixels}.
    """
    labels = _get_coco91()
    probs = _sigmoid(logits[0])              # (Q, 91)
    scores = probs.max(axis=1)               # (Q,)
    class_ids = probs.argmax(axis=1)         # (Q,)
    keep = np.where(scores > conf_threshold)[0]
    detections = []
    for q in keep:
        cx, cy, bw, bh = boxes[0, q]
        x0 = float((cx - bw / 2) * orig_w)
        y0 = float((cy - bh / 2) * orig_h)
        x1 = float((cx + bw / 2) * orig_w)
        y1 = float((cy + bh / 2) * orig_h)
        cls = int(class_ids[q])
        label = labels[cls] if 0 <= cls < len(labels) else str(cls)
        detections.append({
            "label": label,
            "score": round(float(scores[q]), 4),
            "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        })
    detections.sort(key=lambda d: -d["score"])
    return detections


def _postprocess_gdino(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    ids: list[int],
    pieces: list[str],
    orig_w: int,
    orig_h: int,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[dict]:
    """
    GroundingDINO postprocess.
      logits:     [1, Q, 256] float32
      pred_boxes: [1, Q, 4]   float32  cxcywh normalized [0,1]
    Returns list of {label, score, bbox:[x0,y0,x1,y1]}.
    """
    probs = _sigmoid(logits[0])              # (Q, 256)
    L = len(ids)
    box_score = probs.max(axis=1)            # (Q,) — max over full 256 token dims
    keep = np.where(box_score > box_threshold)[0]

    def phrase_for_query(q: int) -> str:
        sel = [i for i in range(1, L - 1) if probs[q, i] > text_threshold]
        words = [pieces[i] for i in sel]
        out = ""
        for w in words:
            if w.startswith("##"):
                out += w[2:]
            else:
                out += (" " + w) if out else w
        return out.strip()

    detections = []
    for q in keep:
        cx, cy, bw, bh = pred_boxes[0, q]
        x0 = float((cx - bw / 2) * orig_w)
        y0 = float((cy - bh / 2) * orig_h)
        x1 = float((cx + bw / 2) * orig_w)
        y1 = float((cy + bh / 2) * orig_h)
        label = phrase_for_query(int(q))
        detections.append({
            "label": label,
            "score": round(float(box_score[q]), 4),
            "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        })
    detections.sort(key=lambda d: -d["score"])
    return detections


def _coco_rle(binary_mask: np.ndarray) -> list[int]:
    """COCO column-major (Fortran order) uncompressed RLE."""
    flat = binary_mask.flatten(order="F")
    counts, prev, run = [], 0, 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev = v
            run = 1
    counts.append(run)
    return counts


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="VisionServe-Python", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "providers": ORT_PROVIDERS}


@app.post("/predict")
async def predict(
    model: str = Form(...),
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """
    Run inference on the uploaded image.

    Parameters
    ----------
    model  : "rf-detr" | "grounding-dino" | "grounded-sam"
    image  : image file (JPEG/PNG)
    prompt : text prompt (required for grounding-dino and grounded-sam)

    Returns
    -------
    JSON with keys: detections, masks, duration_ms
    """
    t_start = time.perf_counter()

    # ── Load image ────────────────────────────────────────────────────────────
    img_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")
    orig_w, orig_h = img.size

    detections: list[dict] = []
    masks: list[dict] = []

    # ── RF-DETR detection ─────────────────────────────────────────────────────
    if model == "rf-detr":
        sess = _get_session("rf-detr")
        inp = _preprocess_rfdetr(img)
        outs = sess.run(None, {"input": inp})
        # outs[0] = logits [1,Q,91], outs[1] = boxes [1,Q,4]
        logits, boxes = outs[0], outs[1]
        detections = _postprocess_rfdetr(logits, boxes, orig_w, orig_h)

    # ── GroundingDINO open-vocab detection ───────────────────────────────────
    elif model == "grounding-dino":
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="grounding-dino requires a 'prompt' field (e.g. 'cat. remote.')",
            )
        sess = _get_session("grounding-dino")
        pv, pm, ids_arr, attn, types, ids_list, pieces = _preprocess_gdino(img, prompt)
        outs = sess.run(
            None,
            {
                "pixel_values":  pv,
                "pixel_mask":    pm,
                "input_ids":     ids_arr,
                "attention_mask": attn,
                "token_type_ids": types,
            },
        )
        logits, pred_boxes = outs[0], outs[1]
        detections = _postprocess_gdino(
            logits, pred_boxes, ids_list, pieces, orig_w, orig_h
        )

    # ── Grounded-SAM (GDino detection + MobileSAM segmentation) ──────────────
    elif model == "grounded-sam":
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="grounded-sam requires a 'prompt' field",
            )
        # 1) GDino detection
        gdino_sess = _get_session("grounded-sam-gdino")
        pv, pm, ids_arr, attn, types, ids_list, pieces = _preprocess_gdino(img, prompt)
        gdino_outs = gdino_sess.run(
            None,
            {
                "pixel_values":  pv,
                "pixel_mask":    pm,
                "input_ids":     ids_arr,
                "attention_mask": attn,
                "token_type_ids": types,
            },
        )
        logits, pred_boxes = gdino_outs[0], gdino_outs[1]
        detections = _postprocess_gdino(
            logits, pred_boxes, ids_list, pieces, orig_w, orig_h
        )

        # 2) MobileSAM encode
        enc_sess = _get_session("grounded-sam-enc")
        enc_in, scale = _preprocess_sam_encoder(img)
        embeddings = enc_sess.run(None, {"input_image": enc_in})[0]

        # 3) MobileSAM decode — one mask per detection box
        dec_sess = _get_session("grounded-sam-dec")
        orig_size = np.array([orig_h, orig_w], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask   = np.zeros(1, dtype=np.float32)

        for det in detections:
            x0, y0, x1, y1 = det["bbox"]
            # SAM expects points in resized (scale * orig) coordinate space
            pt = np.array([[x0, y0], [x1, y1]], dtype=np.float32) * scale
            point_coords  = pt.reshape(1, 2, 2)
            point_labels  = np.array([[2, 3]], dtype=np.float32)  # box prompt
            dec_outs = dec_sess.run(
                None,
                {
                    "image_embeddings": embeddings,
                    "point_coords":     point_coords,
                    "point_labels":     point_labels,
                    "mask_input":       mask_input,
                    "has_mask_input":   has_mask,
                    "orig_im_size":     orig_size,
                },
            )
            mask_logits = dec_outs[0]  # [1, 1, H, W] upsampled to orig size
            binary = (mask_logits[0, 0] > 0.0).astype(np.uint8)
            rle = _coco_rle(binary)
            masks.append({
                "label": det["label"],
                "score": det["score"],
                "rle":   rle[:20],  # truncate for JSON size — first 20 counts
                "size":  [int(orig_h), int(orig_w)],
                "positive_px": int(binary.sum()),
            })

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model!r}. Use rf-detr, grounding-dino, or grounded-sam.",
        )

    duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "detections": detections,
        "masks": masks,
        "duration_ms": duration_ms,
    }


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11436, workers=1)
