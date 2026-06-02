#!/usr/bin/env python3
"""
bench_python_ort.py — Direct Python ONNX Runtime performance benchmark.

Measures cold start, warm latency (p50/p95/p99/mean), throughput, and memory
for RF-DETR, GroundingDINO, and MobileSAM (Grounded-SAM pipeline).
Preprocess/postprocess implemented faithfully to match VisionServe Go code.

Output: benchmarks/results/python_ort.json
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = Path("/tmp/cats.jpg")
RFDETR_MODEL = MODELS_DIR / "rf-detr" / "rf-detr-base-real.onnx"
GDINO_MODEL = MODELS_DIR / "grounding-dino" / "model.onnx"
GDINO_VOCAB = MODELS_DIR / "grounding-dino" / "vocab.txt"
SAM_ENC = MODELS_DIR / "mobile-sam" / "mobile_sam_encoder.onnx"
SAM_DEC = MODELS_DIR / "mobile-sam" / "mobile_sam_decoder_single.onnx"

WARMUP_RUNS = 5
TIMED_RUNS = 30

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Check ORT and providers
# ---------------------------------------------------------------------------
import onnxruntime as ort

AVAILABLE_PROVIDERS = ort.get_available_providers()
HAS_CUDA = "CUDAExecutionProvider" in AVAILABLE_PROVIDERS
print(f"ORT version    : {ort.__version__}")
print(f"Providers      : {AVAILABLE_PROVIDERS}")
print(f"CUDA available : {HAS_CUDA}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rss_mb() -> int:
    return psutil.Process().memory_info().rss // (1024 * 1024)


def get_vram_mb() -> int:
    """Return VRAM used (MiB) via nvidia-smi, or -1 if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return sum(vals)
    except Exception:
        return -1


def stats(timings_ms: list) -> dict:
    a = sorted(timings_ms)
    n = len(a)
    mean = sum(a) / n
    p50 = a[int(n * 0.50)]
    p95 = a[min(int(n * 0.95), n - 1)]
    p99 = a[min(int(n * 0.99), n - 1)]
    return {"p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3), "mean_ms": round(mean, 3)}


def make_session(model_path: str, providers: list) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image() -> Image.Image:
    if not IMAGE_PATH.exists():
        import urllib.request
        print("Downloading cats.jpg ...")
        urllib.request.urlretrieve(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            IMAGE_PATH,
        )
    return Image.open(IMAGE_PATH).convert("RGB")


# ---------------------------------------------------------------------------
# RF-DETR preprocess / postprocess
# ---------------------------------------------------------------------------

def rfdetr_letterbox(img: Image.Image, target_w: int, target_h: int):
    """Letterbox: scale keeping aspect ratio, pad to target with black."""
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, orig_w, orig_h


def rfdetr_preprocess(img: Image.Image):
    """
    Letterbox to 560x560, NCHW float32, ImageNet normalize.
    Returns (input_tensor, scale, pad_x, pad_y, orig_w, orig_h).
    """
    lb, scale, pad_x, pad_y, orig_w, orig_h = rfdetr_letterbox(img, 560, 560)
    arr = np.array(lb, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD    # normalize
    nchw = arr.transpose(2, 0, 1)[np.newaxis]     # NCHW [1,3,H,W]
    return nchw, scale, pad_x, pad_y, orig_w, orig_h


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def rfdetr_postprocess(logits, boxes, scale, pad_x, pad_y, orig_w, orig_h,
                        conf_thresh=0.5):
    """
    Faithful to Go postprocess.go:
    - sigmoid on logits [1,Q,C], argmax per query
    - cxcywh normalized -> xywh on input (560x560) -> map back to original coords
    """
    INPUT_W, INPUT_H = 560, 560
    logits = logits[0]   # [Q, C]
    boxes  = boxes[0]    # [Q, 4]

    scores = sigmoid_np(logits)           # [Q, C]
    best_cls = np.argmax(scores, axis=1)  # [Q]
    best_score = scores[np.arange(len(best_cls)), best_cls]  # [Q]

    mask = best_score > conf_thresh
    best_cls   = best_cls[mask]
    best_score = best_score[mask]
    boxes_f    = boxes[mask]

    detections = []
    for i in range(len(best_cls)):
        cx, cy, bw, bh = boxes_f[i]
        # normalized cxcywh -> pixel on input image
        cx *= INPUT_W; cy *= INPUT_H; bw *= INPUT_W; bh *= INPUT_H
        x = cx - bw / 2; y = cy - bh / 2
        # map to original: (input - pad) / scale
        ox = (x - pad_x) / scale
        oy = (y - pad_y) / scale
        ow = bw / scale
        oh = bh / scale
        # clamp
        ox = max(0.0, ox); oy = max(0.0, oy)
        ow = min(ow, orig_w - ox); oh = min(oh, orig_h - oy)
        detections.append({
            "class_idx": int(best_cls[i]),
            "conf": float(best_score[i]),
            "bbox_xywh": [float(ox), float(oy), float(ow), float(oh)],
        })
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections[:300]


# ---------------------------------------------------------------------------
# GroundingDINO tokenizer (WordPiece, mirrors Go implementation)
# ---------------------------------------------------------------------------

class BertWordpieceTokenizer:
    """
    Minimal BERT bert-base-uncased WordPiece tokenizer.
    Mirrors groundingdino/tokenizer.go exactly.
    """
    UNK = 100; CLS = 101; SEP = 102

    def __init__(self, vocab_path: str):
        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                tok = line.rstrip("\n")
                self.vocab[tok] = idx
        self.id_to_tok = [""] * len(self.vocab)
        for tok, idx in self.vocab.items():
            self.id_to_tok[idx] = tok

    @staticmethod
    def _is_punct(ch: str) -> bool:
        o = ord(ch)
        if 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126:
            return True
        import unicodedata
        return unicodedata.category(ch).startswith("P")

    def _basic_split(self, text: str):
        words = []
        for ws in text.split():
            cur = []
            for ch in ws:
                if self._is_punct(ch):
                    if cur:
                        words.append("".join(cur)); cur = []
                    words.append(ch)
                else:
                    cur.append(ch)
            if cur:
                words.append("".join(cur))
        return words

    def _wordpiece(self, word: str):
        runes = list(word)
        pieces = []
        start = 0
        while start < len(runes):
            end = len(runes)
            found = False
            cur = ""
            while end > start:
                sub = "".join(runes[start:end])
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = sub; found = True; break
                end -= 1
            if not found:
                return ["[UNK]"]
            pieces.append(cur)
            start = end
        return pieces if pieces else ["[UNK]"]

    def encode(self, text: str):
        tokens = []
        for w in self._basic_split(text.lower()):
            tokens.extend(self._wordpiece(w))
        ids = [self.CLS]
        for t in tokens:
            ids.append(self.vocab.get(t, self.UNK))
        ids.append(self.SEP)
        L = len(ids)
        return {
            "input_ids":      np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, L), dtype=np.int64),
            "token_type_ids": np.zeros((1, L), dtype=np.int64),
        }

    def decode(self, ids):
        parts = []
        for i in ids:
            if 0 <= i < len(self.id_to_tok):
                tok = self.id_to_tok[i]
                if tok.startswith("##"):
                    if parts:
                        parts[-1] = parts[-1] + tok[2:]
                    else:
                        parts.append(tok[2:])
                else:
                    parts.append(tok)
        return " ".join(parts).strip()


def gdino_preprocess(img: Image.Image):
    """Force-resize to 800x800 (squash), ImageNet normalize, NCHW."""
    resized = img.resize((800, 800), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    nchw = arr.transpose(2, 0, 1)[np.newaxis]           # [1,3,800,800]
    pixel_mask = np.ones((1, 800, 800), dtype=np.int64)
    return nchw, pixel_mask


def gdino_postprocess(logits_out, boxes_out, orig_w, orig_h,
                       input_ids, tokenizer,
                       box_thresh=0.3, text_thresh=0.25):
    """
    Mirrors Go groundingdino.go postprocess:
    sigmoid, max over 256 text dims, threshold; decode phrase from token positions.
    """
    logits = logits_out[0]  # [Q, 256]
    boxes  = boxes_out[0]   # [Q, 4]
    ids_list = input_ids[0].tolist()
    last_tok = len(ids_list) - 1

    detections = []
    for q in range(logits.shape[0]):
        row = logits[q]
        probs = sigmoid_np(row)
        score = float(probs.max())
        if score <= box_thresh:
            continue
        phrase_ids = []
        for i, p in enumerate(probs):
            if 1 <= i < last_tok and p > text_thresh:
                phrase_ids.append(ids_list[i])
        phrase = tokenizer.decode(phrase_ids)
        if not phrase:
            continue
        cx, cy, bw, bh = boxes[q]
        x = (float(cx) - float(bw) / 2) * orig_w
        y = (float(cy) - float(bh) / 2) * orig_h
        detections.append({
            "class": phrase,
            "conf": score,
            "bbox_xywh": [x, y, float(bw) * orig_w, float(bh) * orig_h],
        })
    return detections


# ---------------------------------------------------------------------------
# SAM preprocess / postprocess
# ---------------------------------------------------------------------------

def sam_encoder_preprocess(img: Image.Image):
    """Resize long side to 1024, HWC float32 0..255 (no normalization in Go)."""
    orig_w, orig_h = img.size
    scale = 1024.0 / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32)  # HWC, 0..255
    return arr[np.newaxis], scale               # [1, H, W, 3], scale


def sam_decoder_inputs(embedding, orig_w, orig_h, scale, box_xywh):
    """
    Build decoder inputs from a bounding box.
    Box [x,y,w,h] in original coords -> 2 points (top-left label 2, bottom-right label 3)
    scaled to the resized-1024 space.
    Mirrors Go mobilesam/segment.go Segment() exactly.
    """
    x, y, w, h = box_xywh
    pts = np.array([[[x * scale, y * scale],
                     [(x + w) * scale, (y + h) * scale]]], dtype=np.float32)  # [1,2,2]
    labels = np.array([[2.0, 3.0]], dtype=np.float32)                          # [1,2]
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask   = np.zeros((1,), dtype=np.float32)
    orig_size  = np.array([float(orig_h), float(orig_w)], dtype=np.float32)
    return {
        "image_embeddings": embedding,
        "point_coords":     pts,
        "point_labels":     labels,
        "mask_input":       mask_input,
        "has_mask_input":   has_mask,
        "orig_im_size":     orig_size,
    }


def sam_postprocess(masks_out, iou_out):
    """
    Pick best mask (highest IoU), threshold logit > 0, encode column-major RLE.
    Mirrors Go mobilesam/segment.go maskToResult().
    """
    masks = masks_out[0]   # [N, H, W] or [1, N, H, W]
    if masks.ndim == 4:
        masks = masks[0]   # [N, H, W]
    iou = iou_out[0] if iou_out is not None else None

    best = 0
    conf = 0.0
    if iou is not None and len(iou.flatten()) > 0:
        iou_flat = iou.flatten()
        best = int(np.argmax(iou_flat))
        conf = float(iou_flat[best])

    mask = masks[best]  # [H, W]
    binary = mask > 0

    # column-major RLE (mirrors Go encodeRLEColumnMajor)
    h, w = binary.shape
    col_major = binary.T.flatten()  # column-major order
    counts = []
    prev = False; run = 0
    for v in col_major:
        b = bool(v)
        if b == prev:
            run += 1
        else:
            counts.append(run); prev = b; run = 1
    counts.append(run)

    # bbox from binary mask
    ys, xs = np.where(binary)
    if len(xs) > 0:
        bbox = [float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
    else:
        bbox = [0.0, 0.0, 0.0, 0.0]

    return {"rle": " ".join(map(str, counts)), "bbox_xywh": bbox, "conf": conf}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_model(name: str, run_fn, providers: list, warmup: int = WARMUP_RUNS,
                    timed: int = TIMED_RUNS):
    """
    Generic benchmark loop.
    run_fn() must:
      1. Load the ONNX session(s)
      2. Return a tuple (cold_start_ms, warm_latencies_ms, rss_mb_after_warmup, vram_mb)
         where warm_latencies_ms is a list of `timed` floats (each = one full inference ms).
    """
    print(f"\n--- {name} ---")
    result = run_fn(providers, warmup, timed)
    return result


# ---------------------------------------------------------------------------
# Per-model benchmark implementations
# ---------------------------------------------------------------------------

def bench_rfdetr(providers: list, warmup: int, timed: int):
    img = load_image()
    orig_w, orig_h = img.size
    model_path = str(RFDETR_MODEL)

    # --- Cold start ---
    t0 = time.perf_counter()
    sess = make_session(model_path, providers)
    # Run first inference to complete cold start
    inp, scale, px, py, ow, oh = rfdetr_preprocess(img)
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {input_name: inp})
    cold_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Cold start: {cold_ms:.1f} ms")

    # --- Warmup ---
    for _ in range(warmup):
        inp2, *_ = rfdetr_preprocess(img)
        sess.run(out_names, {input_name: inp2})

    rss_mb = get_rss_mb()
    vram_mb = get_vram_mb()
    print(f"  RSS after warmup: {rss_mb} MB | VRAM: {vram_mb} MB")

    # --- Timed runs ---
    timings = []
    for _ in range(timed):
        t0 = time.perf_counter()
        inp3, scale3, px3, py3, ow3, oh3 = rfdetr_preprocess(img)
        raw = sess.run(out_names, {input_name: inp3})
        # identify logits vs boxes by last dim
        if raw[0].shape[-1] == 4:
            boxes_raw, logits_raw = raw[0], raw[1]
        else:
            logits_raw, boxes_raw = raw[0], raw[1]
        _ = rfdetr_postprocess(logits_raw, boxes_raw, scale3, px3, py3, ow3, oh3)
        timings.append((time.perf_counter() - t0) * 1000.0)

    s = stats(timings)
    throughput = timed / (sum(timings) / 1000.0)
    print(f"  Timings -> {s} | throughput: {throughput:.2f} inf/s")

    # Quick sanity check
    inp_s, scale_s, px_s, py_s, ow_s, oh_s = rfdetr_preprocess(img)
    raw_s = sess.run(out_names, {input_name: inp_s})
    if raw_s[0].shape[-1] == 4:
        b_s, l_s = raw_s[0], raw_s[1]
    else:
        l_s, b_s = raw_s[0], raw_s[1]
    dets = rfdetr_postprocess(l_s, b_s, scale_s, px_s, py_s, ow_s, oh_s)
    print(f"  Detections (conf>0.5): {len(dets)}")
    if dets:
        print(f"    Top det: class_idx={dets[0]['class_idx']} conf={dets[0]['conf']:.3f}")

    return {
        "cold_start_ms": round(cold_ms, 3),
        "warm_latency": s,
        "throughput_inf_per_s": round(throughput, 3),
        "memory_rss_mb": rss_mb,
        "vram_mb": vram_mb if vram_mb >= 0 else "N/A",
        "n_detections_sample": len(dets),
    }


def bench_grounding_dino(providers: list, warmup: int, timed: int):
    img = load_image()
    orig_w, orig_h = img.size
    model_path = str(GDINO_MODEL)
    tok = BertWordpieceTokenizer(str(GDINO_VOCAB))
    PROMPT = "cat. remote."
    enc = tok.encode(PROMPT)

    # --- Cold start ---
    t0 = time.perf_counter()
    sess = make_session(model_path, providers)
    pv, pm = gdino_preprocess(img)
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    feed = {
        "pixel_values":   pv,
        "pixel_mask":     pm,
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc["token_type_ids"],
    }
    # Filter to only names the model expects
    feed = {k: v for k, v in feed.items() if k in in_names}
    outs = sess.run(out_names, feed)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Cold start: {cold_ms:.1f} ms")

    # --- Warmup ---
    for _ in range(warmup):
        pv2, pm2 = gdino_preprocess(img)
        feed2 = {
            "pixel_values":   pv2,
            "pixel_mask":     pm2,
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
        }
        feed2 = {k: v for k, v in feed2.items() if k in in_names}
        sess.run(out_names, feed2)

    rss_mb = get_rss_mb()
    vram_mb = get_vram_mb()
    print(f"  RSS after warmup: {rss_mb} MB | VRAM: {vram_mb} MB")

    # --- Timed runs ---
    timings = []
    for _ in range(timed):
        t0 = time.perf_counter()
        pv3, pm3 = gdino_preprocess(img)
        enc3 = tok.encode(PROMPT)
        feed3 = {
            "pixel_values":   pv3,
            "pixel_mask":     pm3,
            "input_ids":      enc3["input_ids"],
            "attention_mask": enc3["attention_mask"],
            "token_type_ids": enc3["token_type_ids"],
        }
        feed3 = {k: v for k, v in feed3.items() if k in in_names}
        raw3 = sess.run(out_names, feed3)
        # pick logits vs boxes
        logits_t = boxes_t = None
        for i, n in enumerate(out_names):
            if "box" in n.lower():
                boxes_t = raw3[i]
            elif "logit" in n.lower():
                logits_t = raw3[i]
        if logits_t is None or boxes_t is None:
            for r in raw3:
                if r.shape[-1] == 4:
                    boxes_t = r
                elif r.shape[-1] == 256:
                    logits_t = r
        _ = gdino_postprocess(logits_t, boxes_t, orig_w, orig_h,
                               enc3["input_ids"], tok)
        timings.append((time.perf_counter() - t0) * 1000.0)

    s = stats(timings)
    throughput = timed / (sum(timings) / 1000.0)
    print(f"  Timings -> {s} | throughput: {throughput:.2f} inf/s")

    # Sample detections
    pv_s, pm_s = gdino_preprocess(img)
    enc_s = tok.encode(PROMPT)
    feed_s = {
        "pixel_values":   pv_s,
        "pixel_mask":     pm_s,
        "input_ids":      enc_s["input_ids"],
        "attention_mask": enc_s["attention_mask"],
        "token_type_ids": enc_s["token_type_ids"],
    }
    feed_s = {k: v for k, v in feed_s.items() if k in in_names}
    raw_s = sess.run(out_names, feed_s)
    logits_s = boxes_s = None
    for i, n in enumerate(out_names):
        if "box" in n.lower():
            boxes_s = raw_s[i]
        elif "logit" in n.lower():
            logits_s = raw_s[i]
    if logits_s is None or boxes_s is None:
        for r in raw_s:
            if r.shape[-1] == 4:
                boxes_s = r
            elif r.shape[-1] == 256:
                logits_s = r
    dets = gdino_postprocess(logits_s, boxes_s, orig_w, orig_h,
                              enc_s["input_ids"], tok)
    print(f"  Detections (prompt={PROMPT!r}, conf>0.3): {len(dets)}")
    if dets:
        print(f"    Top det: class={dets[0]['class']!r} conf={dets[0]['conf']:.3f}")

    return {
        "cold_start_ms": round(cold_ms, 3),
        "warm_latency": s,
        "throughput_inf_per_s": round(throughput, 3),
        "memory_rss_mb": rss_mb,
        "vram_mb": vram_mb if vram_mb >= 0 else "N/A",
        "n_detections_sample": len(dets),
        "prompt": PROMPT,
    }


def bench_grounded_sam(providers: list, warmup: int, timed: int):
    """
    Full Grounded-SAM pipeline: GroundingDINO detection + MobileSAM encoder + decoder.
    One full inference = preprocess + GDINO run + for each detected box: SAM enc + SAM dec.
    """
    img = load_image()
    orig_w, orig_h = img.size
    tok = BertWordpieceTokenizer(str(GDINO_VOCAB))
    PROMPT = "cat."

    # --- Cold start: load all 3 sessions ---
    t0 = time.perf_counter()
    gdino_sess = make_session(str(GDINO_MODEL), providers)
    enc_sess   = make_session(str(SAM_ENC), providers)
    dec_sess   = make_session(str(SAM_DEC), providers)

    # First inference (cold)
    gdino_in_names = [i.name for i in gdino_sess.get_inputs()]
    gdino_out_names = [o.name for o in gdino_sess.get_outputs()]
    enc_in_name  = enc_sess.get_inputs()[0].name
    dec_out_names = [o.name for o in dec_sess.get_outputs()]

    pv, pm = gdino_preprocess(img)
    enc_tok = tok.encode(PROMPT)
    feed = {
        "pixel_values":   pv,
        "pixel_mask":     pm,
        "input_ids":      enc_tok["input_ids"],
        "attention_mask": enc_tok["attention_mask"],
        "token_type_ids": enc_tok["token_type_ids"],
    }
    feed = {k: v for k, v in feed.items() if k in gdino_in_names}
    gdino_outs = gdino_sess.run(gdino_out_names, feed)

    logits_t = boxes_t = None
    for i, n in enumerate(gdino_out_names):
        if "box" in n.lower():
            boxes_t = gdino_outs[i]
        elif "logit" in n.lower():
            logits_t = gdino_outs[i]
    if logits_t is None or boxes_t is None:
        for r in gdino_outs:
            if r.shape[-1] == 4:
                boxes_t = r
            elif r.shape[-1] == 256:
                logits_t = r

    dets = gdino_postprocess(logits_t, boxes_t, orig_w, orig_h,
                              enc_tok["input_ids"], tok)
    # Use top detection (or a fallback box) for SAM
    if dets:
        sam_box = dets[0]["bbox_xywh"]
    else:
        sam_box = [orig_w * 0.1, orig_h * 0.1, orig_w * 0.8, orig_h * 0.8]

    enc_in_arr, scale = sam_encoder_preprocess(img)
    embedding_out = enc_sess.run(None, {enc_in_name: enc_in_arr})
    embedding = embedding_out[0]

    dec_inputs = sam_decoder_inputs(embedding, orig_w, orig_h, scale, sam_box)
    dec_in_names = [i.name for i in dec_sess.get_inputs()]
    dec_feed = {k: v for k, v in dec_inputs.items() if k in dec_in_names}
    dec_outs = dec_sess.run(dec_out_names, dec_feed)

    # Find masks and iou in dec outputs
    masks_out = iou_out = None
    for i, n in enumerate(dec_out_names):
        n_lower = n.lower()
        if "iou" in n_lower:
            iou_out = dec_outs[i]
        elif n_lower == "masks" or ("mask" in n_lower and "low_res" not in n_lower):
            masks_out = dec_outs[i]
    if masks_out is None:
        best_area = -1
        for out in dec_outs:
            if out.ndim == 4:
                area = out.shape[2] * out.shape[3]
                if area > best_area:
                    best_area = area; masks_out = out
    if iou_out is None:
        for out in dec_outs:
            if out.ndim == 2:
                iou_out = out; break

    _ = sam_postprocess(masks_out, iou_out)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Cold start: {cold_ms:.1f} ms")

    # --- Warmup ---
    for _ in range(warmup):
        pv2, pm2 = gdino_preprocess(img)
        enc_tok2 = tok.encode(PROMPT)
        feed2 = {
            "pixel_values":   pv2,
            "pixel_mask":     pm2,
            "input_ids":      enc_tok2["input_ids"],
            "attention_mask": enc_tok2["attention_mask"],
            "token_type_ids": enc_tok2["token_type_ids"],
        }
        feed2 = {k: v for k, v in feed2.items() if k in gdino_in_names}
        g2 = gdino_sess.run(gdino_out_names, feed2)
        lt2 = bt2 = None
        for i, n in enumerate(gdino_out_names):
            if "box" in n.lower(): bt2 = g2[i]
            elif "logit" in n.lower(): lt2 = g2[i]
        if lt2 is None or bt2 is None:
            for r in g2:
                if r.shape[-1] == 4: bt2 = r
                elif r.shape[-1] == 256: lt2 = r
        d2 = gdino_postprocess(lt2, bt2, orig_w, orig_h, enc_tok2["input_ids"], tok)
        box2 = d2[0]["bbox_xywh"] if d2 else sam_box
        ei2, sc2 = sam_encoder_preprocess(img)
        emb2 = enc_sess.run(None, {enc_in_name: ei2})[0]
        di2 = sam_decoder_inputs(emb2, orig_w, orig_h, sc2, box2)
        df2 = {k: v for k, v in di2.items() if k in dec_in_names}
        dec_sess.run(dec_out_names, df2)

    rss_mb = get_rss_mb()
    vram_mb = get_vram_mb()
    print(f"  RSS after warmup: {rss_mb} MB | VRAM: {vram_mb} MB")

    # --- Timed runs (full pipeline) ---
    timings = []
    for _ in range(timed):
        t_start = time.perf_counter()
        # Preprocess + GDINO
        pv3, pm3 = gdino_preprocess(img)
        enc_tok3 = tok.encode(PROMPT)
        feed3 = {
            "pixel_values":   pv3,
            "pixel_mask":     pm3,
            "input_ids":      enc_tok3["input_ids"],
            "attention_mask": enc_tok3["attention_mask"],
            "token_type_ids": enc_tok3["token_type_ids"],
        }
        feed3 = {k: v for k, v in feed3.items() if k in gdino_in_names}
        g3 = gdino_sess.run(gdino_out_names, feed3)
        lt3 = bt3 = None
        for i, n in enumerate(gdino_out_names):
            if "box" in n.lower(): bt3 = g3[i]
            elif "logit" in n.lower(): lt3 = g3[i]
        if lt3 is None or bt3 is None:
            for r in g3:
                if r.shape[-1] == 4: bt3 = r
                elif r.shape[-1] == 256: lt3 = r
        d3 = gdino_postprocess(lt3, bt3, orig_w, orig_h, enc_tok3["input_ids"], tok)
        box3 = d3[0]["bbox_xywh"] if d3 else sam_box
        # SAM encoder
        ei3, sc3 = sam_encoder_preprocess(img)
        emb3 = enc_sess.run(None, {enc_in_name: ei3})[0]
        # SAM decoder
        di3 = sam_decoder_inputs(emb3, orig_w, orig_h, sc3, box3)
        df3 = {k: v for k, v in di3.items() if k in dec_in_names}
        do3 = dec_sess.run(dec_out_names, df3)
        # postprocess
        mo3 = io3 = None
        for i, n in enumerate(dec_out_names):
            nl = n.lower()
            if "iou" in nl: io3 = do3[i]
            elif nl == "masks" or ("mask" in nl and "low_res" not in nl): mo3 = do3[i]
        if mo3 is None:
            ba = -1
            for o in do3:
                if o.ndim == 4 and o.shape[2] * o.shape[3] > ba:
                    ba = o.shape[2] * o.shape[3]; mo3 = o
        if io3 is None:
            for o in do3:
                if o.ndim == 2: io3 = o; break
        _ = sam_postprocess(mo3, io3)
        timings.append((time.perf_counter() - t_start) * 1000.0)

    s = stats(timings)
    throughput = timed / (sum(timings) / 1000.0)
    print(f"  Timings -> {s} | throughput: {throughput:.2f} inf/s")
    print(f"  GDINO dets for SAM: {len(dets)}")

    return {
        "cold_start_ms": round(cold_ms, 3),
        "warm_latency": s,
        "throughput_inf_per_s": round(throughput, 3),
        "memory_rss_mb": rss_mb,
        "vram_mb": vram_mb if vram_mb >= 0 else "N/A",
        "n_gdino_detections_sample": len(dets),
        "prompt": PROMPT,
        "note": "Full pipeline: GroundingDINO + SAM encoder + SAM decoder per detection",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_for_device(device: str):
    """Run all benchmarks for a given device ('cpu' or 'gpu')."""
    if device == "gpu":
        if not HAS_CUDA:
            return {
                "note": "CUDAExecutionProvider not available in this ORT build",
                "available_providers": AVAILABLE_PROVIDERS,
            }
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"\n{'='*60}")
    print(f"Device: {device.upper()}  providers={providers}")
    print(f"{'='*60}")

    results = {}

    print("\n[1/3] RF-DETR")
    try:
        results["rf_detr"] = bench_rfdetr(providers, WARMUP_RUNS, TIMED_RUNS)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["rf_detr"] = {"error": str(e)}

    print("\n[2/3] GroundingDINO")
    try:
        results["grounding_dino"] = bench_grounding_dino(providers, WARMUP_RUNS, TIMED_RUNS)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["grounding_dino"] = {"error": str(e)}

    print("\n[3/3] Grounded-SAM (GroundingDINO + MobileSAM)")
    try:
        results["grounded_sam"] = bench_grounded_sam(providers, WARMUP_RUNS, TIMED_RUNS)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["grounded_sam"] = {"error": str(e)}

    return results


def main():
    print("VisionServe Python ORT Benchmark")
    print(f"ORT version      : {ort.__version__}")
    print(f"CUDA provider    : {HAS_CUDA}")
    print(f"Image            : {IMAGE_PATH}")
    print(f"RF-DETR model    : {RFDETR_MODEL} ({RFDETR_MODEL.stat().st_size // (1024*1024)} MB)")
    print(f"GDINO model      : {GDINO_MODEL} ({GDINO_MODEL.stat().st_size // (1024*1024)} MB)")
    print(f"SAM encoder      : {SAM_ENC} ({SAM_ENC.stat().st_size // (1024*1024)} MB)")
    print(f"SAM decoder      : {SAM_DEC} ({SAM_DEC.stat().st_size // (1024*1024)} MB)")

    output = {
        "baseline": "python_ort_direct",
        "ort_version": ort.__version__,
        "python_version": sys.version,
        "available_providers": AVAILABLE_PROVIDERS,
        "cuda_available": HAS_CUDA,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "image": str(IMAGE_PATH),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": {},
        "gpu": {},
    }

    output["cpu"] = run_for_device("cpu")
    output["gpu"] = run_for_device("gpu")

    out_path = RESULTS_DIR / "python_ort.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {out_path}")

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<22} {'Device':<6} {'Cold(ms)':>10} {'p50(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} {'mean(ms)':>10} {'inf/s':>8} {'RSS(MB)':>8}")
    print("-" * 80)
    for device in ["cpu", "gpu"]:
        dev_results = output[device]
        if "note" in dev_results:
            print(f"{'N/A':<22} {device:<6}   {dev_results['note']}")
            continue
        for model_key, label in [
            ("rf_detr",       "RF-DETR"),
            ("grounding_dino","GroundingDINO"),
            ("grounded_sam",  "Grounded-SAM"),
        ]:
            r = dev_results.get(model_key, {})
            if "error" in r:
                print(f"{label:<22} {device:<6}   ERROR: {r['error'][:50]}")
                continue
            wl = r.get("warm_latency", {})
            print(f"{label:<22} {device:<6} {r.get('cold_start_ms', 'N/A'):>10} "
                  f"{wl.get('p50_ms', 'N/A'):>9} {wl.get('p95_ms', 'N/A'):>9} "
                  f"{wl.get('p99_ms', 'N/A'):>9} {wl.get('mean_ms', 'N/A'):>10} "
                  f"{r.get('throughput_inf_per_s', 'N/A'):>8} "
                  f"{r.get('memory_rss_mb', 'N/A'):>8}")

    print("=" * 80)
    if not HAS_CUDA:
        print("NOTE: CUDAExecutionProvider was NOT available in this ORT build.")
        print("      GPU results are marked N/A. To enable GPU inference, install")
        print("      onnxruntime-gpu or rebuild ORT with CUDA support.")
    else:
        print("NOTE: CUDAExecutionProvider IS available.")


if __name__ == "__main__":
    main()
