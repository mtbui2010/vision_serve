#!/usr/bin/env python3
# verify_gdino.py — DEV-ONLY verification harness for GroundingDINO ONNX.
#
# NOT part of the Go runtime. This script exists purely to VERIFY the exact I/O,
# tokenizer behavior, preprocessing, and postprocessing of the community ONNX
# export (onnx-community/grounding-dino-tiny-ONNX, Apache-2.0) so the logic can
# be ported to pure Go. Run with the `label` conda env. Do NOT ship this.
#
#   python3 verify_gdino.py
#
# Model source: onnx-community/grounding-dino-tiny-ONNX (Apache-2.0),
# derived from IDEA-Research/grounding-dino-tiny (Apache-2.0).

import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.onnx")
VOCAB = os.path.join(HERE, "vocab.txt")
IMG = "/home/trung/trung_workdir/vision_serve/demo/images/000000039769.jpg"
PROMPT = "cat. remote."
OUT_PNG = "/tmp/gdino_overlay.png"

# Preprocess constants (from preprocessor_config.json)
SIZE = 800
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Special tokens (BERT). SPECIAL_TOKENS in transformers GroundingDINO = [CLS],[SEP],".","?"
CLS, SEP, UNK, PAD = 101, 102, 100, 0


# ---------------------------------------------------------------------------
# Minimal pure-Python WordPiece tokenizer (mirrors what Go would implement).
# Verifies we can reproduce HF BertTokenizer ids without the transformers lib.
# ---------------------------------------------------------------------------
def load_vocab(path):
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[line.rstrip("\n")] = i
    return vocab


def basic_tokenize(text):
    # lowercase + split on whitespace + split punctuation into own tokens
    text = text.lower().strip()
    out = []
    cur = ""
    for ch in text:
        if ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
        elif not ch.isalnum():  # punctuation -> standalone token
            if cur:
                out.append(cur)
                cur = ""
            out.append(ch)
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def wordpiece(token, vocab, max_chars=100):
    if len(token) > max_chars:
        return ["[UNK]"]
    if token in vocab:
        return [token]
    sub = []
    start = 0
    while start < len(token):
        end = len(token)
        found = None
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


def encode(text, vocab):
    ids = [CLS]
    pieces = ["[CLS]"]
    for tok in basic_tokenize(text):
        for wp in wordpiece(tok, vocab):
            pieces.append(wp)
            ids.append(vocab.get(wp, UNK))
    ids.append(SEP)
    pieces.append("[SEP]")
    return ids, pieces


# ---------------------------------------------------------------------------
# Image preprocess: resize to 800x800 (bilinear), rescale /255, normalize.
# pixel_mask = all ones (no padding since we resize to exact square).
# ---------------------------------------------------------------------------
def preprocess(img_path):
    im = Image.open(img_path).convert("RGB")
    ow, oh = im.size
    rs = im.resize((SIZE, SIZE), Image.BILINEAR)  # resample=2 = BILINEAR
    arr = np.asarray(rs).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None]  # 1,3,H,W
    pixel_mask = np.ones((1, SIZE, SIZE), dtype=np.int64)
    return arr.astype(np.float32), pixel_mask, (ow, oh)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    vocab = load_vocab(VOCAB)
    print(f"vocab size: {len(vocab)}")

    # --- tokenizer cross-check against HF if available ---
    ids, pieces = encode(PROMPT, vocab)
    print("our tokens :", pieces)
    print("our ids    :", ids)
    try:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(HERE)
        hf_ids = hf(PROMPT)["input_ids"]
        print("HF  ids    :", hf_ids)
        print("MATCH      :", hf_ids == ids)
    except Exception as e:
        print("HF tokenizer compare skipped:", e)

    pixel_values, pixel_mask, (ow, oh) = preprocess(IMG)
    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    token_type_ids = np.zeros_like(input_ids)

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    outs = sess.run(
        None,
        {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    logits, pred_boxes = outs
    print("logits shape:", logits.shape, "pred_boxes shape:", pred_boxes.shape)

    # --- postprocess (mirrors HF post_process_grounded_object_detection) ---
    # logits[0]: (num_queries, 256) token-aligned logits. sigmoid -> per-token prob.
    # score per box = max over ALL 256 token dims (HF uses dim=-1 over full 256).
    # label per box = decode of all token positions where prob > text_threshold,
    #   excluding position 0 ([CLS]) and the final position (get_phrases_from_posmap).
    probs = sigmoid(logits[0])  # (900, 256)
    L = len(ids)
    box_score = probs.max(axis=1)  # (900,) max over full 256 dims, like HF

    BOX_THRESH = 0.30  # GroundingDINO default box_threshold ~0.25-0.35
    TEXT_THRESH = 0.25  # text_threshold for phrase token selection
    keep = np.where(box_score > BOX_THRESH)[0]
    print(f"\nboxes over thresh {BOX_THRESH}: {len(keep)}")

    def phrase_for_box(q):
        # collect token positions (within prompt range) above text_threshold,
        # exclude [CLS] (pos 0) and last pos; decode to phrase string.
        sel = [i for i in range(1, L - 1) if probs[q, i] > TEXT_THRESH]
        words = [pieces[i] for i in sel]
        out = ""
        for w in words:
            if w.startswith("##"):
                out += w[2:]
            else:
                out += (" " + w) if out else w
        return out.strip()

    im = Image.open(IMG).convert("RGB")
    draw = ImageDraw.Draw(im)
    results = []
    for q in keep:
        cx, cy, bw, bh = pred_boxes[0, q]  # normalized cxcywh in [0,1]
        x0 = (cx - bw / 2) * ow
        y0 = (cy - bh / 2) * oh
        x1 = (cx + bw / 2) * ow
        y1 = (cy + bh / 2) * oh
        label = phrase_for_box(q)
        sc = float(box_score[q])
        results.append((label, sc, [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.text((x0 + 2, y0 + 2), f"{label} {sc:.2f}", fill=(255, 255, 0))

    results.sort(key=lambda r: -r[1])
    print("\n=== detections (label, score, xyxy orig) ===")
    for r in results:
        print(" ", r)
    im.save(OUT_PNG)
    print("\noverlay saved:", OUT_PNG, "orig size:", (ow, oh))


if __name__ == "__main__":
    main()
