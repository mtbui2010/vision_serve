#!/usr/bin/env python3
"""Verify that models/clip-text (text tower) shares an embedding space with models/clip
(image tower). DEV TOOLING — VisionServe never runs Python at runtime (CLAUDE.md).

Two checks, because "both are 512-d" proves nothing:

  1. identity  — models/clip/model.onnx vs HuggingFace openai/clip-vit-base-patch32
                 vision tower. Cosine must be 1.0: same checkpoint => same space.
  2. zero-shot — crop ground-truth objects, embed crops with the image tower and
                 "a photo of a {class}" with the text tower, argmax cosine. A wrong
                 projection / wrong pooling / missing normalisation lands at chance.

    python scripts/verify_clip_text.py --dataset /path/to/etri_simple [--images 60]

The dataset layout is one JSON per image in <dataset>/annotations with
{"image", "annotations": [{"bbox": [x1,y1,x2,y2], "category": str}]} and the images in
<dataset>/images.
"""
import argparse
import glob
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
TEMPLATE = "a photo of a {}"


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--images", type=int, default=60)
    args = ap.parse_args()

    img_onnx = os.path.join(args.models, "clip", "model.onnx")
    txt_onnx = os.path.join(args.models, "clip-text", "model.onnx")
    ep = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    img_sess = ort.InferenceSession(img_onnx, providers=ep)
    txt_sess = ort.InferenceSession(txt_onnx, providers=ep)
    img_in = img_sess.get_inputs()[0].name

    from transformers import CLIPTokenizerFast
    tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    def embed_text(prompts):
        ids = tok(prompts, padding="max_length", max_length=77, truncation=True,
                  return_tensors="np")["input_ids"].astype(np.int64)
        return l2(txt_sess.run(None, {"input_ids": ids})[0])

    def preprocess(im):
        a = np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255.0
        return ((a - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)

    def embed_images(pil_images):
        out = []
        for i in range(0, len(pil_images), 64):
            batch = np.stack([preprocess(c) for c in pil_images[i:i + 64]])
            out.append(l2(img_sess.run(None, {img_in: batch})[0]))
        return np.concatenate(out)

    ann_dir = os.path.join(args.dataset, "annotations")
    img_dir = os.path.join(args.dataset, "images")
    files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))

    # --- check 1: is the shipped image tower this checkpoint's vision tower? ---
    try:
        import torch
        from transformers import CLIPVisionModelWithProjection
        hf = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").eval()
        probe = [Image.open(os.path.join(img_dir, json.load(open(f))["image"])) for f in files[:8]]
        batch = np.stack([preprocess(p) for p in probe])
        onnx_emb = l2(img_sess.run(None, {img_in: batch})[0])
        with torch.no_grad():
            hf_emb = l2(hf(pixel_values=torch.tensor(batch)).image_embeds.numpy())
        print("[1] shipped image tower vs HF vision tower")
        print("    per-image cosine:", np.round((onnx_emb * hf_emb).sum(1), 6).tolist())
        print("    max |delta|     :", float(np.abs(onnx_emb - hf_emb).max()))
    except Exception as e:  # torch is optional for check 2
        print("[1] skipped:", e)

    # --- check 2: zero-shot crop -> class name ---
    classes = sorted({a["category"] for f in files for a in json.load(open(f))["annotations"]})
    T = embed_text([TEMPLATE.format(c) for c in classes])

    crops, labels = [], []
    for f in files[:args.images]:
        d = json.load(open(f))
        im = Image.open(os.path.join(img_dir, d["image"])).convert("RGB")
        W, H = im.size
        for a in d["annotations"]:
            x1, y1, x2, y2 = a["bbox"]
            box = (max(0, x1), max(0, y1), min(W, x2), min(H, y2))
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            crops.append(im.crop(box))
            labels.append(classes.index(a["category"]))

    labels = np.array(labels)
    sim = embed_images(crops) @ T.T
    pred = sim.argmax(1)
    top5 = np.mean([labels[i] in np.argsort(-sim[i])[:5] for i in range(len(labels))])
    print(f"\n[2] zero-shot: {len(crops)} crops, {len(classes)} classes, template {TEMPLATE!r}")
    print(f"    top-1 = {(pred == labels).mean():.3f} ({int((pred == labels).sum())}/{len(labels)})"
          f"   chance = {1 / len(classes):.3f}")
    print(f"    top-5 = {top5:.3f}")
    print(f"    mean cos(correct) = {sim[np.arange(len(labels)), labels].mean():.4f}"
          f"   mean cos(all) = {sim.mean():.4f}")
    print("\n    per-class top-1:")
    for ci, c in enumerate(classes):
        m = labels == ci
        if m.sum():
            print(f"      {c:18s} n={int(m.sum()):3d}  acc={float((pred[m] == ci).mean()):.2f}")


if __name__ == "__main__":
    main()
