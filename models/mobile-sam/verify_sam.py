#!/usr/bin/env python3
# =============================================================================
# verify_sam.py — DEV-ONLY spike. NOT part of the runtime.
#
# Purpose: verify the EXACT I/O contract of the MobileSAM ONNX files in this
# directory so the behavior can be ported to pure Go (yalue/onnxruntime_go).
# The production runtime is Go-only; this Python script exists solely to
# inspect tensors and run an end-to-end sanity check during development.
#
# Run with: /home/trung/miniconda3/envs/label/bin/python3 verify_sam.py
# Requires: onnxruntime, numpy, Pillow. Downloads a COCO test image to /tmp.
# =============================================================================
import os
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ENC = os.path.join(HERE, "mobile_sam_encoder.onnx")
DEC_SINGLE = os.path.join(HERE, "mobile_sam_decoder_single.onnx")
DEC_MULTI = os.path.join(HERE, "mobile_sam_decoder_multi.onnx")

IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
IMG_PATH = "/tmp/cats.jpg"

# SAM normalization constants (verified baked INTO encoder graph; reproduced
# here only to document — encoder does normalize+pad itself).
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
LONG_SIDE = 1024


def dump_io(path, title):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"\n===== {title}: {os.path.basename(path)} =====")
    print("INPUTS:")
    for i in s.get_inputs():
        print(f"  {i.name:18} shape={i.shape} dtype={i.type}")
    print("OUTPUTS:")
    for o in s.get_outputs():
        print(f"  {o.name:18} shape={o.shape} dtype={o.type}")
    return s


def get_preprocess_shape(oldh, oldw, long_side=LONG_SIDE):
    """SAM ResizeLongestSide: scale so the LONGER side == long_side."""
    scale = long_side / max(oldh, oldw)
    neww = int(oldw * scale + 0.5)
    newh = int(oldh * scale + 0.5)
    return newh, neww, scale


def main():
    if not os.path.exists(IMG_PATH):
        urllib.request.urlretrieve(IMG_URL, IMG_PATH)
    pil = Image.open(IMG_PATH).convert("RGB")
    orig_w, orig_h = pil.size
    print(f"orig image WxH = {orig_w}x{orig_h}")

    enc = dump_io(ENC, "ENCODER")
    dec_s = dump_io(DEC_SINGLE, "DECODER single")
    dec_m = dump_io(DEC_MULTI, "DECODER multi")

    # ---- Preprocess for ENCODER ----
    # Encoder input is HWC float [h,w,3], range 0..255 (graph does mean/std + pad).
    # We must RESIZE long side to 1024 keeping aspect ratio BEFORE feeding.
    newh, neww, scale = get_preprocess_shape(orig_h, orig_w)
    print(f"\nresized HxW = {newh}x{neww}  scale = {scale:.6f}")
    resized = pil.resize((neww, newh), Image.BILINEAR)
    enc_in = np.asarray(resized, dtype=np.float32)  # HWC, 0..255
    print("encoder input shape:", enc_in.shape, "min/max:", enc_in.min(), enc_in.max())

    emb = enc.run(None, {"input_image": enc_in})[0]
    print("image_embeddings shape:", emb.shape, "dtype:", emb.dtype)

    # ---- BOX prompt for the LEFT cat ----
    # Box in ORIGINAL image pixel coords [x0,y0,x1,y1].
    box_orig = np.array([12, 60, 320, 437], dtype=np.float32)  # left cat
    # SAM encodes a box as TWO points: top-left (label 2) and bottom-right (label 3).
    # Point coords must be in the RESIZED (long-side-1024) coordinate space,
    # i.e. multiply original coords by `scale`. (NOT padded-space-specific: pad is
    # bottom/right only, so resized coords already match the padded 1024 canvas.)
    pt = box_orig.reshape(2, 2) * scale  # [[x0,y0],[x1,y1]] in 1024 space
    point_coords = pt.reshape(1, 2, 2).astype(np.float32)
    point_labels = np.array([[2, 3]], dtype=np.float32)

    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.zeros(1, dtype=np.float32)
    orig_im_size = np.array([orig_h, orig_w], dtype=np.float32)  # (H, W)

    feeds = {
        "image_embeddings": emb,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": orig_im_size,
    }

    print("\n--- DECODER single ---")
    masks_s, iou_s, low_s = dec_s.run(None, feeds)
    print("masks:", masks_s.shape, "iou:", iou_s.shape, iou_s, "low_res:", low_s.shape)
    print("masks logit min/max:", masks_s.min(), masks_s.max())

    print("\n--- DECODER multi ---")
    masks_m, iou_m, low_m = dec_m.run(None, feeds)
    print("masks:", masks_m.shape, "iou:", iou_m.shape, iou_m, "low_res:", low_m.shape)
    print("masks logit min/max:", masks_m.min(), masks_m.max())

    # Pick best mask from multi by iou
    best = int(np.argmax(iou_m[0]))
    print("multi best mask idx by iou:", best, "iou:", iou_m[0][best])

    # ---- Threshold (SAM uses logit > 0) and overlay ----
    def save_overlay(mask_logits, fname, idx=0):
        m = mask_logits[0, idx]  # already upsampled to orig_im_size by the graph
        binm = (m > 0.0).astype(np.uint8)
        print(f"  {fname}: mask shape {m.shape}, positive px = {binm.sum()}")
        over = np.asarray(pil).copy()
        red = np.zeros_like(over)
        red[..., 0] = 255
        alpha = 0.5
        sel = binm.astype(bool)
        over[sel] = (over[sel] * (1 - alpha) + red[sel] * alpha).astype(np.uint8)
        out = Image.fromarray(over)
        d = ImageDraw.Draw(out)
        d.rectangle([box_orig[0], box_orig[1], box_orig[2], box_orig[3]],
                    outline=(0, 255, 0), width=3)
        out.save(fname)
        return binm

    print("\noverlays:")
    bs = save_overlay(masks_s, "/tmp/sam_single.png", 0)
    bm = save_overlay(masks_m, "/tmp/sam_multi_best.png", best)

    # ---- Demonstrate column-major (COCO) RLE on the single mask ----
    rle = coco_rle_counts(bs)
    print("\nCOCO column-major RLE (single mask): first 12 counts:", rle[:12],
          "total runs:", len(rle), "sum:", sum(rle), "== H*W:", bs.shape[0]*bs.shape[1])


def coco_rle_counts(binary_mask):
    """COCO uncompressed RLE: counts of alternating 0/1 runs, COLUMN-MAJOR
    (Fortran order), always STARTING with a run of 0s."""
    flat = binary_mask.flatten(order="F")  # column-major
    counts = []
    prev = 0  # RLE convention: first run is background (0)
    run = 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev = v
            run = 1
    counts.append(run)
    return counts


if __name__ == "__main__":
    main()
