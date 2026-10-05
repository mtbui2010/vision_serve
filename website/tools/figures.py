#!/usr/bin/env python3
"""Regenerate the documentation figures from REAL VisionServe runs.

Every figure in website/docs/assets/img/ is drawn from the JSON a running VisionServe server
returned for a real photo (or, for OCR, a synthetic receipt this script renders). Nothing is
mocked: boxes, masks, grasps, depth maps, embeddings and preprocessed tensors all come from the
HTTP API. The script also writes figures.json (what was asked, how many results, server-side
duration_ms, device) and CREDITS.md (photo credits).

Usage
-----
1. Start a server with the models installed (any port; GPU optional):

       source scripts/gpu-env.sh            # optional, for the CUDA EP
       visionserve serve --addr 127.0.0.1:11640 --models ./models --idle-unload-seconds 0

2. Run this script with a Python that has numpy, Pillow, matplotlib and requests:

       python website/tools/figures.py \
           --server http://127.0.0.1:11640 \
           --coco-images /path/to/coco/val2017 \
           --coco-annotations /path/to/coco/annotations/instances_val2017.json

   Options: --out DIR (default: website/docs/assets/img next to this script),
   --host-note TEXT (recorded in figures.json, e.g. "1x RTX A6000, CUDA EP"),
   --only detection,depth,... (regenerate a subset; figures.json entries of the other figures
   are kept).

3. The `inspect` step (guides/inspect.md) also needs `--inspect-models DIR`, the server's
   registry. Use a scratch registry holding a copy of models/rf-detr: the step writes two copies
   of its manifest with one preprocessing mistake each (letterbox, mean/std), runs the
   converter's tiers B1 / B2 / C against them (Python package clients/python/visionserve/convert,
   plus onnxruntime, pyyaml and pycocotools), prints the report and removes the copies. The
   `preprocessing` step uses the same directory for its letterbox panel: it draws a temporary
   copy of models/rf-detr-nano with `letterbox: true` (labelled "illustration"; rf-detr-nano
   itself squashes, as RF-DETR is trained) and removes it afterwards.

Photos: only COCO val2017 images whose Flickr license is "Attribution License" (CC BY 2.0)
are used (PHOTOS below); each figure credits its photo. Durations are the server's own
duration_ms of a warm call (each request is sent once to warm up, then measured).

Models that are not installed on the server are skipped with a message.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "clients" / "python"))
from visionserve.types import Mask  # noqa: E402  (the SDK's column-major RLE decoder)

# COCO val2017 photos with license 4 = "Attribution License" (CC BY 2.0).
PHOTOS = [34873, 177015, 389381, 372819, 29596, 363840, 8021, 263969, 564133]

# Categorical palette (fixed order; colour follows the class, assigned by first appearance).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

# Per-model normalisation (from each model's manifest.yaml) to turn a returned tensor back into
# pixels. mobile-sam feeds raw 0..255 HWC pixels, so it has none.
NORM = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
}

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class Colors:
    """Colour follows the class name: the first class seen gets slot 1, and so on."""

    def __init__(self):
        self.map = {}

    def __call__(self, key):
        if key not in self.map:
            self.map[key] = PALETTE[len(self.map) % len(PALETTE)]
        return self.map[key]


# --------------------------------------------------------------------------------------------
# Server access
# --------------------------------------------------------------------------------------------

class Server:
    def __init__(self, url):
        self.url = url.rstrip("/")
        r = requests.get(self.url + "/api/models", timeout=30)
        r.raise_for_status()
        self.models = {m["name"]: m for m in r.json()}

    def has(self, name):
        return self.models.get(name, {}).get("state") in ("available", "loaded")

    def _post(self, path, model, image_bytes, fields):
        data = {"model": model, **{k: str(v) for k, v in fields.items()}}
        files = {"image": ("image.jpg", image_bytes, "application/octet-stream")}
        r = requests.post(self.url + path, data=data, files=files, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"{path} {model}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def predict(self, model, image_bytes, **fields):
        """Warm call, then the measured call; returns the measured JSON."""
        self._post("/api/predict", model, image_bytes, fields)
        return self._post("/api/predict", model, image_bytes, fields)

    def preprocess(self, model, image_bytes, **fields):
        return self._post("/api/preprocess", model, image_bytes, fields)

    def explain_numpy(self, model, image_bytes, **fields):
        """POST /api/explain with format=numpy: (heatmap HxW float32, explained detection, query)."""
        data = {"model": model, "format": "numpy", **{k: str(v) for k, v in fields.items()}}
        files = {"image": ("image.jpg", image_bytes, "application/octet-stream")}
        r = requests.post(self.url + "/api/explain", data=data, files=files, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"/api/explain {model}: HTTP {r.status_code}: {r.text[:300]}")
        h, w = (int(v) for v in r.headers["X-Heatmap-Shape"].split(","))
        hm = np.frombuffer(r.content, "<f4").reshape(h, w)
        det = json.loads(r.headers["X-Explain-Detection"])
        query = int(r.headers.get("X-Explain-Query", -1))
        return hm, det, query


def tensor(inp):
    dt = {"float32": "<f4", "int64": "<i8"}[inp["dtype"]]
    return np.frombuffer(base64.b64decode(inp["data"]), dt).reshape(inp["shape"])


def decode_mask(m, w, h):
    return Mask.from_json(m).to_ndarray(w, h)


# --------------------------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------------------------

def tag_size(draw, text, size):
    l, t, r, b = draw.textbbox((0, 0), text, font=font(size, bold=True))
    return r - l + 6, b - t + 6


def draw_tag(draw, x, y, text, color, size=14):
    """A filled tag with white text, top-left corner at (x, y)."""
    f = font(size, bold=True)
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    draw.rectangle([x, y, x + r - l + 6, y + b - t + 6], fill=color)
    draw.text((x + 3 - l, y + 3 - t), text, font=f, fill="white")


def label_tag(draw, x, y, text, color, size=14):
    """A tag whose bottom-left sits at (x, y), or below y when there is no room above."""
    tw, th = tag_size(draw, text, size)
    draw_tag(draw, x, y - th if y - th >= 0 else y, text, color, size)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_boxes(img, dets, colors, width=3, size=14, label_fn=None):
    """Boxes first, then labels (most confident first), each moved to the first free spot:
    above the box, inside its top edge, below it, inside its bottom edge."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    for det in dets:
        x, y, w, h = det["bbox"]
        d.rectangle([x, y, x + w, y + h], outline=colors(det["class"]), width=width)
    placed = []
    for det in sorted(dets, key=lambda z: -z["conf"]):
        x, y, w, h = det["bbox"]
        text = label_fn(det) if label_fn else f'{det["class"]} {det["conf"]:.2f}'
        tw, th = tag_size(d, text, size)
        lx = min(max(x, 0), W - tw)
        cands = [(lx, y - th), (lx, y), (lx, y + h), (lx, y + h - th)]
        cands = [(cx, cy) for cx, cy in cands if 0 <= cy <= H - th] or [(lx, max(0, y))]
        rect = None
        for cx, cy in cands:
            r = (cx, cy, cx + tw, cy + th)
            if not any(_overlaps(r, p) for p in placed):
                rect = r
                break
        rect = rect or (cands[0][0], cands[0][1], cands[0][0] + tw, cands[0][1] + th)
        placed.append(rect)
        draw_tag(d, rect[0], rect[1], text, colors(det["class"]), size)
    return im


def overlay_masks(img, masks, colors_rgb, alpha=0.5, outline=True):
    """Blend boolean masks (H, W) onto img with the given RGB colours; draw a thin contour."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    for m, c in zip(masks, colors_rgb):
        arr[m] = arr[m] * (1 - alpha) + np.array(c, np.float32) * alpha
    if outline:
        for m, c in zip(masks, colors_rgb):
            edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            arr[edge] = c
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def caption_bar(img, text, size=15):
    """Append a white strip with a caption under the image."""
    f = font(size)
    lines = text.split("\n")
    lh = size + 6
    out = Image.new("RGB", (img.width, img.height + lh * len(lines) + 10), "white")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((8, img.height + 5 + i * lh), line, font=f, fill=INK2)
    return out


def hstack(images, gap=8, height=None):
    if height:
        images = [im.resize((round(im.width * height / im.height), height), Image.LANCZOS) for im in images]
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), "white")
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def fit_width(img, max_w=800):
    if img.width <= max_w:
        return img
    return img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)


def style_axes(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def barh(ax, labels, values, title, xlabel, color=PALETTE[0], xmax=None, fmt="{:.2f}"):
    """Horizontal bar chart, best at the top, value labels at the bar ends."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    xmax = xmax or max(values) * 1.25
    ax.set_xlim(0, xmax)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.015, yi, fmt.format(v), va="center", fontsize=9, color=INK2)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    style_axes(ax)


# --------------------------------------------------------------------------------------------
# The figure set
# --------------------------------------------------------------------------------------------

class Builder:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.srv = Server(args.server)
        self.entries = []
        self.skipped = []
        self.used_photos = set()
        with open(args.coco_annotations) as f:
            ann = json.load(f)
        self.coco = {im["id"]: im for im in ann["images"] if im["id"] in PHOTOS}
        lic = {l["id"]: l for l in ann["licenses"]}
        for pid in PHOTOS:
            name = lic[self.coco[pid]["license"]]["name"]
            if name != "Attribution License":
                raise SystemExit(f"COCO #{pid} is {name!r}, not CC BY 2.0 — refusing to use it")

    # -- inputs ---------------------------------------------------------------------------------

    def photo(self, pid):
        self.used_photos.add(pid)
        path = Path(self.args.coco_images) / f"{pid:012d}.jpg"
        raw = path.read_bytes()
        return Image.open(io.BytesIO(raw)).convert("RGB"), raw

    def credit(self, pid):
        return f"Photo: COCO val2017 #{pid}, Flickr {self.coco[pid]['flickr_url']}, CC BY 2.0"

    def source(self, pid):
        return {"dataset": "COCO val2017", "coco_id": pid, "file": f"{pid:012d}.jpg",
                "flickr_url": self.coco[pid]["flickr_url"], "license": "CC BY 2.0",
                "credit": self.credit(pid)}

    def need(self, *models):
        missing = [m for m in models if not self.srv.has(m)]
        if missing:
            self.skipped.append(", ".join(missing))
            print(f"  skip: model(s) not installed: {', '.join(missing)}")
            return False
        return True

    # -- output ---------------------------------------------------------------------------------

    def save(self, img, name, **entry):
        img = fit_width(img)
        path = self.out / name
        if name.endswith(".jpg"):
            img.save(path, quality=85, optimize=True, progressive=True)
        else:
            img.save(path, optimize=True)
        self.record(name, **entry)

    def record(self, name, **entry):
        size = (self.out / name).stat().st_size
        if size > 250 * 1024:
            print(f"  WARNING: {name} is {size // 1024} KB (> 250 KB)")
        self.entries.append({"file": name, **entry})
        print(f"  wrote {name} ({size // 1024} KB)")

    @staticmethod
    def run_info(res):
        return {"duration_ms": round(res.get("duration_ms", 0), 1), "device": res.get("device", "")}

    # -- 1. detection ---------------------------------------------------------------------------

    def detection(self):
        if not self.need("rf-detr"):
            return
        for pid in (29596, 372819):
            img, raw = self.photo(pid)
            res = self.srv.predict("rf-detr", raw)
            dets = res.get("detections", [])
            out = draw_boxes(img, dets, Colors())
            self.save(out, f"detect-rfdetr-{pid}.jpg", figure="Detection (RF-DETR, COCO classes)",
                      model="rf-detr", request={}, image=self.source(pid),
                      detections=len(dets), **self.run_info(res))

    # -- 2. prompted segmentation ---------------------------------------------------------------

    def segmentation(self):
        if not self.need("mobile-sam"):
            return
        # Box prompt: the cat on 177015.
        pid = 177015
        img, raw = self.photo(pid)
        box = (310, 175, 310, 195)  # x, y, w, h in original pixels
        res = self.srv.predict("mobile-sam", raw, box=",".join(map(str, box)))
        masks = [decode_mask(m, img.width, img.height) for m in res.get("masks", [])]
        out = overlay_masks(img, masks, [hex_rgb(PALETTE[0])] * len(masks))
        d = ImageDraw.Draw(out)
        x, y, w, h = box
        d.rectangle([x, y, x + w, y + h], outline=PALETTE[3], width=3)
        label_tag(d, x, y, "box prompt", PALETTE[3])
        self.save(out, f"segment-sam-box-{pid}.jpg", figure="Segmentation from a box prompt",
                  model="mobile-sam", request={"box": "%d,%d,%d,%d" % box},
                  image=self.source(pid), masks=len(masks),
                  mask_conf=[round(m["conf"], 3) for m in res.get("masks", [])], **self.run_info(res))

        # Point prompt: one elephant on 564133.
        pid = 564133
        img, raw = self.photo(pid)
        pt = (470, 195)  # on the head of the front elephant
        res = self.srv.predict("mobile-sam", raw, point=f"{pt[0]},{pt[1]},1")
        masks = [decode_mask(m, img.width, img.height) for m in res.get("masks", [])]
        out = overlay_masks(img, masks, [hex_rgb(PALETTE[0])] * len(masks))
        d = ImageDraw.Draw(out)
        r = 7
        d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=PALETTE[3], outline="white", width=2)
        text = "point prompt (x=%d, y=%d)" % pt
        tw, th = tag_size(d, text, 14)
        draw_tag(d, pt[0] - tw // 2, pt[1] - r - th - 30, text, PALETTE[3])
        d.line([(pt[0], pt[1] - r - 30), (pt[0], pt[1] - r)], fill=PALETTE[3], width=2)
        self.save(out, f"segment-sam-point-{pid}.jpg", figure="Segmentation from a point prompt",
                  model="mobile-sam", request={"point": f"{pt[0]},{pt[1]},1"},
                  image=self.source(pid), masks=len(masks),
                  mask_conf=[round(m["conf"], 3) for m in res.get("masks", [])], **self.run_info(res))

    # -- 3. automatic masks ---------------------------------------------------------------------

    def automask(self):
        if not self.need("mobile-sam"):
            return
        pid = 389381
        img, raw = self.photo(pid)
        res = self.srv.predict("mobile-sam", raw)
        ms = res.get("masks", [])
        masks = [decode_mask(m, img.width, img.height) for m in ms]
        # Largest first so small masks stay visible on top. Masks covering half the image or
        # more (the tablecloth behind everything) are left uncoloured so the objects read.
        area = img.width * img.height
        big = [i for i in range(len(masks)) if masks[i].sum() >= 0.5 * area]
        order = sorted((i for i in range(len(masks)) if i not in big), key=lambda i: -masks[i].sum())
        rng = random.Random(7)
        cols = [tuple(rng.randint(40, 255) for _ in range(3)) for _ in order]
        out = overlay_masks(img, [masks[i] for i in order], cols, alpha=0.6)
        out = hstack([img, out], gap=8)
        note = f"{len(ms)} masks from one request with no prompt"
        if big:
            note += f" ({len(big)} background-sized mask{'s' if len(big) > 1 else ''} left uncoloured)"
        out = caption_bar(fit_width(out), note)
        self.save(out, f"automask-sam-{pid}.jpg", figure="Automatic masks (no prompt)",
                  model="mobile-sam", request={}, image=self.source(pid), masks=len(ms),
                  masks_uncoloured_background=len(big), **self.run_info(res))

    # -- 4. open-vocabulary detection -----------------------------------------------------------

    def open_vocab(self):
        if not self.need("grounding-dino"):
            return
        for pid, prompt in ((177015, "cat. laptop. couch."),
                            (389381, "broccoli. carrot. kiwi fruit. fig. rice.")):
            img, raw = self.photo(pid)
            res = self.srv.predict("grounding-dino", raw, prompt=prompt)
            dets = res.get("detections", [])
            out = draw_boxes(img, dets, Colors())
            out = caption_bar(out, f'prompt: "{prompt}"')
            self.save(out, f"openvocab-gdino-{pid}.jpg", figure="Open-vocabulary detection from a text prompt",
                      model="grounding-dino", request={"prompt": prompt}, image=self.source(pid),
                      detections=len(dets), classes=sorted({d["class"] for d in dets}),
                      **self.run_info(res))

    # -- 5. grounded-sam ------------------------------------------------------------------------

    def grounded_sam(self):
        if not self.need("grounded-sam"):
            return
        pid, prompt = 372819, "dog. person. bench."
        img, raw = self.photo(pid)
        res = self.srv.predict("grounded-sam", raw, prompt=prompt)
        dets, ms = res.get("detections", []), res.get("masks", [])
        colors = Colors()
        # masks[i] belongs to detections[i] (one mask per box).
        masks = [decode_mask(m, img.width, img.height) for m in ms]
        cols = [hex_rgb(colors(dets[i]["class"])) if i < len(dets) else hex_rgb(PALETTE[0])
                for i in range(len(masks))]
        out = overlay_masks(img, masks, cols, alpha=0.5)
        out = draw_boxes(out, dets, colors, width=2, size=13)
        out = caption_bar(out, f'prompt: "{prompt}"')
        self.save(out, f"grounded-sam-{pid}.jpg", figure="Grounded-SAM: text prompt to masks",
                  model="grounded-sam", request={"prompt": prompt}, image=self.source(pid),
                  detections=len(dets), masks=len(ms), **self.run_info(res))

    # -- 6. depth -------------------------------------------------------------------------------

    def depth(self):
        model = "depth-anything-v2" if self.srv.has("depth-anything-v2") else "midas"
        if model == "midas" and not self.srv.has("depth-anything-v2"):
            self.skipped.append("depth-anything-v2 (weights not installed; midas used)")
        if not self.need(model):
            return
        for pid in (29596, 372819):
            img, raw = self.photo(pid)
            res = self.srv.predict(model, raw)
            dw, dh = res["depth_width"], res["depth_height"]
            dm = np.asarray(res["depth_map"], np.float32).reshape(dh, dw)
            fh = 3.9 * img.height / img.width + 0.4
            fig, axes = plt.subplots(1, 2, figsize=(8, fh), gridspec_kw={"wspace": 0.03})
            axes[0].imshow(img)
            axes[0].set_title("photo", fontsize=10, color=INK2, loc="left")
            im = axes[1].imshow(dm, cmap="magma", extent=(0, img.width, img.height, 0))
            axes[1].set_title(f"{model} relative depth ({dw}x{dh}, bright = near)", fontsize=10,
                              color=INK2, loc="left")
            for a in axes:
                a.axis("off")
            fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.34 / fh, bottom=0.02 / fh)
            self._save_chart(fig, f"depth-{model}-{pid}.jpg", figure="Monocular depth", model=model,
                             request={}, image=self.source(pid), depth_size=[dw, dh],
                             **self.run_info(res))

    def _save_chart(self, fig, name, **entry):
        """Save a matplotlib figure; one that contains a photo goes out as JPEG (`name`)."""
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, facecolor="white", format="png")
        plt.close(fig)
        img = Image.open(buf).convert("RGB")
        self.save(img, name, **entry)

    # -- 7. faces -------------------------------------------------------------------------------

    def faces(self):
        if not self.need("scrfd"):
            return
        for pid in (8021, 263969):
            img, raw = self.photo(pid)
            res = self.srv.predict("scrfd", raw)
            dets = res.get("detections", [])
            out = draw_boxes(img, dets, Colors(), width=2, size=12)
            self.save(out, f"faces-scrfd-{pid}.jpg", figure="Face detection", model="scrfd",
                      request={}, image=self.source(pid), detections=len(dets), **self.run_info(res))

    # -- 8. OCR ---------------------------------------------------------------------------------

    def ocr(self):
        if not self.need("paddle-ocr"):
            return
        lines = [("CORNER CAFE", 34, True), ("12 Market Street", 20, False),
                 ("Order #1047   2026-10-04 08:15", 18, False), ("", 10, False),
                 ("Flat white            4.50", 22, False), ("Croissant             3.20", 22, False),
                 ("Orange juice          3.80", 22, False), ("", 10, False),
                 ("TOTAL                11.50", 26, True), ("Thank you, see you soon!", 20, False)]
        W, H = 520, 470
        img = Image.new("RGB", (W, H), (246, 244, 238))
        d = ImageDraw.Draw(img)
        y = 26
        for text, size, bold in lines:
            if text:
                f = ImageFont.truetype(str(FONT_DIR / ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf")), size)
                l, t, r, b = d.textbbox((0, 0), text, font=f)
                d.text(((W - (r - l)) // 2 - l, y), text, font=f, fill=(30, 30, 30))
            y += size + 16
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        src = self.out / "ocr-input-receipt.png"
        img.save(src, optimize=True)
        res = self.srv.predict("paddle-ocr", buf.getvalue())
        dets = res.get("detections", [])
        colors = Colors()
        boxed = img.copy()
        dd = ImageDraw.Draw(boxed)
        for det in dets:
            x, yy, w, h = det["bbox"]
            dd.rectangle([x, yy, x + w, yy + h], outline=colors("text"), width=2)
        # Right panel: one row per text line (boxes grouped by vertical centre, left to right),
        # each detection's `class` (the recognised string) quoted, with its conf.
        rows = []
        for det in sorted(dets, key=lambda z: z["bbox"][1] + z["bbox"][3] / 2):
            yc = det["bbox"][1] + det["bbox"][3] / 2
            if rows and abs(rows[-1][0] - yc) < 10:
                rows[-1][1].append(det)
            else:
                rows.append([yc, [det]])
        panel = Image.new("RGB", (W, H), "white")
        pd = ImageDraw.Draw(panel)
        f, fs = font(15), font(12)
        pd.text((10, 4), "detections[].class  (conf)", font=font(13, True), fill=INK2)
        for yc, row in rows:
            x = 10
            for det in sorted(row, key=lambda z: z["bbox"][0]):
                s = f'"{det["class"]}"'
                pd.text((x, yc - 9), s, font=f, fill=INK)
                x += pd.textlength(s, font=f) + 4
                c = f'({det["conf"]:.2f})'
                pd.text((x, yc - 6), c, font=fs, fill=MUTED)
                x += pd.textlength(c, font=fs) + 18
        out = hstack([boxed, panel], gap=8)
        self.save(out, "ocr-paddle-receipt.png", figure="OCR (text detection + recognition)",
                  model="paddle-ocr", request={},
                  image={"source": "synthetic receipt rendered by website/tools/figures.py",
                         "file": "ocr-input-receipt.png", "license": "CC0 (generated)",
                         "credit": "Synthetic image, generated by website/tools/figures.py"},
                  detections=len(dets), texts=[d_["class"] for d_ in dets], **self.run_info(res))
        os.remove(src)  # the input is rendered again on every run; the figure shows it already

    # -- 9. classification ----------------------------------------------------------------------

    def classification(self):
        models = [m for m in ("efficientnet-b0", "mobilenet-v3") if self.srv.has(m)]
        if not models:
            self.need("efficientnet-b0", "mobilenet-v3")
            return
        pid = 564133
        img, raw = self.photo(pid)
        results = {m: self.srv.predict(m, raw) for m in models}
        fig = plt.figure(figsize=(8, 2.9))
        gs = fig.add_gridspec(len(models), 2, width_ratios=[1.05, 1], wspace=0.55, hspace=0.75)
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.imshow(img)
        ax0.axis("off")
        for i, m in enumerate(models):
            cl = results[m]["classifications"][:5]
            ax = fig.add_subplot(gs[i, 1])
            barh(ax, [c["class"] for c in cl], [c["conf"] for c in cl], f"{m} — top-5",
                 "probability" if i == len(models) - 1 else "", xmax=1.15)
        fig.subplots_adjust(left=0.01, right=0.97, top=0.9, bottom=0.14)
        self._save_chart(fig, f"classify-{pid}.jpg", figure="Image classification (ImageNet-1k, top-5)",
                         model=models, request={}, image=self.source(pid),
                         top5={m: [[c["class"], round(c["conf"], 4)] for c in results[m]["classifications"][:5]] for m in models},
                         duration_ms={m: round(results[m]["duration_ms"], 1) for m in models},
                         device=results[models[0]].get("device", ""))

    # -- 10. CLIP zero-shot ---------------------------------------------------------------------

    def zero_shot(self):
        if not self.need("clip", "clip-text"):
            return
        labels = ["a photo of dogs running on grass", "a photo of dogs sleeping on a sofa",
                  "a photo of people sitting on a bench", "a photo of a cat on a sofa",
                  "a photo of a kitchen", "a photo of elephants"]
        pid = 372819
        img, raw = self.photo(pid)
        ri = self.srv.predict("clip", raw)
        # clip-text embeds one row per '.'-separated phrase; the image part is ignored.
        rt = self.srv.predict("clip-text", raw, prompt=". ".join(labels))
        iv = np.asarray(ri["embeddings"][0], np.float64)
        tv = np.asarray(rt["embeddings"], np.float64)
        iv /= np.linalg.norm(iv)
        tv /= np.linalg.norm(tv, axis=1, keepdims=True)
        cos = tv @ iv
        logits = 100.0 * cos  # CLIP's learned logit scale (100) for ViT-B/32
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)
        fig = plt.figure(figsize=(8, 2.9))
        ax0 = fig.add_axes([0.0, 0.04, 0.36, 0.86])
        ax0.imshow(img)
        ax0.axis("off")
        ax = fig.add_axes([0.62, 0.17, 0.27, 0.68])
        lab = ["…" + labels[i].replace("a photo of", "") for i in order]
        barh(ax, lab, [cos[i] for i in order], "", "cosine similarity (image · text)",
             xmax=max(cos) * 1.3, fmt="{:.3f}")
        ax.text(1.03, 1.02, "p", transform=ax.transAxes, fontsize=9, color=INK2, fontweight="bold")
        for yi, i in zip(np.arange(len(order))[::-1], order):
            ax.text(1.03, yi, f"{p[i]:.2f}", va="center", fontsize=9, color=INK,
                    transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
        ax.tick_params(axis="y", labelsize=9.5)
        fig.text(0.38, 0.92, "CLIP zero-shot: captions “a photo of …”;  "
                 "p = softmax(100 · cosine)", fontsize=10.5, color=INK)
        self._save_chart(fig, f"zeroshot-clip-{pid}.jpg", figure="CLIP zero-shot classification",
                         model=["clip", "clip-text"],
                         request={"clip": {}, "clip-text": {"prompt": ". ".join(labels)}},
                         image=self.source(pid),
                         scores=[{"label": labels[i], "cosine": round(float(cos[i]), 4),
                                  "prob": round(float(p[i]), 4)} for i in order],
                         embedding_dim=int(iv.shape[0]),
                         duration_ms={"clip": round(ri["duration_ms"], 1), "clip-text": round(rt["duration_ms"], 1)},
                         device=ri.get("device", ""))

    # -- 11. grasp ------------------------------------------------------------------------------

    def grasp(self):
        if not self.need("grasp-rfdetr"):
            return
        pid = 389381
        img, raw = self.photo(pid)
        gmin = 25  # px; hides the narrowest jaw openings so the drawn grasps are visible
        res = self.srv.predict("grasp-rfdetr", raw, gripper_min=gmin)
        dets, ms, grasps = res.get("detections", []), res.get("masks", []), res.get("grasps", [])
        colors = Colors()
        masks = [decode_mask(m, img.width, img.height) for m in ms]
        cols = [hex_rgb(colors(dets[i]["class"])) if i < len(dets) else hex_rgb(PALETTE[0])
                for i in range(len(masks))]
        out = overlay_masks(img, masks, cols, alpha=0.35)
        out = draw_boxes(out, dets, colors, width=2, size=13)
        d = ImageDraw.Draw(out)
        # Best 3 grasps per object. A grasp is the jaw-closing line through (x, y) at angle
        # theta, `width` px long, with a short jaw plate drawn at each end.
        shown = []
        for cls in {g.get("class", "") for g in grasps}:
            best = sorted([g for g in grasps if g.get("class", "") == cls], key=lambda g: -g["quality"])[:3]
            shown += best
        for g in shown:
            c = colors(g.get("class", ""))
            dx, dy = math.cos(g["theta"]), math.sin(g["theta"])
            hw = g["width"] / 2
            p1 = (g["x"] - dx * hw, g["y"] - dy * hw)
            p2 = (g["x"] + dx * hw, g["y"] + dy * hw)
            d.line([p1, p2], fill="white", width=5)
            d.line([p1, p2], fill=c, width=3)
            jaw = 7
            for px, py in (p1, p2):
                d.line([(px - dy * jaw, py + dx * jaw), (px + dy * jaw, py - dx * jaw)], fill="white", width=6)
                d.line([(px - dy * jaw, py + dx * jaw), (px + dy * jaw, py - dx * jaw)], fill=c, width=4)
            d.ellipse([g["x"] - 3, g["y"] - 3, g["x"] + 3, g["y"] + 3], fill="white")
        out = caption_bar(out, f"{len(grasps)} grasps returned (gripper_min={gmin} px); best 3 per object drawn\n"
                               "line = closing direction through (x, y) at theta, length = width; bars = jaws", 13)
        self.save(out, f"grasp-rfdetr-{pid}.jpg", figure="Planar parallel-jaw grasps",
                  model="grasp-rfdetr", request={"gripper_min": gmin}, image=self.source(pid),
                  detections=len(dets),
                  masks=len(ms), grasps=len(grasps), grasps_drawn=len(shown), **self.run_info(res))

    # -- 12. background / support surface -------------------------------------------------------

    def background(self):
        if not self.need("background"):
            return
        for pid, method in ((363840, "sam"), (29596, "sam")):
            img, raw = self.photo(pid)
            res = self.srv.predict("background", raw, method=method)
            ms = res.get("masks", [])
            if not ms:
                print(f"  background: no mask on #{pid} ({method}); skipped")
                continue
            m = decode_mask(ms[0], img.width, img.height)
            bg = overlay_masks(img, [m], [hex_rgb(PALETTE[1])], alpha=0.55)
            fg = np.asarray(img).copy()
            fg[m] = (fg[m] * 0.15 + np.array([255, 255, 255]) * 0.85).astype(np.uint8)
            out = hstack([caption_bar(bg, "support surface (returned mask)", 13),
                          caption_bar(Image.fromarray(fg), "objects = complement of the mask", 13)])
            self.save(out, f"background-{method}-{pid}.jpg", figure="Background / support-surface mask",
                      model="background", request={"method": method}, image=self.source(pid),
                      masks=len(ms), **self.run_info(res))

    # -- 13. preprocessing modes ----------------------------------------------------------------

    def _tensor_image(self, pre, norm):
        t = tensor(pre["inputs"][0]).astype(np.float32)
        shape = list(t.shape)
        if t.ndim == 4:  # NCHW
            t = t[0].transpose(1, 2, 0)
        if norm:
            mean, std = NORM[norm]
            t = (t * np.array(std) + np.array(mean)) * 255.0
        return Image.fromarray(t.clip(0, 255).astype(np.uint8)), shape

    # The letterbox panel used rf-detr-nano, which letterboxed until 2026-10-05 although RF-DETR is
    # trained squashed (BUGS_TO_FIX.md #1, -3.16 mAP). Rather than show a model served wrongly, the
    # panel is a temporary copy of rf-detr-nano with letterbox: true (needs --inspect-models), and
    # the figure says it is an illustration.
    LETTERBOX_COPY = {"rf-detr-nano-letterbox": [("\n  letterbox: false\n", "\n  letterbox: true\n")]}

    def preprocessing(self):
        pid = 177015
        img, raw = self.photo(pid)
        made = []
        if self.args.inspect_models and self.srv.has("rf-detr-nano"):
            made = self._scratch_copies("rf-detr-nano", self.LETTERBOX_COPY)
        try:
            self._preprocessing(pid, img, raw, {d.name for d in made})
        finally:
            self._remove_copies(made)

    def _preprocessing(self, pid, img, raw, copies):
        modes = [("rf-detr", "stretch (squash) resize", "imagenet"),
                 ("rf-detr-nano-letterbox", "letterbox (illustration)", "imagenet"),
                 ("clip", "center crop", "clip"),
                 ("depth-anything-v2", "keep aspect, multiple of 14", "imagenet"),
                 ("mobile-sam", "longest side = 1024", None)]
        panels, info = [], []
        for model, mode, norm in modes:
            if model in self.LETTERBOX_COPY and model not in copies:
                self.skipped.append(f"{model} (preprocessing panel '{mode}': needs --inspect-models and rf-detr-nano)")
                print(f"  skip panel {mode}: the letterbox copy needs --inspect-models (a scratch registry) "
                      "holding rf-detr-nano")
                continue
            if model not in copies and not self.srv.has(model):
                self.skipped.append(f"{model} (preprocessing panel '{mode}')")
                print(f"  skip panel {mode}: {model} not installed")
                continue
            pre = self.srv.preprocess(model, raw)
            timg, shape = self._tensor_image(pre, norm)
            panels.append((timg, mode, model, shape))
            info.append({"model": model, "mode": mode, "tensor": pre["inputs"][0]["name"],
                         "shape": shape, "meta": pre.get("meta")})
        cells = [(img, "original photo", f"{img.width}x{img.height}")] + \
                [(t, mode, f"{model}  {shape}") for t, mode, model, shape in panels]
        cols = 3
        rows = math.ceil(len(cells) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(8, 3.05 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (timg, title, sub) in zip(axes.flat, cells):
            # Each tensor drawn inside the same square frame, so a tensor's own aspect ratio
            # (square for squash/letterbox/crop, 4:3 for longest-side) is what you see.
            side = max(timg.width, timg.height)
            ax.imshow(timg, extent=(0, timg.width, timg.height, 0))
            ax.add_patch(plt.Rectangle((0, 0), timg.width, timg.height, fill=False, ec=MUTED, lw=0.8))
            ax.set_xlim(-1, side + 1)
            ax.set_ylim(side + 1, -1)
            ax.set_title(f"{title}\n", fontsize=10.5, color=INK, fontweight="bold")
            ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
                    color=INK2)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.08, hspace=0.25)
        self._save_chart(fig, f"preprocess-modes-{pid}.jpg",
                         figure="The same photo as each model's input tensor (via /api/preprocess)",
                         model=[p[2] for p in panels], request={"endpoint": "/api/preprocess"},
                         image=self.source(pid), panels=info)

    # -- 14. box mapped back to original coordinates --------------------------------------------

    def box_mapping(self):
        # rf-detr-nano squashes (as trained): scale_x != scale_y on a non-square photo, no padding.
        model = "rf-detr-nano"
        if not self.need(model):
            return
        pid = 177015
        img, raw = self.photo(pid)
        pre = self.srv.preprocess(model, raw)
        meta = pre["meta"]
        timg, shape = self._tensor_image(pre, "imagenet")
        res = self.srv.predict(model, raw)
        dets = [d for d in res.get("detections", []) if d["class"] == "cat"] or res.get("detections", [])
        det = max(dets, key=lambda z: z["conf"])
        x, y, w, h = det["bbox"]
        sx, sy, px, py = meta["scale_x"], meta["scale_y"], meta["pad_x"], meta["pad_y"]
        # input = orig * scale + pad  (the inverse of what the server applied to its decode)
        ix, iy, iw, ih = x * sx + px, y * sy + py, w * sx, h * sy
        c = PALETTE[1]
        a = timg.copy()
        da = ImageDraw.Draw(a)
        da.rectangle([ix, iy, ix + iw, iy + ih], outline=c, width=2)
        b = img.copy()
        db = ImageDraw.Draw(b)
        db.rectangle([x, y, x + w, y + h], outline=c, width=3)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.85), gridspec_kw={"width_ratios": [384, 640 * 384 / 480]})
        axes[0].imshow(a)
        mode = "letterbox" if px or py else "squash"
        axes[0].set_title(f"model input {shape[2]}x{shape[3]} ({mode})", fontsize=10, color=INK, loc="left")
        axes[1].imshow(b)
        axes[1].set_title(f"original image {meta['orig_width']}x{meta['orig_height']}", fontsize=10, color=INK, loc="left")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
        axes[0].set_xlabel(f"input space  [x, y, w, h] = [{ix:.0f}, {iy:.0f}, {iw:.0f}, {ih:.0f}]",
                           fontsize=9.5, color=INK)
        axes[1].set_xlabel(f"original space  bbox = [{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}]   ({det['class']} {det['conf']:.2f})",
                           fontsize=9.5, color=INK)
        scale = f"scale = {sx:g}" if sx == sy else f"scale_x = {sx:g}, scale_y = {sy:g}"
        fig.text(0.5, 0.025,
                 f"meta: {scale}, pad_x = {px}, pad_y = {py};   original = (input - pad) / scale",
                 ha="center", fontsize=9.5, color=INK2)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.17, wspace=0.04)
        self._save_chart(fig, f"bbox-mapping-{pid}.jpg",
                         figure="A detection in model-input space and mapped back to the original image",
                         model=model, request={"endpoints": ["/api/preprocess", "/api/predict"]},
                         image=self.source(pid), detections=len(res.get("detections", [])),
                         meta=meta, input_box=[round(v, 1) for v in (ix, iy, iw, ih)],
                         original_box=[round(v, 1) for v in (x, y, w, h)], **self.run_info(res))

    # -- 15. explain (attention heatmap) ----------------------------------------------------------

    def explain(self):
        model = "rfdetr-small"  # RF-DETR small (COCO) with an `explain: type: attention` block
        if not self.need(model):
            return
        pid = 177015
        img, raw = self.photo(pid)
        pred = self.srv.predict(model, raw)
        cmap = matplotlib.colormaps["magma"]
        panels, shown = [], []
        for idx in (0, 1):
            hm, det, query = self.srv.explain_numpy(model, raw, detection_idx=idx)
            if hm.shape != (img.height, img.width):
                hm = np.asarray(Image.fromarray(hm).resize(img.size, Image.NEAREST))
            # Darken the photo and lay the heatmap over it, its opacity growing with the value
            # (square root, so the weaker attended tokens stay visible next to the peak).
            v = np.sqrt(np.clip(hm, 0, 1))
            base = np.asarray(img).astype(np.float32) * 0.5
            heat = cmap(0.25 + 0.75 * v)[..., :3] * 255.0
            a = v[..., None] * 0.9
            out = Image.fromarray((base * (1 - a) + heat * a).clip(0, 255).astype(np.uint8))
            out = draw_boxes(out, [det], lambda _c: PALETTE[2], width=3, size=15)
            out = caption_bar(out, f"detection_idx={idx}: {det['class']} {det['conf']:.2f} "
                                   f"(object query {query})", 17)
            panels.append(out)
            shown.append({"detection_idx": idx, "class": det["class"], "conf": round(det["conf"], 4),
                          "bbox": [round(v, 1) for v in det["bbox"]], "query": query})
        out = hstack(panels, gap=8)
        self.save(out, f"explain-rfdetr-{pid}.jpg",
                  figure="Explain: decoder cross-attention for one detection (/api/explain)",
                  model=model, request={"endpoint": "/api/explain", "format": "numpy",
                                        "detection_idx": [0, 1]},
                  image=self.source(pid), detections=len(pred.get("detections", [])),
                  explained=shown, **self.run_info(pred))

    # -- 16. inspect: the converter's tiers B1 / B2 / C against deliberately WRONG manifests ------

    # Copies of the rf-detr manifest with one preprocessing mistake each (weights symlinked).
    INSPECT_WRONG = {
        "rf-detr-letterbox": [("letterbox: false", "letterbox: true")],
        "rf-detr-mean05": [("mean: [0.485, 0.456, 0.406]", "mean: [0.5, 0.5, 0.5]"),
                           ("std:  [0.229, 0.224, 0.225]", "std:  [0.5, 0.5, 0.5]")],
    }

    def _inspect_registry(self):
        """Write the wrong copies next to models/rf-detr in the server's registry."""
        return self._scratch_copies("rf-detr", self.INSPECT_WRONG)

    def _scratch_copies(self, src_name, copies):
        """Write copies of models/<src_name>'s manifest, one edit list {name: [(old, new), ...]}
        each, into the server's registry (--inspect-models; the server picks new names up on first
        use), weights symlinked. Returns the directories written, for _remove_copies."""
        import yaml
        root = Path(self.args.inspect_models)
        src = root / src_name
        text = (src / "manifest.yaml").read_text()
        weights = src / yaml.safe_load(text)["model_file"]
        made = []
        for name, subs in copies.items():
            d = root / name
            if d.exists():
                raise SystemExit(f"{d} exists; remove it first (this script writes and removes it)")
            t = text.replace(f"name: {src_name}\n", f"name: {name}\n")
            for a, b in subs:
                if a not in t:
                    raise SystemExit(f"{src}/manifest.yaml has no {a!r}; update the copy's edits")
                t = t.replace(a, b)
            d.mkdir()
            made.append(d)
            (d / "manifest.yaml").write_text(t)
            for f in src.iterdir():
                if f.suffix == ".txt":
                    (d / f.name).write_bytes(f.read_bytes())
            os.symlink(weights.resolve(), d / weights.name)
        return made

    @staticmethod
    def _remove_copies(made):
        for d in made:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()

    def _inspect_bundle(self, name):
        """A converter Bundle describing models/<name> as the server reads it."""
        import yaml
        from visionserve.convert.common import Bundle
        root = Path(self.args.inspect_models)
        m = yaml.safe_load((root / name / "manifest.yaml").read_text())
        src = root / "rf-detr"
        labels = (src / m["labels"]).read_text().splitlines()
        inp = m["input"]
        return Bundle(name=name, task=m["task"], architecture=m["architecture"], license=m["license"],
                      width=inp["width"], height=inp["height"], onnx={"model": str(src / m["model_file"])},
                      letterbox=bool(inp.get("letterbox")), mean=inp["normalize"]["mean"],
                      std=inp["normalize"]["std"], postprocess=m["postprocess"], labels=labels)

    def inspect(self):
        if not self.args.inspect_models:
            print("  skip: pass --inspect-models DIR (the server's registry, with models/rf-detr in it)")
            self.skipped.append("inspect (no --inspect-models)")
            return
        made = self._inspect_registry()
        try:
            self._inspect_run()
        finally:
            self._remove_copies(made)

    def _inspect_run(self):
        import dataclasses

        from visionserve import Client
        from visionserve.convert.evaluate import load_eval, tier_c
        from visionserve.convert.reference import (ManifestReference, OnnxForward, UserReference,
                                                   load_reference_script)
        from visionserve.convert.report import Report, parse_thresholds
        from visionserve.convert.verify import Plan, load_images, tier_b1, tier_b2

        client = Client(self.args.server, timeout=600)
        th = parse_thresholds(None)
        pid = 177015
        img, _raw = self.photo(pid)
        photo = Path(self.args.coco_images) / f"{pid:012d}.jpg"
        images = load_images([Path(self.args.coco_images) / f"{p:012d}.jpg" for p in PHOTOS])
        # Tier B1's reference: the training-time preprocessing, as a --reference-script would give it.
        train = load_reference_script(HERE / "inspect_reference.py")["preprocess"]

        def train_bgr(pil):  # the same transform on a photo read by OpenCV (BGR channel order)
            return train(Image.fromarray(np.ascontiguousarray(np.asarray(pil.convert("RGB"))[..., ::-1])))

        cases = [("rf-detr", train, "correct manifest"),
                 ("rf-detr-letterbox", train, "manifest says letterbox: true"),
                 ("rf-detr-mean05", train, "manifest says mean = std = 0.5"),
                 ("rf-detr", train_bgr, "training read photos as BGR")]
        report = Report(models=[c[0] for c in cases])
        b1, rows = [], []
        for name, fn, what in cases:
            b = self._inspect_bundle(name)
            ref = UserReference(b, "input", fn, None, None, f"inspect_reference.py ({what})")
            t = tier_b1(Plan(b, "input", ref, None, None, b.onnx["model"]), client, name, images, th)
            t.title = f"B1 {name}: {what}"
            report.add(t)
            b1.append({"model": name, "case": what, "status": t.status, "summary": t.summary,
                       "diagnosis": t.metrics.get("diagnosis"), "notes": t.notes})
            if name != "rf-detr" or fn is train_bgr:
                srv = client.preprocess(name, photo).inputs["input"][0]
                rows.append((what, fn(img), srv, t, np.array(b.std, np.float32)))
        # B2 and C: the served answers vs the same ONNX on the CORRECT preprocessing, decoded in
        # Python like the Go side (an independent check of the Go decode and box mapping).
        good = self._inspect_bundle("rf-detr")
        ref = ManifestReference(good, "input", OnnxForward(good.onnx["model"]), "the exported ONNX (CPU)")
        plan = Plan(good, "input", None, ref, None, good.onnx["model"])
        ev = load_eval(self.args.coco_annotations, self.args.coco_images, self.args.inspect_eval_max)
        b2c = []
        for name in ("rf-detr", "rf-detr-letterbox"):
            t2 = tier_b2(plan, client, name, images, th)
            t2.title = f"B2 {name}"
            tc = tier_c(plan, client, dataclasses.replace(good, name=name), ev, 1.0)
            tc.title = f"C {name}"
            report.add(t2)
            report.add(tc)
            b2c.append({"model": name, "b2": {"status": t2.status, "summary": t2.summary},
                        "c": {"status": tc.status, "summary": tc.summary}})
        print(report.table())
        self._inspect_figure(rows, pid, b1, b2c, ev.source)

    def _inspect_figure(self, rows, pid, b1, b2c, eval_source):
        mean, std = (np.array(v, np.float32) for v in NORM["imagenet"])

        def as_seen(t):  # a tensor read back with the TRAINING normalisation: what the model "sees"
            return Image.fromarray(((t.transpose(1, 2, 0) * std + mean) * 255).clip(0, 255).astype(np.uint8))

        fig, axes = plt.subplots(len(rows), 3, figsize=(8, 2.75 * len(rows)), squeeze=False)
        for r, (what, ref, srv, t, srv_std) in enumerate(rows):
            # Gray levels exactly as tier B1 counts them: |Δ tensor| x the served manifest's std x 255.
            lv = (np.abs(srv - ref) * srv_std[:, None, None] * 255.0).mean(0)
            codes = ", ".join(t.metrics.get("diagnosis") or {}) or "none"
            for c, (im, title) in enumerate([(as_seen(ref), "reference (training)"),
                                             (as_seen(srv), "server (/api/preprocess)")]):
                axes[r, c].imshow(im)
                axes[r, c].set_title(title, fontsize=9.5, color=INK, loc="left")
            h = axes[r, 2].imshow(lv, cmap="magma", vmin=0, vmax=max(1.0, float(np.percentile(lv, 99.5))))
            axes[r, 2].set_title(f"|Δ| gray levels, mean {lv.mean():.1f}", fontsize=9.5, color=INK, loc="left")
            cb = fig.colorbar(h, ax=axes[r, 2], fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7.5, colors=INK2, length=0)
            cb.outline.set_visible(False)
            axes[r, 0].set_ylabel(f"{what}\ndiagnosis: {codes}", fontsize=9, color=INK)
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(GRID)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.02, wspace=0.08, hspace=0.22)
        self._save_chart(fig, f"inspect-b1-{pid}.jpg",
                         figure="Tier B1: the server's input tensor vs the training preprocessing, three "
                                "deliberate mistakes",
                         model=["rf-detr", *self.INSPECT_WRONG], request={"endpoint": "/api/preprocess"},
                         image=self.source(pid), b1=b1, b2_c=b2c, eval_set=str(eval_source))

    # -- write ----------------------------------------------------------------------------------

    def write_index(self, keep):
        path = self.out / "figures.json"
        old = []
        if path.exists():
            # Keep the other figures' entries (with --only), and always those another script wrote
            # (they carry "generated_by", e.g. clients_figures.py), minus any whose file is gone.
            old = [e for e in json.loads(path.read_text())["figures"]
                   if (keep or e.get("generated_by"))
                   and e["file"] not in {n["file"] for n in self.entries}
                   and (self.out / e["file"]).exists()]
        doc = {"generated_by": "website/tools/figures.py",
               "server": "VisionServe (see each entry's device)",
               "host": self.args.host_note,
               "duration_note": "duration_ms is the server-side time of a warm request",
               "figures": sorted(old + self.entries, key=lambda e: e["file"])}
        path.write_text(json.dumps(doc, indent=2) + "\n")
        self.all_figures = doc["figures"]
        print(f"wrote {path}")

    def write_credits(self):
        lines = ["# Photo credits", "",
                 "The example photos in these figures are from the COCO 2017 validation set and are",
                 "licensed by their Flickr authors under the Creative Commons Attribution 2.0 license",
                 "(CC BY 2.0, https://creativecommons.org/licenses/by/2.0/). They were run through",
                 "VisionServe and overlaid with its results (boxes, masks, grasps, depth).", ""]
        used = {}
        for e in self.all_figures:
            pid = e.get("image", {}).get("coco_id")
            if pid:
                used.setdefault(pid, []).append(e["file"])
        for pid in PHOTOS:
            if pid in used:
                lines.append(f"- {self.credit(pid)}  \n  used in: " + ", ".join(f"`{f}`" for f in used[pid]))
        lines += ["", "The OCR receipt is a synthetic image rendered by `website/tools/figures.py`.",
                  "Figures are regenerated with `website/tools/figures.py`, and the `clients-*` ones with",
                  "`website/tools/clients_figures.py` (usage at the top of each file).", ""]
        (self.out / "CREDITS.md").write_text("\n".join(lines))


STEPS = ["detection", "segmentation", "automask", "open_vocab", "grounded_sam", "depth", "faces",
         "ocr", "classification", "zero_shot", "grasp", "background", "preprocessing", "box_mapping",
         "explain", "inspect"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:11435")
    ap.add_argument("--coco-images", required=True, help="COCO val2017 image directory")
    ap.add_argument("--coco-annotations", required=True, help="instances_val2017.json (for credits)")
    ap.add_argument("--out", default=str(HERE.parent / "docs" / "assets" / "img"))
    ap.add_argument("--host-note", default="", help="free text recorded in figures.json, e.g. the GPU model")
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(STEPS))
    ap.add_argument("--inspect-models", default="", metavar="DIR",
                    help="inspect and preprocessing steps: the server's --models dir (a SCRATCH registry with "
                         "models/rf-detr and models/rf-detr-nano); they write temporary manifest copies there "
                         "(two deliberately wrong rf-detr, one letterboxed rf-detr-nano) and remove them after")
    ap.add_argument("--inspect-eval-max", type=int, default=200, metavar="N",
                    help="inspect step: COCO val images for tier C (default 200; 50 was too noisy: the "
                         "correct manifest measured -1.45 mAP there, +0.34 on 200)")
    args = ap.parse_args()
    steps = [s for s in args.only.split(",") if s] or STEPS
    b = Builder(args)
    for s in steps:
        print(f"[{s}]")
        getattr(b, s)()
    b.write_index(keep=bool(args.only))
    b.write_credits()
    if b.skipped:
        print("skipped:", "; ".join(sorted(set(b.skipped))))


if __name__ == "__main__":
    main()
