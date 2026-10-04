#!/usr/bin/env python3
"""Regenerate the figures of the Clients pages (website/docs/clients/) from REAL VisionServe runs.

Each figure shows what one request option changes, drawn from the JSON a running server returned
for a COCO val2017 photo (CC BY 2.0). It reuses figures.py: the same photos, drawing style,
figures.json index and CREDITS.md (entries written here carry "generated_by", so a full
figures.py run keeps them).

Usage (same server and data as figures.py):

    python website/tools/clients_figures.py \
        --server http://127.0.0.1:11640 \
        --coco-images /path/to/coco/val2017 \
        --coco-annotations /path/to/coco/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures as F  # noqa: E402

GENERATED_BY = "website/tools/clients_figures.py"


class ClientsBuilder(F.Builder):
    def record(self, name, **entry):
        super().record(name, generated_by=GENERATED_BY, **entry)

    def box_threshold(self):
        """grounding-dino at the default box_threshold (0.30) and at 0.40."""
        if not self.need("grounding-dino"):
            return
        pid, prompt = 177015, "cat. laptop. couch. remote control."
        img, raw = self.photo(pid)
        panels, counts, colors = [], {}, F.Colors()
        for bt in (None, 0.4):
            fields = {"prompt": prompt, **({"box_threshold": bt} if bt else {})}
            res = self.srv.predict("grounding-dino", raw, **fields)
            dets = res.get("detections", [])
            counts[str(bt or "default")] = len(dets)
            out = F.draw_boxes(img, dets, colors, width=4, size=20)
            label = "box_threshold=%s" % (bt if bt else "0.30 (default)")
            panels.append(F.caption_bar(out, "%s: %d boxes" % (label, len(dets)), 22))
        self.save(F.hstack(panels, gap=10), f"clients-box-threshold-{pid}.jpg",
                  figure="box_threshold: the same prompt at the default and at 0.40",
                  model="grounding-dino", request={"prompt": prompt, "box_threshold": [None, 0.4]},
                  image=self.source(pid), detections=counts, **self.run_info(res))

    def roi(self):
        """rf-detr on the whole photo, and with roi= (the model sees only the crop)."""
        if not self.need("rf-detr"):
            return
        pid, roi = 372819, (200, 100, 260, 240)
        img, raw = self.photo(pid)
        colors = F.Colors()
        full = self.srv.predict("rf-detr", raw)
        crop = self.srv.predict("rf-detr", raw, roi=",".join(map(str, roi)))
        a = F.caption_bar(F.draw_boxes(img, full.get("detections", []), colors, width=3, size=13),
                          "no roi: %d detections" % len(full.get("detections", [])), 20)
        b = img.copy()  # the dashed ROI rectangle first, so the labels stay on top of it
        d = ImageDraw.Draw(b)
        x, y, w, h = roi
        for i in range(x, x + w, 12):
            d.line([(i, y), (min(i + 6, x + w), y)], fill=F.INK2, width=3)
            d.line([(i, y + h), (min(i + 6, x + w), y + h)], fill=F.INK2, width=3)
        for j in range(y, y + h, 12):
            d.line([(x, j), (x, min(j + 6, y + h))], fill=F.INK2, width=3)
            d.line([(x + w, j), (x + w, min(j + 6, y + h))], fill=F.INK2, width=3)
        b = F.draw_boxes(b, crop.get("detections", []), colors, width=3, size=13)
        b = F.caption_bar(b,"roi=[200, 100, 260, 240]: %d detections" % len(crop.get("detections", [])), 20)
        self.save(F.hstack([a, b], gap=10), f"clients-roi-{pid}.jpg",
                  figure="roi: the model runs on the dashed crop; boxes come back in photo pixels",
                  model="rf-detr", request={"roi": list(roi)}, image=self.source(pid),
                  detections={"full": len(full.get("detections", [])),
                              "roi": len(crop.get("detections", []))}, **self.run_info(crop))

    def dilate(self):
        """One grounded-sam mask at dilate=-3, 0 and 5, zoomed in."""
        if not self.need("grounded-sam"):
            return
        pid, prompt = 372819, "dog."
        img, raw = self.photo(pid)
        zoom = (186, 200, 302, 345)  # around the dog at the bottom left
        target = [216, 227, 58, 92]  # its detection box (rounded), to pick the same mask each time
        panels, areas = [], {}
        for dl in (-3, 0, 5):
            res = self.srv.predict("grounded-sam", raw, prompt=prompt, **({"dilate": dl} if dl else {}))
            dets = res.get("detections", [])
            i = min(range(len(dets)), key=lambda k: sum(abs(a - b) for a, b in zip(dets[k]["bbox"], target)))
            m = F.decode_mask(res["masks"][i], img.width, img.height)
            areas[str(dl)] = int(m.sum())
            out = F.overlay_masks(img, [m], [F.hex_rgb(F.PALETTE[0])], alpha=0.45)
            out = out.crop(zoom)
            out = out.resize((out.width * 5 // 2, out.height * 5 // 2), Image.NEAREST)
            panels.append(F.caption_bar(out, "dilate=%d: %d px" % (dl, m.sum()), 20))
        self.save(F.hstack(panels, gap=10), f"clients-dilate-{pid}.jpg",
                  figure="dilate: the same mask shrunk by 3 px, as returned, grown by 5 px",
                  model="grounded-sam", request={"prompt": prompt, "dilate": [-3, 0, 5]},
                  image=self.source(pid), mask_pixels=areas, **self.run_info(res))


STEPS = ["box_threshold", "roi", "dilate"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:11435")
    ap.add_argument("--coco-images", required=True, help="COCO val2017 image directory")
    ap.add_argument("--coco-annotations", required=True, help="instances_val2017.json (for credits)")
    ap.add_argument("--out", default=str(F.HERE.parent / "docs" / "assets" / "img"))
    ap.add_argument("--host-note", default="", help="free text recorded in figures.json")
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(STEPS))
    args = ap.parse_args()
    b = ClientsBuilder(args)
    for s in [s for s in args.only.split(",") if s] or STEPS:
        print(f"[{s}]")
        getattr(b, s)()
    b.write_index(keep=True)
    b.write_credits()


if __name__ == "__main__":
    main()
