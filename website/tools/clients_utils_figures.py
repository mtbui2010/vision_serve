#!/usr/bin/env python3
"""Regenerate the "Visualize results" figures of website/docs/clients/python.md from REAL runs.

Unlike figures.py and clients_figures.py, which draw with their own style, every picture here is
made by the Python SDK's own ``visionserve.visualize.draw`` (from clients/python in this
checkout), with the same calls the page shows, so a figure is exactly what a reader gets. Photos
are COCO val2017 (CC BY 2.0); figures.json and CREDITS.md are updated through figures.py (entries
carry "generated_by", so a full figures.py run keeps them).

Usage (needs numpy and Pillow; the server must have the models used below):

    python website/tools/clients_utils_figures.py \
        --server http://127.0.0.1:11435 \
        --coco-images /path/to/coco/val2017 \
        --coco-annotations /path/to/coco/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "clients" / "python"))
import figures as F  # noqa: E402
from visionserve import Client, select_target_grasp, select_target_object  # noqa: E402
from visionserve.visualize import draw  # noqa: E402

GENERATED_BY = "website/tools/clients_utils_figures.py"


class DrawBuilder(F.Builder):
    def __init__(self, args):
        super().__init__(args)
        self.client = Client(args.server)

    def record(self, name, **entry):
        super().record(name, generated_by=GENERATED_BY, **entry)

    @staticmethod
    def info(res):
        return {"duration_ms": round(res.duration_ms, 1), "device": res.device}

    def grounded_sam(self):
        """draw() on a Grounded-SAM result: mask fills, one labelled box per object."""
        if not self.need("grounded-sam"):
            return
        pid, prompt = 372819, "dog. bench."
        _, raw = self.photo(pid)
        res = self.client.predict("grounded-sam", raw, prompt=prompt)
        self.save(draw(res, raw), f"clients-draw-grounded-sam-{pid}.jpg",
                  figure="draw(res, photo) on a Grounded-SAM result", model="grounded-sam",
                  request={"prompt": prompt}, image=self.source(pid),
                  detections=len(res.detections), masks=len(res.masks), **self.info(res))

    def automask(self):
        """Automatic masks drawn with the default boxes and with mask_boxes=False."""
        if not self.need("mobile-sam"):
            return
        pid = 389381
        _, raw = self.photo(pid)
        res = self.client.predict("mobile-sam", raw)
        a = F.caption_bar(draw(res, raw), "draw(auto, photo): %d masks" % len(res.masks), 20)
        b = F.caption_bar(draw(res, raw, mask_boxes=False), "draw(auto, photo, mask_boxes=False)", 20)
        self.save(F.hstack([a, b], gap=10), f"clients-draw-automask-{pid}.jpg",
                  figure="automatic masks with and without their boxes", model="mobile-sam",
                  request={}, image=self.source(pid), masks=len(res.masks), **self.info(res))

    def grasp(self):
        """A grasp result with the chosen object and grasp in red."""
        if not self.need("grasp-rfdetr"):
            return
        pid = 389381
        _, raw = self.photo(pid)
        res = self.client.predict("grasp-rfdetr", raw)
        obj = select_target_object(res, cls="broccoli")
        best = select_target_grasp(res.grasps, cls="broccoli")
        img = draw(res, raw, target_box=obj, target_grasp=best, max_grasps_per_object=1)
        self.save(img, f"clients-draw-grasp-{pid}.jpg",
                  figure="draw() on a grasp result, target object and grasp in red",
                  model="grasp-rfdetr", request={"max_grasps_per_object": 3},
                  image=self.source(pid), grasps=len(res.grasps), **self.info(res))

    def depth(self):
        """The photo and draw()'s depth picture, stretched to the photo's size, side by side."""
        if not self.need("midas"):
            return
        pid = 372819
        photo, raw = self.photo(pid)
        res = self.client.predict("midas", raw)
        pic = draw(res, raw).resize(photo.size)
        both = Image.new("RGB", (photo.width * 2 + 8, photo.height), "white")
        both.paste(photo, (0, 0))
        both.paste(pic, (photo.width + 8, 0))
        self.save(both, f"clients-draw-depth-{pid}.jpg",
                  figure="a photo and draw() of its midas depth map (red = near)", model="midas",
                  request={}, image=self.source(pid),
                  depth_size=[res.depth_width, res.depth_height], **self.info(res))


STEPS = ["grounded_sam", "automask", "grasp", "depth"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:11435")
    ap.add_argument("--coco-images", required=True, help="COCO val2017 image directory")
    ap.add_argument("--coco-annotations", required=True, help="instances_val2017.json (for credits)")
    ap.add_argument("--out", default=str(F.HERE.parent / "docs" / "assets" / "img"))
    ap.add_argument("--host-note", default="", help="free text recorded in figures.json")
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(STEPS))
    args = ap.parse_args()
    b = DrawBuilder(args)
    for s in [s for s in args.only.split(",") if s] or STEPS:
        print(f"[{s}]")
        getattr(b, s)()
    b.write_index(keep=True)
    b.write_credits()


if __name__ == "__main__":
    main()
