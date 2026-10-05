#!/usr/bin/env python3
"""Make a small labelled set from COCO val2017 for the hands-on notebooks.

`visionserve optimize --labels` (and `check --labels`) need labelled photos: a COCO
annotations json plus the images it names. The full COCO val2017 set has 5000 images; this
script keeps the first N images (by image id) that have at least one object, skips the
hands-on example photos, links (or copies) them into OUT/images and writes OUT/instances.json
with only their annotations. Categories are kept as they are (the 80 COCO classes, ids 1..90).

    python tools/make_coco_subset.py --coco-json COCO/annotations/instances_val2017.json \
        --coco-images COCO/val2017 --out ./coco-subset --n 100

Get COCO val2017 from https://cocodataset.org/#download (val2017.zip, annotations_trainval2017.zip).
The images are under CC BY licences (see the COCO terms of use); this script only selects them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# The 8 hands-on photos (hands-on/images) are COCO val2017 images too. They are used for
# calibration, so they are kept OUT of the labelled set.
HANDS_ON_IDS = {177015, 372819, 389381, 29596, 363840, 564133, 8021, 34873}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coco-json", required=True, help="instances_val2017.json")
    ap.add_argument("--coco-images", required=True, help="the val2017 image folder")
    ap.add_argument("--out", required=True, help="output folder (gets images/ and instances.json)")
    ap.add_argument("--n", type=int, default=100, help="number of images (default 100)")
    ap.add_argument("--copy", action="store_true", help="copy the images (default: symbolic links)")
    a = ap.parse_args(argv)

    data = json.loads(Path(a.coco_json).read_text())
    with_objects = {ann["image_id"] for ann in data["annotations"] if not ann.get("iscrowd", 0)}
    images = sorted((im for im in data["images"]
                     if im["id"] in with_objects and im["id"] not in HANDS_ON_IDS),
                    key=lambda im: im["id"])[: a.n]
    keep = {im["id"] for im in images}

    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(a.coco_images)
    for im in images:
        src, dst = src_dir / im["file_name"], img_dir / im["file_name"]
        if not src.is_file():
            print(f"error: {src} not found", file=sys.stderr)
            return 1
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if a.copy:
            shutil.copyfile(src, dst)
        else:
            os.symlink(src.resolve(), dst)

    subset = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in keep],
        "categories": data["categories"],
    }
    (out / "instances.json").write_text(json.dumps(subset))
    print(f"{len(images)} images, {len(subset['annotations'])} objects -> {out}/instances.json, {img_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
