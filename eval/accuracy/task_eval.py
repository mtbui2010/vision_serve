"""Task-accuracy measured THROUGH the running VisionServe server (W6 Part B).

Re-evaluates one metric per task by hitting the HTTP ``/api/predict`` API of a live server, so
the number reflects the *served* pipeline (preprocess + ORT + postprocess + wire), not a notebook.
Report |served - published| in the paper.

Subcommands:
    imagenet   ImageNet-val Top-1 (classification)        [implemented]
    coco       COCO mAP@[.5:.95] via pycocotools (detection) [needs class->id map: TODO]
    miou       mean IoU for segmentation (GT-box prompts)  [skeleton: TODO loader]
    widerface  WiderFace easy AP (SCRFD face detection)    [skeleton: TODO loader]

Class names for ImageNet come from the repo file:
    internal/catalog/labels/imagenet1k.txt   (1000 classes; last line has no trailing newline)

Usage:
    python -m eval.accuracy.task_eval imagenet \
        --target http://localhost:11435 --model mobilenet-v3 \
        --images /data/imagenet/val --gt /data/imagenet/val_gt.txt \
        --labels internal/catalog/labels/imagenet1k.txt --limit 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Optional

from eval.common.api import build_json_request, encode_image, parse_result
from eval.common.provenance import build_meta, write_meta

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _post_predict(target: str, model: str, image_path: str,
                  prompt: Optional[str] = None, box: Optional[str] = None):
    body = json.dumps(build_json_request(model, encode_image(image_path),
                                         prompt=prompt, box=box)).encode()
    req = urllib.request.Request(  # noqa: S310 - localhost benchmarking
        target.rstrip("/") + "/api/predict", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        return parse_result(r.read())


def load_imagenet_labels(path: str) -> list[str]:
    """Load the 1000 ImageNet class names. The repo file's last line lacks a newline, so we
    split on newline and keep all non-empty lines (a plain readlines() also works)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    labels = [ln for ln in text.split("\n") if ln != ""]
    if len(labels) != 1000:
        print(f"WARNING: expected 1000 ImageNet labels, got {len(labels)} from {path}",
              file=sys.stderr)
    return labels


def _load_imagenet_gt(gt_path: str) -> dict[str, int]:
    """Ground truth file: lines of '<image_basename> <class_index>'. Returns {basename: idx}.

    TODO: requires a val ground-truth file in this format. The canonical ImageNet devkit ships
    a per-image class index list; convert it to '<filename> <idx>' once and pass it here.
    """
    gt: dict[str, int] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                gt[os.path.basename(parts[0])] = int(parts[1])
    if not gt:
        raise ValueError(f"no ground-truth rows parsed from {gt_path!r} "
                         "(expected '<image> <class_index>' per line)")
    return gt


def eval_imagenet(args: argparse.Namespace) -> dict[str, object]:
    """ImageNet-val Top-1 via the running server. Top-1 = served top class matches GT index."""
    labels = load_imagenet_labels(args.labels)
    gt = _load_imagenet_gt(args.gt)
    files = sorted(
        f for f in os.listdir(args.images) if f.lower().endswith(IMAGE_EXTS)
    )
    if args.limit:
        files = files[: args.limit]

    correct = total = skipped = 0
    device = ""
    for fn in files:
        base = os.path.basename(fn)
        if base not in gt:
            skipped += 1
            continue
        res = _post_predict(args.target, args.model, os.path.join(args.images, fn))
        device = device or res.device
        pred = res.top1()
        if pred is None:
            skipped += 1
            continue
        gt_label = labels[gt[base]] if 0 <= gt[base] < len(labels) else None
        total += 1
        if gt_label is not None and pred == gt_label:
            correct += 1

    top1 = correct / total if total else 0.0
    return {
        "task": "classification", "metric": "top1", "value": top1,
        "n": total, "skipped": skipped, "device_reported": device,
    }


def eval_coco(args: argparse.Namespace) -> dict[str, object]:
    """COCO mAP@[.5:.95] via pycocotools, predictions gathered through the server.

    Implemented end-to-end EXCEPT the class-name -> COCO-category-id mapping, which depends on
    the label file the model uses (rf-detr ships coco91.txt; see models/rf-detr/manifest.yaml).
    """
    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pycocotools required for COCO mAP; pip install pycocotools") from exc

    coco_gt = COCO(args.gt)  # instances_val2017.json
    img_ids = coco_gt.getImgIds()
    if args.limit:
        img_ids = img_ids[: args.limit]

    # TODO: requires a mapping from the served Detection.class string -> COCO category_id.
    # Build it from the model's label file (e.g. internal/catalog/labels/coco91.txt) once.
    name_to_catid: dict[str, int] = _coco_name_to_catid(coco_gt)

    detections: list[dict] = []
    device = ""
    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        path = os.path.join(args.images, info["file_name"])
        res = _post_predict(args.target, args.model, path)
        device = device or res.device
        for det in res.detections:
            cat = name_to_catid.get(det.get("class", ""))
            if cat is None:
                continue  # unmapped class -> skip rather than guess a category id
            x, y, w, h = det["bbox"]  # VisionServe bbox is [x,y,w,h] in original coords
            detections.append({
                "image_id": img_id, "category_id": cat,
                "bbox": [x, y, w, h], "score": float(det.get("conf", 0.0)),
            })

    if not detections:
        return {"task": "detection", "metric": "mAP", "value": None,
                "note": "no mappable detections (fill the class->catid map / check the model)",
                "device_reported": device}

    coco_dt = coco_gt.loadRes(detections)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"task": "detection", "metric": "mAP@[.5:.95]", "value": float(ev.stats[0]),
            "n": len(img_ids), "device_reported": device}


def _coco_name_to_catid(coco_gt) -> dict[str, int]:
    """Map COCO category *names* to ids. The served class strings must match these names; if the
    model uses a different label set, supply your own mapping here (TODO per model)."""
    return {c["name"]: c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}


def eval_miou(args: argparse.Namespace) -> dict[str, object]:
    """mean IoU for segmentation (MobileSAM etc.) with GT-box prompts, through the server.

    TODO: requires a dataset of (image, GT mask, GT box) triples and an RLE-decode of the served
    column-major RLE masks to compare against GT. The server call shape (box prompt) is shown.
    """
    # Example of the served call once the dataset loader is supplied:
    #   res = _post_predict(args.target, args.model, image_path, box="x,y,w,h")
    #   served_mask = decode_column_major_rle(res.masks[0]["rle"], H, W)
    #   iou = intersection(served_mask, gt_mask) / union(...)
    raise NotImplementedError(
        "eval_miou is a TODO: supply a (image, GT mask, GT box) dataset loader and a "
        "column-major RLE decoder matching pkg/api Mask.rle. Do not fabricate mIoU."
    )


def eval_widerface(args: argparse.Namespace) -> dict[str, object]:
    """WiderFace easy AP for SCRFD face detection, through the server.

    TODO: requires the WiderFace val images + GT and the official AP evaluation protocol.
    """
    raise NotImplementedError(
        "eval_widerface is a TODO: supply WiderFace val + the official AP evaluator. "
        "Do not fabricate AP."
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["imagenet", "coco", "miou", "widerface"])
    ap.add_argument("--target", default="http://localhost:11435")
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="image directory")
    ap.add_argument("--gt", help="ground-truth file/json (per-task)")
    ap.add_argument("--labels", default="internal/catalog/labels/imagenet1k.txt",
                    help="class label file (imagenet task)")
    ap.add_argument("--limit", type=int, default=0, help="cap images (0 = all)")
    ap.add_argument("--out", default=None, help="optional JSON result path")
    args = ap.parse_args(argv)

    dispatch = {
        "imagenet": eval_imagenet,
        "coco": eval_coco,
        "miou": eval_miou,
        "widerface": eval_widerface,
    }
    if args.task in ("imagenet", "coco") and not args.gt:
        ap.error(f"--gt is required for task '{args.task}'")

    result = dispatch[args.task](args)
    print(json.dumps(result, indent=2))

    if args.out:
        write_meta(args.out, build_meta(
            experiment=f"task_eval:{args.task} (W6 Part B)",
            target=args.target, model=args.model,
            device_reported=str(result.get("device_reported", "")),
            n_requests=int(result.get("n", 0) or 0),
            dataset=args.images,
        ))
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
