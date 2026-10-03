"""Tier C (optional, --eval): accuracy on the USER's labelled data, reference vs served.

  detection       COCO json + image dir -> COCO mAP@[.5:.95] and mAP@.5 (pycocotools), computed for
                  the original framework pipeline AND for the server, on the same images.
                  Model class names are mapped to COCO category NAMES (case/underscore-insensitive);
                  names that do not map are reported, and unmapped categories are excluded from the
                  mean (so they do not count as AP 0 for a class the model was never trained on).
  classification  an ImageFolder-style directory (sub-directory name = label) -> top-1 / top-5.

Both sides run at the manifest's conf_threshold: the number is "what users get", not the paper
mAP (which is computed at a ~0.001 threshold). The delta is what matters: served - reference.

--eval accepts: a COCO json file; a directory holding `_annotations.coco.json` (Roboflow / rfdetr
dataset layout; `test/`, `valid/` sub-directories are searched in that order); or an ImageFolder
directory. Images are looked up next to the json, in images/, ../images, ../<split> (COCO's
annotations/instances_val2017.json -> val2017/), or --eval-images DIR.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

from .common import log
from .constants import IMAGE_EXT
from .report import FAIL, PASS, SKIP, WARN, TierResult

_NA = {"n/a", "na", "", "background", "__background__", "no object"}


def norm_name(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


# Same object, different dataset spelling (PASCAL VOC names in many COCO-trained checkpoints'
# id2label, e.g. PekingU/rtdetr_*). Matching is on norm_name with spaces removed, then via these.
ALIASES = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch", "tvmonitor": "tv",
           "pottedplant": "potted plant", "diningtable": "dining table", "tv monitor": "tv"}


def _key(s: str) -> str:
    return norm_name(s).replace(" ", "")


@dataclasses.dataclass
class EvalSet:
    kind: str                    # detection | classification
    items: list                  # detection: [(path, image_id)]; classification: [(path, label)]
    source: str = ""
    coco: Optional[dict] = None  # detection ground truth restricted to `items`
    categories: Optional[Dict[int, str]] = None
    labels: Optional[List[str]] = None


def _find_ann(p: Path) -> Optional[Path]:
    if p.is_file() and p.suffix.lower() == ".json":
        return p
    if p.is_dir():
        if (p / "_annotations.coco.json").is_file():
            return p / "_annotations.coco.json"
        for sub in ("test", "valid", "val"):
            if (p / sub / "_annotations.coco.json").is_file():
                return p / sub / "_annotations.coco.json"
    return None


def load_eval(path, images_dir=None, max_n: int = 200) -> EvalSet:
    p = Path(path)
    ann = _find_ann(p)
    if ann is not None:
        return _load_coco(ann, images_dir, max_n)
    if p.is_dir():
        return _load_imagefolder(p, max_n)
    raise ValueError(f"--eval {p}: expected a COCO json, a dir with _annotations.coco.json, or an ImageFolder dir")


def _load_coco(ann: Path, images_dir, max_n) -> EvalSet:
    d = json.loads(ann.read_text())
    imgs = sorted(d.get("images") or [], key=lambda im: im["id"])
    if not imgs:
        raise ValueError(f"{ann}: no images")
    split = ann.stem.split("_")[-1]
    cands = ([Path(images_dir)] if images_dir else []) + [ann.parent, ann.parent / "images",
                                                         ann.parent.parent / "images", ann.parent.parent / split]
    root = None
    for c in cands:
        if (c / imgs[0]["file_name"]).is_file() or (c / Path(imgs[0]["file_name"]).name).is_file():
            root = c
            break
    if root is None:
        raise ValueError(f"cannot find the images of {ann} (tried {', '.join(str(c) for c in cands)}); "
                         "pass --eval-images DIR")
    items = []
    for im in imgs:
        f = root / im["file_name"]
        if not f.is_file():
            f = root / Path(im["file_name"]).name
        if f.is_file():
            items.append((f, im["id"]))
        if max_n and len(items) >= max_n:
            break
    keep = {i for _, i in items}
    anns = []
    for a in d.get("annotations") or []:
        if a["image_id"] not in keep:
            continue
        a = dict(a)
        # Renumber 1..N: pycocotools records a match as the ground truth's id and treats id 0 as
        # "unmatched" (Roboflow exports start at 0), and a missing id must not collide with another.
        a["id"] = len(anns) + 1
        a.setdefault("iscrowd", 0)
        a.setdefault("area", float(a["bbox"][2]) * float(a["bbox"][3]))
        anns.append(a)
    cats = {c["id"]: c["name"] for c in d.get("categories") or []}
    gt = {"images": [im for im in imgs if im["id"] in keep], "annotations": anns,
          "categories": d.get("categories") or [], "info": d.get("info", {}), "licenses": []}
    return EvalSet("detection", items, f"{ann} ({len(items)} images, {len(anns)} boxes)", gt, cats)


def _load_imagefolder(p: Path, max_n) -> EvalSet:
    subs = sorted(s for s in p.iterdir() if s.is_dir())
    per = {s.name: sorted(f for f in s.iterdir() if f.suffix.lower() in IMAGE_EXT) for s in subs}
    per = {k: v for k, v in per.items() if v}
    if not per:
        raise ValueError(f"--eval {p}: no COCO json and no class sub-directories with images")
    # round-robin over classes so a cap keeps every class represented
    items, i = [], 0
    while any(i < len(v) for v in per.values()) and (not max_n or len(items) < max_n):
        for k, v in per.items():
            if i < len(v) and (not max_n or len(items) < max_n):
                items.append((v[i], k))
        i += 1
    return EvalSet("classification", items, f"{p} ({len(items)} images, {len(per)} classes)", labels=list(per))


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------

def coco_map(gt: dict, results: List[dict], img_ids, cat_ids):
    """(mAP@[.5:.95], mAP@.5) in percent via pycocotools, over img_ids x cat_ids."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not results:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = copy.deepcopy(gt)
        g.createIndex()
        dt = g.loadRes(copy.deepcopy(results))
        e = COCOeval(g, dt, "bbox")
        e.params.imgIds = list(img_ids)
        e.params.catIds = list(cat_ids)
        e.evaluate()
        e.accumulate()
        e.summarize()
    return float(max(0.0, e.stats[0]) * 100), float(max(0.0, e.stats[1]) * 100)


def map_labels(model_labels, categories: Dict[int, str], aliased: Optional[list] = None):
    """-> (label -> category id, model labels without a category, categories without a label).
    Names match case/space/underscore-insensitively, then through ALIASES (recorded in `aliased`)."""
    by_key = {}
    for cid, name in categories.items():
        by_key.setdefault(_key(name), cid)
    mapping, unmapped = {}, []
    for lab in model_labels or []:
        if norm_name(lab) in _NA:
            continue
        cid = by_key.get(_key(lab))
        if cid is None and norm_name(lab) in ALIASES:
            cid = by_key.get(_key(ALIASES[norm_name(lab)]))
            if cid is not None and aliased is not None:
                aliased.append(f"{lab}->{categories[cid]}")
        if cid is None:
            unmapped.append(lab)
        else:
            mapping[lab] = cid
    covered = set(mapping.values())
    return mapping, unmapped, [categories[c] for c in categories if c not in covered]


def open_vocab_label(label: str, mapping: Dict[str, int]) -> Optional[str]:
    """The prompt phrase (a key of `mapping`) an open-vocabulary label stands for, or None.

    Exact match first; then spelling-insensitive (case, '_' / '-' / spaces: the BERT tokenizer
    decodes "water_bottle" as "water _ bottle"); then a label that MERGES several phrases ("cup
    water bottle") maps to the longest phrase it contains as whole words (the most specific)."""
    if label in mapping:
        return label
    by_key = {}
    for k in mapping:
        by_key.setdefault(_key(k), k)
    if _key(label) in by_key:
        return by_key[_key(label)]
    words = norm_name(label).split()
    best = None
    for k in mapping:
        kw = norm_name(k).split()
        if kw and any(words[i:i + len(kw)] == kw for i in range(len(words) - len(kw) + 1)):
            if best is None or len(kw) > len(norm_name(best).split()):
                best = k
    return best


def topk_hits(ranked: List[str], gt: str, k: int) -> int:
    return int(_key(gt) in [_key(x) for x in ranked[:k]])


def grade_drop(drop: float, max_drop: float) -> str:
    if drop > max_drop:
        return FAIL
    if drop > max_drop / 2:
        return WARN
    return PASS


# --------------------------------------------------------------------------------------------
# tier C
# --------------------------------------------------------------------------------------------

def tier_c(plan, client, bundle, ev: EvalSet, max_drop: float, prompt=None) -> TierResult:
    from PIL import Image
    ref = plan.b2
    name = bundle.name
    if ref is None:
        return TierResult("C", "accuracy", SKIP, "no reference predictor", model=name)
    task = bundle.task
    title = f"accuracy on {Path(str(ev.source).split(' (')[0]).name}"
    if ev.kind == "detection" and task in ("detection", "open_vocab"):
        labels = bundle.labels
        if task == "open_vocab":
            labels = [p.strip() for p in (prompt or "").split(".") if p.strip()]
        aliased = []
        mapping, unmapped, uncovered = map_labels(labels, ev.categories, aliased)
        if not mapping:
            return TierResult("C", title, FAIL, f"no model class maps to a dataset category by name (model: "
                              f"{(labels or [])[:5]}..., dataset: {list(ev.categories.values())[:5]}...)", model=name)
        res_ref, res_srv = [], []
        # Open-vocabulary references label boxes with DECODED prompt tokens: transformers'
        # GroundingDINO lower-cases them ("water_bottle" -> "water _ bottle") and merges several
        # phrases into one label ("cup water bottle"). Resolve those to a prompt phrase instead of
        # silently dropping the box (which would lower the reference mAP, not the served one).
        if task == "open_vocab":
            def resolve(c):
                return open_vocab_label(c, mapping)
        else:
            def resolve(c):
                return c if c in mapping else None
        resolved_ref = 0
        for n, (path, img_id) in enumerate(ev.items):
            with Image.open(path) as im:
                pil = im.convert("RGB")
            r = ref.predict(pil, prompt)
            s = client.predict(name, path, prompt=prompt, max_grasps_per_object=None)
            for d in r.detections or []:
                lab = resolve(d["cls"])
                if lab is not None:
                    resolved_ref += int(d["cls"] not in mapping)
                    res_ref.append({"image_id": img_id, "category_id": mapping[lab],
                                    "bbox": [float(v) for v in d["bbox"]], "score": float(d["conf"])})
            for d in s.detections:
                lab = resolve(d.cls)
                if lab is not None:
                    res_srv.append({"image_id": img_id, "category_id": mapping[lab],
                                    "bbox": [float(v) for v in d.bbox], "score": float(d.conf)})
            if (n + 1) % 25 == 0:
                log(f"  C: {n + 1}/{len(ev.items)} images")
        img_ids = [i for _, i in ev.items]
        cat_ids = sorted(set(mapping.values()))
        m_ref, m50_ref = coco_map(ev.coco, res_ref, img_ids, cat_ids)
        m_srv, m50_srv = coco_map(ev.coco, res_srv, img_ids, cat_ids)
        drop = m_ref - m_srv
        st = grade_drop(drop, max_drop)
        conf = (bundle.postprocess or {}).get("conf_threshold")
        summ = (f"mAP {m_ref:.2f} ref / {m_srv:.2f} served (Δ {m_srv - m_ref:+.2f}), mAP50 {m50_ref:.2f} / "
                f"{m50_srv:.2f} (Δ {m50_srv - m50_ref:+.2f}); {len(img_ids)} images, {len(cat_ids)} classes")
        notes = [f"both sides at conf_threshold {conf} (what users get), so absolute mAP is below a paper number "
                 "computed at ~0.001; the delta is the check (FAIL if the drop exceeds --max-map-drop "
                 f"{max_drop:g} points)"]
        if aliased:
            notes.append(f"matched through spelling aliases: {', '.join(aliased)}")
        if resolved_ref:
            notes.append(f"{resolved_ref} reference box label(s) were decoded / merged prompt tokens (e.g. lower-cased "
                         "or several phrases in one label) and were mapped back to their prompt phrase")
        if unmapped:
            notes.append(f"model classes with no dataset category (ignored): {unmapped[:12]}"
                         + (" ..." if len(unmapped) > 12 else ""))
        if uncovered:
            notes.append(f"dataset categories the model has no class for (excluded from the mean): {uncovered[:12]}"
                         + (" ..." if len(uncovered) > 12 else ""))
        return TierResult("C", title, st, summ, model=name, notes=notes, metrics={
            "kind": "detection", "images": len(img_ids), "classes": len(cat_ids), "conf_threshold": conf,
            "map_ref": m_ref, "map50_ref": m50_ref, "map_served": m_srv, "map50_served": m50_srv,
            "delta_map": m_srv - m_ref, "delta_map50": m50_srv - m50_ref, "max_map_drop": max_drop,
            "n_dets_ref": len(res_ref), "n_dets_served": len(res_srv), "unmapped_labels": unmapped,
            "uncovered_categories": uncovered})
    if ev.kind == "classification" and task == "classification":
        lab_by_key = {_key(l): l for l in (bundle.labels or [])}
        miss = sorted({g for _, g in ev.items if _key(g) not in lab_by_key})
        items = [(p, g) for p, g in ev.items if _key(g) in lab_by_key]
        if not items:
            return TierResult("C", title, FAIL, f"no folder name maps to a model label (folders {miss[:5]}...)",
                              model=name)
        h = {"r1": 0, "r5": 0, "s1": 0, "s5": 0}
        k_srv = 0
        for n, (path, g) in enumerate(items):
            with Image.open(path) as im:
                pil = im.convert("RGB")
            r = ref.predict(pil, None)
            ranked = [k for k, _ in sorted((r.probs or {}).items(), key=lambda kv: -kv[1])]
            s = client.predict(name, path, max_grasps_per_object=None)
            srv = [c.cls for c in sorted(s.classifications, key=lambda c: -c.conf)]
            k_srv = max(k_srv, len(srv))
            h["r1"] += topk_hits(ranked, g, 1)
            h["r5"] += topk_hits(ranked, g, 5)
            h["s1"] += topk_hits(srv, g, 1)
            h["s5"] += topk_hits(srv, g, 5)
            if (n + 1) % 50 == 0:
                log(f"  C: {n + 1}/{len(items)} images")
        N = len(items)
        r1, r5, s1, s5 = (100.0 * h[k] / N for k in ("r1", "r5", "s1", "s5"))
        st = grade_drop(r1 - s1, max_drop)
        summ = (f"top-1 {r1:.2f} ref / {s1:.2f} served (Δ {s1 - r1:+.2f}), top-5 {r5:.2f} / {s5:.2f}; {N} images")
        notes = []
        if k_srv < 5:
            notes.append(f"the server returns only its top-{k_srv} (manifest max_detections): served top-5 is a "
                         f"top-{k_srv}")
        if miss:
            notes.append(f"folders with no matching model label (skipped): {miss[:12]}")
        return TierResult("C", title, st, summ, model=name, notes=notes, metrics={
            "kind": "classification", "images": N, "top1_ref": r1, "top5_ref": r5, "top1_served": s1,
            "top5_served": s5, "delta_top1": s1 - r1, "max_drop": max_drop, "skipped_folders": miss})
    return TierResult("C", "accuracy", SKIP, f"a {ev.kind} eval set does not apply to a {task} model", model=name)
