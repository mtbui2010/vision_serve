"""Tier B2 comparison metrics: served output vs the original framework's output on the same image.

numpy only (no torch / onnx), so it is unit-tested offline. Detections everywhere are dicts
{"cls": str, "conf": float, "bbox": [x, y, w, h]} in ORIGINAL image pixels — the SDK's Detection
schema; `as_dets` converts SDK Detection objects.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def as_dets(items) -> List[dict]:
    out = []
    for d in items or []:
        if isinstance(d, dict):
            out.append({"cls": str(d["cls"]), "conf": float(d["conf"]), "bbox": [float(v) for v in d["bbox"]]})
        else:  # visionserve.types.Detection
            out.append({"cls": str(d.cls), "conf": float(d.conf), "bbox": [float(v) for v in d.bbox]})
    return out


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = max(0.0, a[2]) * max(0.0, a[3]) + max(0.0, b[2]) * max(0.0, b[3]) - inter
    return inter / union if union > 0 else 0.0


def same_label(ref_cls: str, srv_cls: str) -> bool:
    return ref_cls == srv_cls


def phrase_in_merged_label(ref_cls: str, srv_cls: str) -> bool:
    """Open-vocabulary label equivalence for GroundingDINO.

    transformers' post_process_grounded_object_detection labels a box with EVERY prompt token above
    text_threshold, so one box can come back as "cup water bottle" (tokens from two phrases), while
    VisionServe assigns each box the single phrase it scores highest. The served label is therefore
    correct when it is one of the phrases contained, as whole words, in the reference label."""
    if ref_cls == srv_cls:
        return True
    r, s = ref_cls.lower().split(), srv_cls.lower().split()
    return bool(s) and any(r[k:k + len(s)] == s for k in range(len(r) - len(s) + 1))


def match_detections(ref: List[dict], srv: List[dict], iou_thr: float = 0.5,
                     label_match: Callable[[str, str], bool] = same_label):
    """Greedy one-to-one matching: every (ref, srv) pair whose labels match (`label_match`, default
    equality) and IoU >= iou_thr, taken in decreasing IoU. Returns (pairs [(i, j, iou)],
    unmatched_ref [i], unmatched_srv [j])."""
    cand = []
    for i, r in enumerate(ref):
        for j, s in enumerate(srv):
            if not label_match(r["cls"], s["cls"]):
                continue
            v = iou_xywh(r["bbox"], s["bbox"])
            if v >= iou_thr:
                cand.append((v, i, j))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_r, used_s, pairs = set(), set(), []
    for v, i, j in cand:
        if i in used_r or j in used_s:
            continue
        used_r.add(i)
        used_s.add(j)
        pairs.append((i, j, v))
    return (pairs, [i for i in range(len(ref)) if i not in used_r],
            [j for j in range(len(srv)) if j not in used_s])


def _xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], np.float64)


def compare_detections(images: List[Tuple[List[dict], List[dict], Tuple[int, int]]],
                       conf_threshold: Optional[float] = None, iou_thr: float = 0.5,
                       boundary: float = 0.05,
                       label_match: Callable[[str, str], bool] = same_label) -> dict:
    """Aggregate detection agreement over images [(ref_dets, srv_dets, (width, height))].

    A detection found on only one side whose conf is within `boundary` of `conf_threshold` is a
    threshold-boundary flip (0.501 on one side, 0.499 on the other) and is counted separately,
    not as a disagreement. matched_frac = matched / max(effective ref count, effective srv count),
    where effective = matched + unmatched above the boundary band; None when both are empty
    (nothing was detected — the check is vacuous and must be reported as such)."""
    n_ref = n_srv = matched = 0
    um_ref = um_srv = bnd_ref = bnd_srv = 0
    dconf, box_mean, box_max, box_rel, cls_conf = [], [], [], [], []
    merged = 0  # matched only through label_match, not by identical labels
    hi = (conf_threshold + boundary) if conf_threshold is not None else None
    for ref, srv, (w, h) in images:
        ref, srv = as_dets(ref), as_dets(srv)
        n_ref += len(ref)
        n_srv += len(srv)
        pairs, ur, us = match_detections(ref, srv, iou_thr, label_match)
        matched += len(pairs)
        merged += sum(1 for i, j, _ in pairs if ref[i]["cls"] != srv[j]["cls"])
        diag = float(np.hypot(w, h)) or 1.0
        for i, j, _ in pairs:
            dconf.append(abs(ref[i]["conf"] - srv[j]["conf"]))
            e = np.abs(_xyxy(ref[i]["bbox"]) - _xyxy(srv[j]["bbox"]))
            box_mean.append(float(e.mean()))
            box_max.append(float(e.max()))
            box_rel.append(float(e.mean()) / diag)
        for i in ur:
            if hi is not None and ref[i]["conf"] < hi:
                bnd_ref += 1
            else:
                um_ref += 1
        for j in us:
            if hi is not None and srv[j]["conf"] < hi:
                bnd_srv += 1
            else:
                um_srv += 1
        # unmatched pairs that overlap but carry a DIFFERENT class are worth naming
        for i in ur:
            for j in us:
                if iou_xywh(ref[i]["bbox"], srv[j]["bbox"]) >= iou_thr and not label_match(ref[i]["cls"], srv[j]["cls"]):
                    cls_conf.append((ref[i]["cls"], srv[j]["cls"]))
    eff = max(matched + um_ref, matched + um_srv)
    return {
        "n_ref": n_ref, "n_srv": n_srv, "matched": matched,
        "unmatched_ref": um_ref, "unmatched_srv": um_srv,
        "boundary_ref": bnd_ref, "boundary_srv": bnd_srv,
        "matched_frac": (matched / eff) if eff else None,
        "mean_dconf": float(np.mean(dconf)) if dconf else None,
        "max_dconf": float(np.max(dconf)) if dconf else None,
        "mean_box_px": float(np.mean(box_mean)) if box_mean else None,
        "max_box_px": float(np.max(box_max)) if box_max else None,
        "mean_box_rel": float(np.mean(box_rel)) if box_rel else None,
        "class_confusions": sorted(set(cls_conf))[:10],
        "matched_via_merged_label": merged,
    }


def compare_classification(pairs: List[Tuple[Dict[str, float], List[Tuple[str, float]]]]) -> dict:
    """pairs = [(reference {label: prob} over ALL classes, served [(label, conf), ...] top-k)].
    top1_agree: served top-1 label == reference argmax. |Δprob| compares each served class's conf
    with the reference probability of the SAME class (the server only returns its top-k)."""
    agree, dprob, n = 0, [], 0
    for ref, srv in pairs:
        if not ref or not srv:
            continue
        n += 1
        ref_top = max(ref.items(), key=lambda kv: kv[1])[0]
        srv_sorted = sorted(srv, key=lambda kv: -kv[1])
        agree += int(srv_sorted[0][0] == ref_top)
        for lab, conf in srv_sorted:
            if lab in ref:
                dprob.append(abs(float(conf) - float(ref[lab])))
    return {"n": n, "top1_agree": (agree / n) if n else None,
            "max_dprob": float(np.max(dprob)) if dprob else None,
            "mean_dprob": float(np.mean(dprob)) if dprob else None}


def cosine(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"embedding sizes differ: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def resize_map(m: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize of a 2-D float map (PIL mode F)."""
    m = np.asarray(m, np.float32)
    if m.shape == (height, width):
        return m
    from PIL import Image
    return np.asarray(Image.fromarray(m, mode="F").resize((width, height), Image.BILINEAR), np.float32)


def pearson(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def softmax(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def percentile_ms(samples: Sequence[float]) -> dict:
    s = np.asarray(samples, np.float64)
    if s.size == 0:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    return {"n": int(s.size), "p50_ms": float(np.percentile(s, 50)), "p95_ms": float(np.percentile(s, 95)),
            "mean_ms": float(s.mean())}
