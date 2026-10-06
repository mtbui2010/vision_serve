"""A small IoU tracker for :meth:`visionserve.Client.watch` (``track=True``). Client side, pure Python.

It is deliberately simple: each new result's detections are matched to the live tracks greedily
by box IoU (highest first, same class only), a match keeps its track's id, an unmatched detection
starts a new track, and a track that went unmatched for more than ``max_age`` results is dropped.
There is no motion model and no appearance model, so it holds identities while objects move less
than about their own size between two processed frames, and swaps or loses them in crowds, under
occlusion, or at a low ``fps``. For those, run a real multi-object tracker on the results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x, y, w, h]`` boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = min(ax2, bx2) - max(a[0], b[0])
    ih = min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = ("id", "bbox", "cls", "missed")

    def __init__(self, tid: int, bbox: Sequence[float], cls: str):
        self.id, self.bbox, self.cls, self.missed = tid, list(bbox), cls, 0


class IoUTracker:
    """Greedy IoU matching with a ``max_age``; sets ``Detection.track_id`` (see the module docstring).

    Args:
        iou_threshold: the least IoU between a detection and a track's last box to continue it.
        max_age: how many results in a row a track may go unmatched before it is dropped.
        match_class: only continue a track with a detection of the same class.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, match_class: bool = True):
        if not 0.0 < float(iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1], got %r" % (iou_threshold,))
        if int(max_age) < 0:
            raise ValueError("max_age must be >= 0, got %r" % (max_age,))
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.match_class = bool(match_class)
        self._tracks: List[_Track] = []
        self._next = 1

    @property
    def active(self) -> Dict[int, List[float]]:
        """``{track_id: last bbox}`` of the live tracks."""
        return {t.id: list(t.bbox) for t in self._tracks}

    def update(self, detections: Sequence[Any]) -> List[Optional[int]]:
        """Assign a ``track_id`` to every detection (in place) and return the ids in order."""
        pairs = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(detections):
                if self.match_class and t.cls != d.cls:
                    continue
                iou = box_iou(t.bbox, d.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        used_t, ids = set(), [None] * len(detections)  # type: ignore[list-item]
        for _, ti, di in pairs:
            if ti in used_t or ids[di] is not None:
                continue
            used_t.add(ti)
            t = self._tracks[ti]
            t.bbox, t.cls, t.missed = list(detections[di].bbox), detections[di].cls, 0
            ids[di] = t.id
        survivors = []
        for ti, t in enumerate(self._tracks):
            if ti not in used_t:
                t.missed += 1
                if t.missed > self.max_age:
                    continue
            survivors.append(t)
        self._tracks = survivors
        for di, d in enumerate(detections):
            if ids[di] is None:
                t = _Track(self._next, d.bbox, d.cls)
                self._next += 1
                self._tracks.append(t)
                ids[di] = t.id
        for d, tid in zip(detections, ids):
            d.track_id = tid
        return ids
