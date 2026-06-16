"""Result dataclasses for the VisionServe client.

These mirror the server's unified wire schema (see ``pkg/api/types.go``). The schema
is the SAME across every task — detection, segmentation, open-vocab, classification,
depth, embedding — so there is a single :class:`Result` type rather than one per model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Detection:
    """A single detected object.

    Attributes:
        bbox: ``[x, y, w, h]`` (top-left corner + width/height) in ORIGINAL image pixels.
        cls:  class label string.
        conf: confidence in ``[0, 1]``.
    """

    bbox: List[float]
    cls: str
    conf: float

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Detection":
        return cls(
            bbox=[float(v) for v in d.get("bbox", [0, 0, 0, 0])],
            cls=str(d.get("class", "")),
            conf=float(d.get("conf", 0.0)),
        )


@dataclass
class Mask:
    """A segmentation mask.

    Attributes:
        rle:  COCO-style COLUMN-MAJOR uncompressed RLE — space-separated integer run
              counts, starting with a background (0) run, read column-major (column
              outer, row inner) over the ORIGINAL image ``H x W``.
        bbox: ``[x, y, w, h]`` bounding box of the mask in ORIGINAL image pixels.
        conf: confidence (e.g. predicted IoU) in ``[0, 1]``.
    """

    rle: str
    bbox: List[float]
    conf: float

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Mask":
        bbox = d.get("bbox") or [0, 0, 0, 0]
        return cls(
            rle=str(d.get("rle", "")),
            bbox=[float(v) for v in bbox],
            conf=float(d.get("conf", 0.0)),
        )

    def to_ndarray(self, width: int, height: int):
        """Decode the column-major RLE into a boolean ``(height, width)`` numpy array.

        This is the exact inverse of the Go encoder ``encodeRLEColumnMajor``: runs
        alternate starting from background (False), and are laid out in column-major
        (Fortran) order — column is the outer loop, row the inner loop. So the i-th
        pixel in run order maps to ``x = i // height``, ``y = i % height``.

        Args:
            width:  ORIGINAL image width (W) the mask was encoded against.
            height: ORIGINAL image height (H) the mask was encoded against.

        Returns:
            ``numpy.ndarray`` of dtype ``bool`` and shape ``(height, width)``.

        Raises:
            ImportError: if numpy is not installed.
            ValueError: if the run counts do not sum to ``width * height``.
        """
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover - exercised only without numpy
            raise ImportError(
                "Mask.to_ndarray() requires numpy. Install with: "
                "pip install 'visionserve[images]'"
            ) from e

        total = int(width) * int(height)
        counts = [int(c) for c in self.rle.split()] if self.rle.strip() else []
        if sum(counts) != total:
            raise ValueError(
                "RLE run counts sum to %d but width*height = %d" % (sum(counts), total)
            )

        # Build a flat array in COLUMN-MAJOR order, then reshape with Fortran order so
        # element index i corresponds to (x = i // height, y = i % height).
        flat = np.zeros(total, dtype=bool)
        idx = 0
        value = False  # runs start with background
        for c in counts:
            if value and c > 0:
                flat[idx : idx + c] = True
            idx += c
            value = not value

        # flat is column-major over (height, width): reshape with order="F".
        return flat.reshape((height, width), order="F")


@dataclass
class Classification:
    """A single classification prediction.

    Attributes:
        cls:  class label string.
        conf: confidence in ``[0, 1]``.
    """

    cls: str
    conf: float

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Classification":
        return cls(
            cls=str(d.get("class", "")),
            conf=float(d.get("conf", 0.0)),
        )


@dataclass
class Grasp:
    """A planar parallel-jaw grasp in ORIGINAL image coordinates.

    Attributes:
        x, y:    grasp center in ORIGINAL image pixels.
        theta:   in-plane gripper-closing angle in radians (the jaws close along
                 the direction ``(cos theta, sin theta)``).
        width:   jaw opening in ORIGINAL image pixels.
        quality: analytic grasp score in ``[0, 1]``.
        cls:     source object label (box mode); ``""`` for class-agnostic grasps.
        conf:    source detector confidence (box mode); ``0.0`` if class-agnostic.
    """

    x: float
    y: float
    theta: float
    width: float
    quality: float
    cls: str = ""
    conf: float = 0.0

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Grasp":
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            theta=float(d.get("theta", 0.0)),
            width=float(d.get("width", 0.0)),
            quality=float(d.get("quality", 0.0)),
            cls=str(d.get("class", "")),
            conf=float(d.get("conf", 0.0)),
        )

    @property
    def pose(self) -> List[float]:
        """Grasp pose as ``[x, y, width, theta]`` for robot control."""
        return [self.x, self.y, self.width, self.theta]

    def contacts(self) -> List[List[float]]:
        """Return the two jaw-contact points ``[[x0,y0],[x1,y1]]`` in image pixels.

        The contacts sit at ``center ± (width/2) * (cos theta, sin theta)``.
        """
        import math

        dx = math.cos(self.theta) * self.width / 2.0
        dy = math.sin(self.theta) * self.width / 2.0
        return [[self.x - dx, self.y - dy], [self.x + dx, self.y + dy]]

    def contacts_flat(self) -> List[float]:
        """Return jaw-contact points as flat ``[x0, y0, x1, y1]`` in image pixels."""
        import math

        dx = math.cos(self.theta) * self.width / 2.0
        dy = math.sin(self.theta) * self.width / 2.0
        return [self.x - dx, self.y - dy, self.x + dx, self.y + dy]


@dataclass
class Result:
    """Unified prediction result returned by ``POST /api/predict``.

    Attributes:
        task:           one of ``detection`` | ``segmentation`` | ``open_vocab`` |
                        ``classification`` | ``depth`` | ``embedding`` | ``grasp``.
        model:          model name that produced the result.
        detections:     list of :class:`Detection` (may be empty).
        masks:          list of :class:`Mask` (may be empty).
        grasps:         list of :class:`Grasp` (may be empty; from a ``grasp`` model).
        classifications: list of :class:`Classification` (may be empty).
        depth_map:      flat list of float depth values, row-major, size
                        ``depth_width * depth_height`` (may be empty).
        depth_width:    width of the depth map in pixels.
        depth_height:   height of the depth map in pixels.
        embeddings:     list of embedding vectors (each a ``List[float]``).
        duration_ms:    server-side inference duration in milliseconds.
        device:         execution device the server ran on, e.g. ``"cpu"``,
                        ``"gpu:0"``, or ``"gpu:0+trt"`` (empty if unreported).
    """

    task: str
    model: str
    detections: List[Detection] = field(default_factory=list)
    masks: List[Mask] = field(default_factory=list)
    grasps: List[Grasp] = field(default_factory=list)
    classifications: List[Classification] = field(default_factory=list)
    depth_map: List[float] = field(default_factory=list)
    depth_width: int = 0
    depth_height: int = 0
    embeddings: List[List[float]] = field(default_factory=list)
    duration_ms: float = 0.0
    device: str = ""

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Result":
        return cls(
            task=str(d.get("task", "")),
            model=str(d.get("model", "")),
            detections=[Detection.from_json(x) for x in (d.get("detections") or [])],
            masks=[Mask.from_json(x) for x in (d.get("masks") or [])],
            grasps=[Grasp.from_json(x) for x in (d.get("grasps") or [])],
            classifications=[
                Classification.from_json(x) for x in (d.get("classifications") or [])
            ],
            depth_map=[float(v) for v in (d.get("depth_map") or [])],
            depth_width=int(d.get("depth_width", 0)),
            depth_height=int(d.get("depth_height", 0)),
            embeddings=[
                [float(v) for v in row] for row in (d.get("embeddings") or [])
            ],
            duration_ms=float(d.get("duration_ms", 0.0)),
            device=str(d.get("device", "")),
        )

    def filter_by_size(
        self,
        *,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> "Result":
        """Return a new Result keeping only objects whose bbox area is in [min_size, max_size].

        If ``image_width`` and ``image_height`` are both given, ``min_size`` /
        ``max_size`` are treated as fractions of the image area (0.0–1.0).
        Otherwise they are absolute pixel² areas.

        Args:
            min_size:     minimum bbox area (inclusive); ``None`` means no lower bound.
            max_size:     maximum bbox area (inclusive); ``None`` means no upper bound.
            image_width:  original image width in pixels (used with ``image_height`` to
                          interpret min/max_size as relative fractions).
            image_height: original image height in pixels.

        Returns:
            A new :class:`Result` instance with filtered :attr:`detections` and
            :attr:`masks`. All other fields are copied unchanged.
        """
        # Resolve absolute thresholds.
        threshold_min: Optional[float] = None
        threshold_max: Optional[float] = None
        if image_width is not None and image_height is not None:
            image_area = float(image_width * image_height)
            if min_size is not None:
                threshold_min = min_size * image_area
            if max_size is not None:
                threshold_max = max_size * image_area
        else:
            threshold_min = float(min_size) if min_size is not None else None
            threshold_max = float(max_size) if max_size is not None else None

        def _keep(bbox: List[float]) -> bool:
            area = bbox[2] * bbox[3]
            if threshold_min is not None and area < threshold_min:
                return False
            if threshold_max is not None and area > threshold_max:
                return False
            return True

        return dataclasses.replace(
            self,
            detections=[d for d in self.detections if _keep(d.bbox)],
            masks=[m for m in self.masks if _keep(m.bbox)],
        )

    def filter_by_conf(
        self,
        min_conf: float = 0.0,
        max_conf: float = 1.0,
    ) -> "Result":
        """Keep only predictions whose ``conf`` is in ``[min_conf, max_conf]``."""
        def _keep(conf: float) -> bool:
            return min_conf <= conf <= max_conf

        return dataclasses.replace(
            self,
            detections=[d for d in self.detections if _keep(d.conf)],
            masks=[m for m in self.masks if _keep(m.conf)],
            classifications=[c for c in self.classifications if _keep(c.conf)],
        )

    def sort_by_conf(self, *, descending: bool = True) -> "Result":
        """Return a new Result with predictions sorted by ``conf``."""
        return dataclasses.replace(
            self,
            detections=sorted(self.detections, key=lambda d: d.conf, reverse=descending),
            masks=sorted(self.masks, key=lambda m: m.conf, reverse=descending),
            classifications=sorted(self.classifications, key=lambda c: c.conf, reverse=descending),
        )

    def top_k(self, k: int) -> "Result":
        """Keep the top-k predictions by confidence."""
        sorted_result = self.sort_by_conf(descending=True)
        return dataclasses.replace(
            sorted_result,
            detections=sorted_result.detections[:k],
            masks=sorted_result.masks[:k],
            classifications=sorted_result.classifications[:k],
        )

    def nms(self, iou_threshold: float = 0.5) -> "Result":
        """Non-Maximum Suppression on ``detections`` only."""
        def _iou(a: List[float], b: List[float]) -> float:
            ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
            bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = inter_w * inter_h
            if inter == 0.0:
                return 0.0
            area_a = a[2] * a[3]
            area_b = b[2] * b[3]
            return inter / (area_a + area_b - inter)

        sorted_dets = sorted(self.detections, key=lambda d: d.conf, reverse=True)
        kept: List[Detection] = []
        for det in sorted_dets:
            if all(_iou(det.bbox, k.bbox) < iou_threshold for k in kept):
                kept.append(det)

        return dataclasses.replace(self, detections=kept)

    def filter_grasps(self, max_per_object: Optional[int] = None) -> "Result":
        """Keep the top-``max_per_object`` highest-quality grasps per detected object.

        Grasps are grouped by the smallest detection or mask bbox whose interior
        contains each grasp centre. When no bbox contains a grasp it is bucketed by
        class label. ``None`` or ``<= 0`` keeps all grasps unchanged.
        """
        if max_per_object is None or max_per_object <= 0 or not self.grasps:
            return self

        objects = [d.bbox for d in self.detections] or [m.bbox for m in self.masks]

        def _obj_key(g: Any) -> Any:
            best: Optional[int] = None
            best_area: Optional[float] = None
            for i, bbox in enumerate(objects):
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                if x <= g.x <= x + w and y <= g.y <= y + h:
                    area = w * h
                    if best_area is None or area < best_area:
                        best_area = area
                        best = i
            return best if best is not None else ("cls", g.cls)

        groups: Dict[Any, List] = {}
        for g in self.grasps:
            key = _obj_key(g) if objects else ("cls", g.cls)
            groups.setdefault(key, []).append(g)

        kept: List = []
        for gs in groups.values():
            gs.sort(key=lambda g: g.quality, reverse=True)
            kept.extend(gs[:max_per_object])

        return dataclasses.replace(self, grasps=kept)

    def group_by_class(self) -> "Dict[str, 'Result']":
        """Return a ``dict[class_label → Result]`` grouping detections and masks by class."""
        groups: Dict[str, Dict[str, list]] = {}
        for det in self.detections:
            groups.setdefault(det.cls, {"detections": [], "masks": []})["detections"].append(det)
        for mask in self.masks:
            groups.setdefault(mask.cls, {"detections": [], "masks": []})["masks"].append(mask)

        result: Dict[str, Result] = {}
        for label, items in groups.items():
            result[label] = dataclasses.replace(
                self,
                detections=items["detections"],
                masks=items["masks"],
                classifications=[],
                depth_map=[],
                depth_width=0,
                depth_height=0,
                embeddings=[],
            )
        return result

    def visualize(self, image: Any, **kwargs: Any) -> "Any":
        """Draw predictions on *image* and return a ``PIL.Image.Image``.

        Convenience wrapper around :func:`visionserve.visualize.draw`.

        Args:
            image: ``PIL.Image``, file path (str), or raw image bytes.
            **kwargs: forwarded to :func:`~visionserve.visualize.draw` — e.g.
                      ``alpha=0.6``, ``target_grasp=<Grasp>`` to highlight a grasp,
                      or ``target_box=<Detection|Mask|[x,y,w,h]>`` to highlight a
                      selected target box in red.

        Returns:
            Annotated ``PIL.Image.Image``.
        """
        from .visualize import draw  # lazy import keeps pillow optional

        return draw(self, image, **kwargs)


@dataclass
class ModelInfo:
    """An entry from ``GET /api/models``."""

    name: str
    task: str
    license: str
    state: str  # "not_downloaded" | "available" | "loaded"

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ModelInfo":
        return cls(
            name=str(d.get("name", "")),
            task=str(d.get("task", "")),
            license=str(d.get("license", "")),
            state=str(d.get("state", "")),
        )


def _is_loaded(info: ModelInfo) -> bool:
    return info.state == "loaded"
