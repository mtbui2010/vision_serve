"""Result dataclasses for the VisionServe client.

These mirror the server's unified wire schema (see ``pkg/api/types.go``). The schema
is the SAME across every task — detection, segmentation, open-vocab, classification,
depth, embedding — so there is a single :class:`Result` type rather than one per model.
"""

from __future__ import annotations

import array
import base64
import dataclasses
import sys
from collections.abc import Sequence as _SequenceABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class FloatArray(_SequenceABC):
    """A read-only, list-like float32 array: how :class:`Result` holds ``depth_map`` and
    ``embeddings`` when the server sent them base64-encoded (``Client(base64_arrays=True)``).

    It supports what reading a ``List[float]`` / ``List[List[float]]`` needs — ``len``,
    indexing, iteration, truthiness, ``==`` with a list — while numpy gets the decoded buffer
    WITHOUT a copy: ``numpy.asarray(x)``, ``x.array``, :meth:`Result.depth_array` and
    :meth:`Result.embeddings_array`. It is NOT a list: ``json.dumps``, ``+`` and ``append`` need
    ``.tolist()`` (or :meth:`Result.to_json`) first, and its values are the exact float32 ones,
    so ``==`` against the same answer sent as JSON numbers can differ in the last digits.
    """

    __slots__ = ("array",)

    def __init__(self, arr: Any):
        self.array = arr  # numpy float32, 1-D (depth_map) or 2-D (embeddings)

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def __getitem__(self, i: Any) -> Any:
        v = self.array[i]
        return FloatArray(v) if getattr(v, "ndim", 0) else float(v)

    def __iter__(self):
        if self.array.ndim == 1:
            return iter(self.array.tolist())
        return (FloatArray(row) for row in self.array)

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        a = self.array
        if dtype is not None and a.dtype != dtype:
            return a.astype(dtype)
        return a.copy() if copy else a

    @property
    def shape(self) -> tuple:
        return tuple(self.array.shape)

    def tolist(self) -> list:
        return self.array.tolist()

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, FloatArray):
            other = other.tolist()
        if isinstance(other, (list, tuple)):
            return self.tolist() == [list(v) if isinstance(v, tuple) else v for v in other]
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # mutable-sequence semantics, like list

    def __repr__(self) -> str:
        return "FloatArray(shape=%r)" % (self.shape,)


def _decode_f32(b64: str, shape: Optional[Sequence[int]] = None) -> Any:
    """Decode the server's base64 little-endian float32 array (``encoding=base64``).

    ``shape=None`` means flat, of whatever length the bytes give. With numpy: a
    :class:`FloatArray` over a writable float32 array (no Python floats are created). Without
    numpy: plain lists, exactly like the JSON-number encoding.
    """
    raw = base64.b64decode(b64)
    if shape is None:
        shape = (len(raw) // 4,)
    n = 1
    for d in shape:
        n *= int(d)
    if len(raw) != 4 * n or not shape:
        raise ValueError("base64 array has %d bytes, shape %s needs %d" % (len(raw), list(shape), 4 * n))
    try:
        import numpy as np
    except ImportError:
        a = array.array("f")
        a.frombytes(raw)
        if sys.byteorder == "big":
            a.byteswap()
        flat = a.tolist()
        if len(shape) == 2:
            d = int(shape[1])
            return [flat[i * d:(i + 1) * d] for i in range(int(shape[0]))]
        return flat
    arr = np.frombuffer(bytearray(raw), dtype="<f4").astype(np.float32, copy=False)
    return FloatArray(arr.reshape(tuple(int(d) for d in shape)))


def _encode_f32(values: Any) -> str:
    """Inverse of :func:`_decode_f32`: base64 of the row-major little-endian float32 bytes."""
    try:
        import numpy as np
    except ImportError:
        flat = [float(v) for row in values for v in (row if isinstance(row, (list, tuple)) else [row])]
        a = array.array("f", flat)
        if sys.byteorder == "big":
            a.byteswap()
        return base64.b64encode(a.tobytes()).decode("ascii")
    return base64.b64encode(np.ascontiguousarray(np.asarray(values, dtype="<f4")).tobytes()).decode("ascii")


def _plain_floats(values: Any) -> list:
    """A list (of lists) of Python floats from a list or a :class:`FloatArray`."""
    if isinstance(values, FloatArray):
        return values.tolist()
    return [_plain_floats(v) if isinstance(v, (list, tuple, FloatArray)) else float(v) for v in values]


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

    def to_json(self) -> Dict[str, Any]:
        """The wire dict (``class``, not ``cls``); inverse of :meth:`from_json`."""
        return {"bbox": list(self.bbox), "class": self.cls, "conf": self.conf}


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

    def to_json(self) -> Dict[str, Any]:
        """The wire dict (``rle`` omitted when empty, as the server does); inverse of
        :meth:`from_json`."""
        out: Dict[str, Any] = {"rle": self.rle} if self.rle else {}
        out.update(bbox=list(self.bbox), conf=self.conf)
        return out

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
        counts = np.array(self.rle.split(), dtype=np.int64) if self.rle.strip() else np.zeros(0, np.int64)
        if counts.size and counts.min() < 0:
            raise ValueError("RLE run counts must be non-negative")
        if int(counts.sum()) != total:
            raise ValueError(
                "RLE run counts sum to %d but width*height = %d" % (int(counts.sum()), total)
            )

        # Runs alternate background/foreground starting with background: run k is foreground
        # when k is odd. np.repeat expands them into the flat COLUMN-MAJOR pixel order, so
        # element index i corresponds to (x = i // height, y = i % height).
        flat = np.repeat((np.arange(counts.size) % 2).astype(bool), counts)

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

    def to_json(self) -> Dict[str, Any]:
        """The wire dict; inverse of :meth:`from_json`."""
        return {"class": self.cls, "conf": self.conf}


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

    def to_json(self) -> Dict[str, Any]:
        """The wire dict (``class`` / ``conf`` omitted for a class-agnostic grasp, as the server
        does); inverse of :meth:`from_json`."""
        out: Dict[str, Any] = {"x": self.x, "y": self.y, "theta": self.theta, "width": self.width,
                               "quality": self.quality}
        if self.cls:
            out["class"] = self.cls
        if self.conf:
            out["conf"] = self.conf
        return out

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
                        ``depth_width * depth_height`` (may be empty). A :class:`FloatArray`
                        (list-like, numpy-backed) when the server sent it base64-encoded; use
                        :meth:`depth_array` for a ``(H, W)`` numpy array either way.
        depth_width:    width of the depth map in pixels.
        depth_height:   height of the depth map in pixels.
        embeddings:     list of embedding vectors (each a ``List[float]``), or a 2-D
                        :class:`FloatArray` when sent base64-encoded; use
                        :meth:`embeddings_array` for an ``(N, D)`` numpy array either way.
        duration_ms:    server-side inference duration in milliseconds.
        device:         execution device the server ran on, e.g. ``"cpu"``,
                        ``"gpu:0"``, or ``"gpu:0+trt"`` (empty if unreported).
        hint:           the server's setup recommendation, if any (e.g. "install the
                        TensorRT EP for faster inference"); empty otherwise.

    The depth map of a ``midas`` / ``depth-anything-v2`` result is RELATIVE inverse depth
    (disparity) min-max normalised to ``[0, 1]`` per image — larger = closer, no units — at
    the MODEL's resolution (``depth_width x depth_height``), not the image's.
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
    hint: str = ""

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Result":
        """Parse the server's JSON (a dict). Both array encodings are accepted: JSON numbers, and
        ``encoding=base64`` (``depth_map_base64`` / ``embeddings_base64`` + ``embeddings_shape``),
        which is decoded here — callers never see the base64."""
        depth_w, depth_h = int(d.get("depth_width", 0) or 0), int(d.get("depth_height", 0) or 0)
        if d.get("depth_map_base64"):
            depth_map: Any = _decode_f32(d["depth_map_base64"], (depth_w * depth_h,) if depth_w and depth_h else None)
        else:
            depth_map = [float(v) for v in (d.get("depth_map") or [])]
        if d.get("embeddings_base64"):
            embeddings: Any = _decode_f32(d["embeddings_base64"], d.get("embeddings_shape") or ())
        else:
            embeddings = [[float(v) for v in row] for row in (d.get("embeddings") or [])]
        return cls(
            task=str(d.get("task", "")),
            model=str(d.get("model", "")),
            detections=[Detection.from_json(x) for x in (d.get("detections") or [])],
            masks=[Mask.from_json(x) for x in (d.get("masks") or [])],
            grasps=[Grasp.from_json(x) for x in (d.get("grasps") or [])],
            classifications=[
                Classification.from_json(x) for x in (d.get("classifications") or [])
            ],
            depth_map=depth_map,
            depth_width=depth_w,
            depth_height=depth_h,
            embeddings=embeddings,
            duration_ms=float(d.get("duration_ms", 0.0)),
            device=str(d.get("device", "")),
            hint=str(d.get("hint", "") or ""),
        )

    def to_json(self, *, encoding: str = "json") -> Dict[str, Any]:
        """The server's JSON wire shape for this result (a dict for ``json.dumps``); the inverse
        of :meth:`from_json`: ``Result.from_json(r.to_json()) == r``.

        Field names and order follow ``pkg/api/types.go`` (``class``, not ``cls``) and empty
        fields are omitted like the Go ``omitempty`` tags, so the dict matches what the server
        sends. ``encoding="base64"`` writes ``depth_map`` / ``embeddings`` the way the server
        does for ``encoding=base64`` (exact float32 bytes, compact); the default writes number
        arrays.
        """
        if encoding not in ("json", "base64"):
            raise ValueError("encoding must be 'json' or 'base64', got %r" % (encoding,))
        out: Dict[str, Any] = {"task": self.task, "model": self.model}
        if self.device:
            out["device"] = self.device
        if self.hint:
            out["hint"] = self.hint
        if self.detections:
            out["detections"] = [d.to_json() for d in self.detections]
        if self.masks:
            out["masks"] = [m.to_json() for m in self.masks]
        if self.grasps:
            out["grasps"] = [g.to_json() for g in self.grasps]
        if self.classifications:
            out["classifications"] = [c.to_json() for c in self.classifications]
        b64 = encoding == "base64"
        rows = len(self.embeddings)
        dims = {len(r) for r in self.embeddings}
        emb_b64 = b64 and len(dims) == 1  # ragged rows have no [N, D] shape: they stay numbers
        if rows and not emb_b64:
            out["embeddings"] = _plain_floats(self.embeddings)
        if len(self.depth_map) and not b64:
            out["depth_map"] = _plain_floats(self.depth_map)
        if self.depth_width:
            out["depth_width"] = self.depth_width
        if self.depth_height:
            out["depth_height"] = self.depth_height
        out["duration_ms"] = self.duration_ms
        if b64 and len(self.depth_map):
            out["depth_map_base64"] = _encode_f32(self.depth_map)
        if emb_b64:
            out["embeddings_base64"] = _encode_f32(self.embeddings)
            out["embeddings_shape"] = [rows, dims.pop()]
        return out

    def depth_array(self):
        """The depth map as a float32 numpy array of shape ``(depth_height, depth_width)``
        (relative inverse depth in ``[0, 1]``, larger = closer — see the class docstring), or
        ``None`` when this result has no depth map. No copy when the server sent base64."""
        import numpy as np

        if not self.depth_map or self.depth_width <= 0 or self.depth_height <= 0:
            return None
        return np.asarray(self.depth_map, dtype=np.float32).reshape(self.depth_height, self.depth_width)

    def embeddings_array(self):
        """The embeddings as a float32 numpy array of shape ``(N, D)``, or ``None`` when this
        result has none. No copy when the server sent base64."""
        import numpy as np

        if not self.embeddings:
            return None
        return np.asarray(self.embeddings, dtype=np.float32)

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
        # Prefer detection bboxes (class-aware); else mask bboxes (class-agnostic automask).
        objects = [d.bbox for d in self.detections] or [m.bbox for m in self.masks]
        return dataclasses.replace(self, grasps=_top_grasps_per_object(self.grasps, objects, max_per_object))

    def group_by_class(self) -> "Dict[str, 'Result']":
        """Return a ``dict[class_label → Result]`` grouping detections and masks by class.

        Masks carry no class in the wire schema. A mask whose bbox equals a detection's bbox
        (Grounded-SAM / grasp pipelines copy the detection box onto its mask) takes that
        detection's class; any other mask (e.g. a box-prompted SAM mask) is grouped under ``""``.
        """
        groups: Dict[str, Dict[str, list]] = {}
        box_cls: Dict[tuple, str] = {}
        for det in self.detections:
            groups.setdefault(det.cls, {"detections": [], "masks": []})["detections"].append(det)
            box_cls.setdefault(tuple(float(v) for v in det.bbox), det.cls)
        for mask in self.masks:
            label = box_cls.get(tuple(float(v) for v in mask.bbox), "")
            groups.setdefault(label, {"detections": [], "masks": []})["masks"].append(mask)

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


def _grasp_object_key(g: Any, objects: Sequence[Sequence[float]]) -> Optional[int]:
    """Index of the SMALLEST ``[x, y, w, h]`` bbox whose interior contains the grasp centre, or
    ``None`` when no bbox contains it."""
    best: Optional[int] = None
    best_area: Optional[float] = None
    for i, (x, y, w, h) in enumerate(objects):
        if x <= g.x <= x + w and y <= g.y <= y + h:
            area = w * h
            if best_area is None or area < best_area:
                best_area, best = area, i
    return best


def _top_grasps_per_object(grasps: Sequence[Any], objects: Sequence[Sequence[float]], k: int) -> List[Any]:
    """The ``k`` highest-quality grasps per object — the one grouping rule behind
    :meth:`Result.filter_grasps` and the visualizer.

    A grasp belongs to the smallest object bbox containing its centre; a grasp no bbox contains
    (or every grasp, when there are no objects) is bucketed by its class label, so each kind is
    still sampled.
    """
    groups: Dict[Any, List[Any]] = {}
    for g in grasps:
        key = _grasp_object_key(g, objects) if objects else None
        groups.setdefault(("cls", g.cls) if key is None else key, []).append(g)
    kept: List[Any] = []
    for gs in groups.values():
        gs.sort(key=lambda g: g.quality, reverse=True)
        kept.extend(gs[:k])
    return kept
