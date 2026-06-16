"""Postprocessing utilities that combine multiple VisionServe results.

Requires numpy (``pip install 'visionserve[images]'``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from .types import Grasp, Result


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics, in pixels.

    ``fx, fy`` are focal lengths and ``cx, cy`` the principal point. Used to
    back-project an image pixel + its depth into a 3D point in the CAMERA frame so
    a TRUE camera→object Euclidean distance can be computed (not just a 2D pixel
    distance). The distance is in the same units as the depth values (e.g. metres
    for a metric RGB-D depth map).
    """

    fx: float
    fy: float
    cx: float
    cy: float


def backproject(u: float, v: float, z: float, K: CameraIntrinsics) -> Tuple[float, float, float]:
    """Back-project pixel ``(u, v)`` at depth ``z`` to ``(X, Y, Z)`` in the camera
    frame: ``X = (u-cx)*z/fx``, ``Y = (v-cy)*z/fy``, ``Z = z``."""
    return ((u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z)


def camera_distance(u: float, v: float, z: float, K: CameraIntrinsics) -> float:
    """Euclidean distance (same units as ``z``) from the camera centre to the 3D
    point that pixel ``(u, v)`` at depth ``z`` back-projects to."""
    x, y, zz = backproject(u, v, z, K)
    return math.sqrt(x * x + y * y + zz * zz)


def get_depth_at_detection(
    depth_result: "Union[Result, Any]",
    det_result: "Result",
    *,
    mode: str = "median",
    depth_scale: Optional[float] = None,
) -> List[Optional[float]]:
    """Return the depth value (in metres) at each detection bbox / mask in *det_result*.

    Extracts the region of the depth map (from *depth_result*) that overlaps
    each detection's bounding box or mask, then aggregates with *mode*.

    Args:
        depth_result: a depth :class:`~visionserve.Result` (e.g. ``midas`` /
                      ``depth-anything-v2``) **or** a 2-D numpy depth array
                      ``(H, W)`` (``uint16`` / ``float32``) at image resolution.
        det_result:   :class:`~visionserve.Result` from a detection or
                      segmentation model.
        mode:         Aggregation over depth pixels in the region —
                      ``"median"`` (default) | ``"mean"`` | ``"min"`` | ``"max"``.
        depth_scale:  metres-per-unit multiplier; ``None`` (default) auto-picks
                      ``0.001`` for integer arrays (mm → m) and ``1.0`` otherwise.

    Returns:
        A list with one entry per detection in *det_result* (same order).
        Each entry is a ``float`` depth value **in metres**, or ``None`` if the
        region had no valid depth pixels.

    Raises:
        ImportError: if numpy is not installed.
        ValueError:  if *depth_result* contains no depth map.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "get_depth_at_detection() requires numpy. "
            "Install with: pip install 'visionserve[images]'"
        ) from e

    depth_arr = _as_depth_meters(depth_result, depth_scale)
    H, W = depth_arr.shape

    agg = {
        "median": np.median,
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
    }.get(mode)
    if agg is None:
        raise ValueError(f"Unknown mode {mode!r}. Use 'median', 'mean', 'min', or 'max'.")

    results: List[Optional[float]] = []

    items = det_result.detections if det_result.detections else det_result.masks
    for item in items:
        x, y, w, h = item.bbox
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(W, int(x + w))
        y2 = min(H, int(y + h))
        if x2 <= x1 or y2 <= y1:
            results.append(None)
            continue
        region = depth_arr[y1:y2, x1:x2]
        valid = region[region > 0]
        if valid.size == 0:
            results.append(None)
        else:
            results.append(float(agg(valid)))

    return results


# ---------------------------------------------------------------------------
# 3D camera-distance helpers (depth + intrinsics)
# ---------------------------------------------------------------------------

def _as_intrinsics(intrinsics: "Union[CameraIntrinsics, Sequence[float]]") -> CameraIntrinsics:
    """Coerce *intrinsics* to a :class:`CameraIntrinsics`.

    Accepts a :class:`CameraIntrinsics` (returned unchanged) or any 4-element
    sequence ``[fx, fy, cx, cy]`` (list / tuple / numpy array).
    """
    if isinstance(intrinsics, CameraIntrinsics):
        return intrinsics
    try:
        vals = [float(v) for v in intrinsics]
    except TypeError as e:
        raise TypeError(
            "intrinsics must be a CameraIntrinsics or a sequence [fx, fy, cx, cy]"
        ) from e
    if len(vals) != 4:
        raise ValueError(
            "intrinsics sequence must have 4 values [fx, fy, cx, cy], got %d" % len(vals)
        )
    return CameraIntrinsics(fx=vals[0], fy=vals[1], cx=vals[2], cy=vals[3])


def _as_depth_meters(depth: "Union[Result, Any]", depth_scale: Optional[float] = None):
    """Return a float ``(H, W)`` depth array **in metres**.

    *depth* may be either:
      * a depth :class:`~visionserve.Result` (uses ``depth_map`` +
        ``depth_width/height``), or
      * a 2-D numpy array (e.g. ``uint16`` / ``float32``) already at the pixel
        resolution of the grasps / detections being measured.

    Values are normalised to metres:
      * ``depth_scale`` (metres = raw × depth_scale), when given, is used verbatim;
      * otherwise INTEGER arrays (e.g. ``uint16`` millimetres from an RGB-D sensor)
        default to ``0.001`` (mm → m), and float arrays / a ``Result`` are assumed
        to already be in metres (scale ``1.0``).
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "depth helpers require numpy. Install with: pip install 'visionserve[images]'"
        ) from e

    if isinstance(depth, np.ndarray):
        arr = depth
    elif hasattr(depth, "depth_map"):  # a depth Result
        if not depth.depth_map:
            raise ValueError("depth_result contains no depth_map")
        arr = np.asarray(depth.depth_map, dtype=float).reshape(
            depth.depth_height, depth.depth_width
        )
    else:
        raise TypeError(
            "depth must be a depth Result or a 2-D numpy array (H, W), got %r" % type(depth)
        )

    if arr.ndim != 2:
        raise ValueError("depth array must be 2-D (H, W), got shape %r" % (arr.shape,))

    if depth_scale is None:
        depth_scale = 0.001 if np.issubdtype(arr.dtype, np.integer) else 1.0
    return arr.astype(float) * float(depth_scale)


def _depth_at_point(depth_arr, x: float, y: float, window: int = 2) -> Optional[float]:
    """Median of the valid (>0) depth pixels in a small window around ``(x, y)``."""
    import numpy as np

    h, w = depth_arr.shape
    xi, yi = int(round(x)), int(round(y))
    x1, x2 = max(0, xi - window), min(w, xi + window + 1)
    y1, y2 = max(0, yi - window), min(h, yi + window + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    region = depth_arr[y1:y2, x1:x2]
    valid = region[region > 0]
    return float(np.median(valid)) if valid.size else None


def object_distances(
    depth_result: "Union[Result, Any]",
    det_result: "Result",
    intrinsics: "Union[CameraIntrinsics, Sequence[float]]",
    *,
    mode: str = "median",
    depth_scale: Optional[float] = None,
) -> List[Optional[float]]:
    """True camera→object Euclidean distance (metres) for each detection/mask.

    Samples the object's depth (over its bbox/mask region, aggregated by *mode*),
    then back-projects the bbox CENTRE pixel with that depth through *intrinsics*.
    One entry per object (same order); ``None`` where depth is unavailable.

    *depth_result* may be a depth :class:`~visionserve.Result` or a 2-D numpy
    array; *intrinsics* a :class:`CameraIntrinsics` or ``[fx, fy, cx, cy]``.
    """
    K = _as_intrinsics(intrinsics)
    depths = get_depth_at_detection(depth_result, det_result, mode=mode, depth_scale=depth_scale)
    items = det_result.detections if det_result.detections else det_result.masks
    out: List[Optional[float]] = []
    for item, z in zip(items, depths):
        if z is None or z <= 0:
            out.append(None)
            continue
        x, y, w, h = item.bbox
        out.append(camera_distance(x + w / 2.0, y + h / 2.0, z, K))
    return out


def grasp_distances(
    depth_result: "Union[Result, Any]",
    grasps: "Sequence[Grasp]",
    intrinsics: "Union[CameraIntrinsics, Sequence[float]]",
    *,
    window: int = 2,
    depth_scale: Optional[float] = None,
) -> List[Optional[float]]:
    """True camera→grasp Euclidean distance (metres) for each grasp (depth sampled
    at the grasp centre). One entry per grasp; ``None`` where depth is unavailable.

    *depth_result* may be a depth :class:`~visionserve.Result` or a 2-D numpy
    array; *intrinsics* a :class:`CameraIntrinsics` or ``[fx, fy, cx, cy]``.
    """
    K = _as_intrinsics(intrinsics)
    depth_arr = _as_depth_meters(depth_result, depth_scale)
    out: List[Optional[float]] = []
    for g in grasps:
        z = _depth_at_point(depth_arr, g.x, g.y, window)
        out.append(None if (z is None or z <= 0) else camera_distance(g.x, g.y, z, K))
    return out


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def _gauss_closeness(value: float, target: float, sigma: float) -> float:
    """A 1.0-at-target, →0-far-away score: ``exp(-0.5 * ((value-target)/sigma)^2)``."""
    if sigma <= 0:
        sigma = 1e-6
    d = (value - target) / sigma
    return math.exp(-0.5 * d * d)


def _resolve_weights(
    weights: Optional[Dict[str, float]],
    *,
    has_near: bool,
    has_distance: bool,
    score_key: str,
) -> Dict[str, float]:
    """Pick the active criteria + weights. An explicit *weights* dict (composite)
    wins; otherwise default to the single most-specific available criterion:
    distance > near > the model's own score (conf/quality)."""
    if weights:
        return {k: float(v) for k, v in weights.items() if v and v > 0}
    if has_distance:
        return {"distance": 1.0}
    if has_near:
        return {"near": 1.0}
    return {score_key: 1.0}

def _near_point(near_point, image_size) -> Optional[Tuple[float, float]]:
    if near_point is None:
        return None
    if isinstance(near_point, str):
        if near_point != "center":
            raise ValueError("near_point string must be 'center'")
        if image_size is None:
            raise ValueError("near_point='center' requires image_size=(W, H)")
        return (image_size[0] / 2.0, image_size[1] / 2.0)
    return (float(near_point[0]), float(near_point[1]))


def select_target_object(
    result: "Result",
    *,
    cls: Optional[Union[str, Sequence[str]]] = None,
    min_conf: float = 0.0,
    near_point: Optional[Union[str, Tuple[float, float]]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    depth_result: "Optional[Union[Result, Any]]" = None,
    intrinsics: "Optional[Union[CameraIntrinsics, Sequence[float]]]" = None,
    target_distance: Optional[float] = None,
    distance_sigma: Optional[float] = None,
    mode: str = "median",
    depth_scale: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    return_index: bool = False,
):
    """Select the best target object from a detection/segmentation result.

    Candidates are first filtered by ``cls`` (a label or list of labels) and
    ``min_conf``. The remaining objects are scored on one or more criteria and the
    highest-scoring object is returned. Criteria (each normalised to ``[0, 1]``):

    * ``"conf"``     — detection confidence.
    * ``"area"``     — bbox area relative to the largest candidate (for "largest").
    * ``"near"``     — 2D proximity of the bbox centre to ``near_point`` (a pixel
      ``(x, y)`` or ``"center"`` for the image centre; nearest → 1).
    * ``"distance"`` — TRUE 3D camera→object distance close to ``target_distance``
      (needs ``depth_result`` + ``intrinsics``). Objects whose camera distance is
      near ``target_distance`` score higher (Gaussian, width ``distance_sigma`` —
      default ``0.15 * target_distance``).

    By default the single most-specific available criterion is used
    (distance > near > conf). Pass ``weights={"conf": .., "near": .., ...}`` to
    combine several into a weighted composite.

    For the ``"distance"`` criterion, *depth_result* may be a depth
    :class:`~visionserve.Result` OR a 2-D numpy depth array ``(H, W)`` (``uint16`` /
    ``float32``) at image resolution, and *intrinsics* may be a
    :class:`CameraIntrinsics` OR ``[fx, fy, cx, cy]``. Depth is normalised to metres
    (``depth_scale``: ``None`` auto-picks ``0.001`` for integer/mm arrays, ``1.0``
    otherwise), so ``target_distance`` is in metres.

    Returns the chosen object (``Detection`` or ``Mask``), or ``None`` if no
    candidate passes the filters. With ``return_index=True`` returns
    ``(object_or_None, index)`` where ``index`` is into the original list.
    """
    items = result.detections if result.detections else result.masks
    cls_set = None
    if cls is not None:
        cls_set = {cls} if isinstance(cls, str) else set(cls)

    cand: List[int] = []
    for i, it in enumerate(items):
        if cls_set is not None and getattr(it, "cls", "") not in cls_set:
            continue
        if getattr(it, "conf", 0.0) < min_conf:
            continue
        cand.append(i)
    if not cand:
        return (None, -1) if return_index else None

    centers = {i: (items[i].bbox[0] + items[i].bbox[2] / 2.0,
                   items[i].bbox[1] + items[i].bbox[3] / 2.0) for i in cand}
    areas = {i: max(0.0, items[i].bbox[2] * items[i].bbox[3]) for i in cand}
    max_area = max(areas.values()) or 1.0

    pt = _near_point(near_point, image_size)
    near_d = {i: math.hypot(centers[i][0] - pt[0], centers[i][1] - pt[1]) for i in cand} if pt else {}
    max_near = max(near_d.values()) if near_d else 0.0

    cam_d: Dict[int, Optional[float]] = {}
    if target_distance is not None:
        if depth_result is None or intrinsics is None:
            raise ValueError("target_distance requires depth_result and intrinsics")
        dl = object_distances(depth_result, result, intrinsics, mode=mode, depth_scale=depth_scale)
        cam_d = {i: (dl[i] if i < len(dl) else None) for i in cand}
        if distance_sigma is None:
            # distance_sigma = max(1e-6, 0.15 * abs(target_distance))
            distance_sigma = max(1e-6, 0.5 * abs(target_distance))

    w = _resolve_weights(weights, has_near=bool(pt), has_distance=target_distance is not None,
                         score_key="conf")

    best_i, best_score = -1, float("-inf")
    for i in cand:
        s, wsum = 0.0, 0.0
        if w.get("conf", 0) > 0:
            s += w["conf"] * float(getattr(items[i], "conf", 0.0)); wsum += w["conf"]
        if w.get("area", 0) > 0:
            s += w["area"] * (areas[i] / max_area); wsum += w["area"]
        if w.get("near", 0) > 0 and pt is not None:
            near_score = 1.0 - (near_d[i] / max_near) if max_near > 0 else 1.0
            s += w["near"] * near_score; wsum += w["near"]
        if w.get("distance", 0) > 0 and target_distance is not None:
            cd = cam_d.get(i)
            s += w["distance"] * (0.0 if cd is None else _gauss_closeness(cd, target_distance, distance_sigma))
            wsum += w["distance"]
        score = s / wsum if wsum > 0 else 0.0
        if score > best_score:
            best_score, best_i = score, i

    chosen = items[best_i] if best_i >= 0 else None
    return (chosen, best_i) if return_index else chosen


def select_target_grasp(
    grasps: "Sequence[Grasp]",
    *,
    cls: Optional[Union[str, Sequence[str]]] = None,
    gripper_min: Optional[float] = None,
    gripper_max: Optional[float] = None,
    target_point: Optional[Tuple[float, float]] = None,
    depth_result: "Optional[Union[Result, Any]]" = None,
    intrinsics: "Optional[Union[CameraIntrinsics, Sequence[float]]]" = None,
    target_distance: Optional[float] = None,
    distance_sigma: Optional[float] = None,
    window: int = 2,
    depth_scale: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    return_index: bool = False,
):
    """Select the best target grasp from a list of grasps.

    Candidates are first filtered by ``cls`` and by gripper-width FEASIBILITY
    (``gripper_min <= width <= gripper_max`` when those bounds are given). The
    survivors are scored on one or more criteria (each normalised to ``[0, 1]``):

    * ``"quality"``  — analytic grasp score.
    * ``"near"``     — 2D proximity of the grasp centre to ``target_point`` (nearest → 1).
    * ``"distance"`` — TRUE 3D camera→grasp distance close to ``target_distance``
      (needs ``depth_result`` + ``intrinsics``; Gaussian, width ``distance_sigma``
      — default ``0.15 * target_distance``).
    * ``"width"``    — preference for a mid-range opening within
      ``[gripper_min, gripper_max]`` (only when both bounds are given).

    By default the single most-specific available criterion is used
    (distance > near > quality). Pass ``weights={...}`` for a weighted composite.

    For the ``"distance"`` criterion, *depth_result* may be a depth
    :class:`~visionserve.Result` OR a 2-D numpy depth array ``(H, W)`` (``uint16`` /
    ``float32``) at the grasps' pixel resolution, and *intrinsics* may be a
    :class:`CameraIntrinsics` OR ``[fx, fy, cx, cy]``. Depth is normalised to metres
    (``depth_scale``: ``None`` auto-picks ``0.001`` for integer/mm arrays, ``1.0``
    otherwise), so ``target_distance`` is in metres.

    Returns the chosen ``Grasp`` (or ``None``); with ``return_index=True`` returns
    ``(grasp_or_None, index)`` into the original list.
    """
    cls_set = None
    if cls is not None:
        cls_set = {cls} if isinstance(cls, str) else set(cls)

    cand: List[int] = []
    for i, g in enumerate(grasps):
        if cls_set is not None and g.cls not in cls_set:
            continue
        if gripper_min is not None and g.width < gripper_min:
            continue
        if gripper_max is not None and g.width > gripper_max:
            continue
        cand.append(i)
    if not cand:
        return (None, -1) if return_index else None

    near_d = {}
    max_near = 0.0
    if target_point is not None:
        near_d = {i: math.hypot(grasps[i].x - target_point[0], grasps[i].y - target_point[1]) for i in cand}
        max_near = max(near_d.values()) if near_d else 0.0

    cam_d: Dict[int, Optional[float]] = {}
    if target_distance is not None:
        if depth_result is None or intrinsics is None:
            raise ValueError("target_distance requires depth_result and intrinsics")
        dl = grasp_distances(depth_result, grasps, intrinsics, window=window, depth_scale=depth_scale)
        cam_d = {i: (dl[i] if i < len(dl) else None) for i in cand}
        if distance_sigma is None:
            distance_sigma = max(1e-6, 0.15 * abs(target_distance))

    width_mid = width_half = None
    if gripper_min is not None and gripper_max is not None and gripper_max > gripper_min:
        width_mid = (gripper_min + gripper_max) / 2.0
        width_half = (gripper_max - gripper_min) / 2.0

    w = _resolve_weights(weights, has_near=bool(near_d), has_distance=target_distance is not None,
                         score_key="quality")

    best_i, best_score = -1, float("-inf")
    for i in cand:
        g = grasps[i]
        s, wsum = 0.0, 0.0
        if w.get("quality", 0) > 0:
            s += w["quality"] * float(g.quality); wsum += w["quality"]
        if w.get("near", 0) > 0 and near_d:
            near_score = 1.0 - (near_d[i] / max_near) if max_near > 0 else 1.0
            s += w["near"] * near_score; wsum += w["near"]
        if w.get("distance", 0) > 0 and target_distance is not None:
            cd = cam_d.get(i)
            s += w["distance"] * (0.0 if cd is None else _gauss_closeness(cd, target_distance, distance_sigma))
            wsum += w["distance"]
        if w.get("width", 0) > 0 and width_mid is not None:
            width_score = max(0.0, 1.0 - abs(g.width - width_mid) / width_half)
            s += w["width"] * width_score; wsum += w["width"]
        score = s / wsum if wsum > 0 else 0.0
        if score > best_score:
            best_score, best_i = score, i

    chosen = grasps[best_i] if best_i >= 0 else None
    return (chosen, best_i) if return_index else chosen


