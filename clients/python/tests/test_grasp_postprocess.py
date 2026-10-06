"""Tests for grasp types, 3D camera-distance, and target selection.

These are pure (no server): they build synthetic Result objects in memory.
"""

import math

import pytest

from visionserve import (
    CameraIntrinsics,
    Grasp,
    Result,
    backproject,
    camera_distance,
    select_target_grasp,
    select_target_object,
)
from visionserve.types import Detection


# --------------------------------------------------------------------------
# types
# --------------------------------------------------------------------------

def test_grasp_from_json_and_contacts():
    g = Grasp.from_json(
        {"x": 50, "y": 30, "theta": 0.0, "width": 20, "quality": 0.9, "class": "cup", "conf": 0.8}
    )
    assert (g.x, g.y, g.width, g.quality, g.cls, g.conf) == (50, 30, 20, 0.9, "cup", 0.8)
    c0, c1 = g.contacts()
    # theta=0 → contacts along x at center ± width/2.
    assert c0 == pytest.approx([40, 30])
    assert c1 == pytest.approx([60, 30])


def test_result_parses_grasps():
    res = Result.from_json(
        {
            "task": "grasp",
            "model": "grasp",
            "grasps": [
                {"x": 1, "y": 2, "theta": 0.1, "width": 12, "quality": 0.7},
                {"x": 3, "y": 4, "theta": 0.2, "width": 30, "quality": 0.4, "class": "box", "conf": 0.6},
            ],
        }
    )
    assert res.task == "grasp"
    assert len(res.grasps) == 2
    assert res.grasps[1].cls == "box"


# --------------------------------------------------------------------------
# 3D camera distance
# --------------------------------------------------------------------------

def test_backproject_and_distance():
    K = CameraIntrinsics(fx=100, fy=100, cx=50, cy=50)
    # principal-point pixel → only Z component.
    assert backproject(50, 50, 2.0, K) == pytest.approx((0.0, 0.0, 2.0))
    assert camera_distance(50, 50, 2.0, K) == pytest.approx(2.0)
    # off-axis pixel adds X.
    assert camera_distance(150, 50, 2.0, K) == pytest.approx(math.sqrt(8.0))


def _depth_result(width, height, fill):
    """Synthetic METRIC depth (metres, float32 array as from an RGB-D sensor); fill is a
    callable(x,y)->depth. A server depth Result is relative inverse depth and is refused by the
    metric helpers (see tests/test_sdk_review_fixes.py)."""
    import numpy as np
    return np.array([[float(fill(x, y)) for x in range(width)] for y in range(height)], np.float32)


def test_object_distances():
    from visionserve import object_distances

    pytest.importorskip("numpy")
    K = CameraIntrinsics(fx=100, fy=100, cx=50, cy=50)
    depth = _depth_result(100, 100, lambda x, y: 2.0)
    det = Result(task="detection", model="rf-detr",
                 detections=[Detection(bbox=[40, 40, 20, 20], cls="cup", conf=0.9)])
    d = object_distances(depth, det, K)
    # center pixel (50,50) at depth 2 → distance 2.
    assert d[0] == pytest.approx(2.0, abs=1e-6)


# --------------------------------------------------------------------------
# target selection
# --------------------------------------------------------------------------

def test_select_target_object_by_class_and_conf():
    res = Result(task="detection", model="m", detections=[
        Detection(bbox=[0, 0, 10, 10], cls="cup", conf=0.6),
        Detection(bbox=[20, 0, 10, 10], cls="bottle", conf=0.9),
        Detection(bbox=[40, 0, 10, 10], cls="cup", conf=0.8),
    ])
    # restrict to cup → highest-conf cup is the 0.8 one.
    chosen = select_target_object(res, cls="cup")
    assert chosen.conf == 0.8
    # min_conf filters everything below.
    assert select_target_object(res, cls="cup", min_conf=0.85) is None


def test_select_target_object_near_point():
    res = Result(task="detection", model="m", detections=[
        Detection(bbox=[0, 0, 10, 10], cls="a", conf=0.5),     # center (5,5)
        Detection(bbox=[90, 90, 10, 10], cls="b", conf=0.5),   # center (95,95)
    ])
    chosen, idx = select_target_object(res, near_point=(0, 0), return_index=True)
    assert idx == 0
    chosen2, idx2 = select_target_object(res, near_point="center", image_size=(100, 100), return_index=True)
    # both equidistant from center → first wins (stable); just assert a valid pick.
    assert idx2 in (0, 1)


def test_select_target_object_by_distance():
    pytest.importorskip("numpy")
    K = CameraIntrinsics(fx=100, fy=100, cx=50, cy=50)
    # left half depth 2, right half depth 5.
    depth = _depth_result(100, 100, lambda x, y: 2.0 if x < 50 else 5.0)
    res = Result(task="detection", model="m", detections=[
        Detection(bbox=[10, 45, 10, 10], cls="near", conf=0.5),   # left → ~2
        Detection(bbox=[70, 45, 10, 10], cls="far", conf=0.5),    # right → ~5
    ])
    chosen = select_target_object(res, depth_result=depth, intrinsics=K, target_distance=2.0)
    assert chosen.cls == "near"
    chosen_far = select_target_object(res, depth_result=depth, intrinsics=K, target_distance=5.0)
    assert chosen_far.cls == "far"


def test_select_target_grasp_quality_and_width():
    grasps = [
        Grasp(x=10, y=10, theta=0, width=8, quality=0.9),    # too narrow
        Grasp(x=20, y=20, theta=0, width=40, quality=0.5),
        Grasp(x=30, y=30, theta=0, width=60, quality=0.7),
    ]
    # default → highest quality (the narrow one).
    assert select_target_grasp(grasps).quality == 0.9
    # width feasibility filter excludes the narrow one → best remaining quality 0.7.
    g = select_target_grasp(grasps, gripper_min=20, gripper_max=80)
    assert g.quality == 0.7


def test_select_target_grasp_target_point_and_composite():
    grasps = [
        Grasp(x=0, y=0, theta=0, width=30, quality=0.4),
        Grasp(x=100, y=100, theta=0, width=30, quality=0.9),
    ]
    # nearest to (0,0) is grasp 0 despite lower quality.
    near = select_target_grasp(grasps, target_point=(0, 0))
    assert near.x == 0
    # composite: quality dominates → grasp 1.
    comp = select_target_grasp(grasps, target_point=(0, 0), weights={"quality": 1.0, "near": 0.1})
    assert comp.x == 100


# --------------------------------------------------------------------------
# visualization: per-object grasp sampling (no Pillow needed for the grouper)
# --------------------------------------------------------------------------

def test_grasps_per_object_top_k():
    from visionserve.visualize import _grasps_per_object

    dets = [
        Detection(bbox=[0, 0, 100, 100], cls="cup", conf=0.9),
        Detection(bbox=[200, 0, 100, 100], cls="box", conf=0.8),
    ]
    grasps = (
        [Grasp(x=50, y=50, theta=0, width=20, quality=q, cls="cup") for q in (0.1, 0.5, 0.9, 0.7, 0.3)]
        + [Grasp(x=250, y=50, theta=0, width=20, quality=q, cls="box") for q in (0.2, 0.8, 0.6, 0.4)]
    )
    res = Result(task="grasp", model="grasp-rfdetr", detections=dets, grasps=grasps)

    sel = _grasps_per_object(res, 3)
    assert sorted((g.quality for g in sel if g.cls == "cup"), reverse=True) == [0.9, 0.7, 0.5]
    assert sorted((g.quality for g in sel if g.cls == "box"), reverse=True) == [0.8, 0.6, 0.4]
    # None / <=0 keeps everything.
    assert len(_grasps_per_object(res, None)) == len(grasps)
    assert len(_grasps_per_object(res, 0)) == len(grasps)


def _food_result():
    """The shape of grasp-rfdetr on COCO 389381 (food.jpg): the carrot's box lies inside the
    bowl's, and the bowl's best grasps sit inside the carrot's box. Boxes, confs and the grasp
    centres are the server's (rounded)."""
    dets = [
        Detection(bbox=[373, 278, 105, 129], cls="broccoli", conf=0.8773410233789939),
        Detection(bbox=[136, 140, 118, 106], cls="carrot", conf=0.7589307443456806),
        Detection(bbox=[58, 49, 411, 272], cls="bowl", conf=0.7163178417646674),
    ]
    g = []
    for x, y, q in ((424.0, 332.5, 0.966), (424.0, 331.0, 0.9655), (423.5, 330.5, 0.965)):
        g.append(Grasp(x=x, y=y, theta=3.0, width=75, quality=q, cls="broccoli", conf=dets[0].conf))
    for x, y, q in ((197.0, 191.5, 0.965), (198.0, 193.0, 0.9645), (197.5, 191.0, 0.964)):
        g.append(Grasp(x=x, y=y, theta=0.5, width=40, quality=q, cls="carrot", conf=dets[1].conf))
    # the bowl's grasps: the best three are INSIDE the carrot's box
    for x, y, q in ((274.0, 228.5, 0.993), (251.5, 220.5, 0.99), (251.0, 221.0, 0.9899), (251.0, 220.5, 0.9898)):
        g.append(Grasp(x=x, y=y, theta=0.8, width=40, quality=q, cls="bowl", conf=dets[2].conf))
    return Result(task="grasp", model="grasp-rfdetr", detections=dets, grasps=g)


def test_grasps_are_grouped_by_their_source_detection_not_by_position():
    res = _food_result()
    one = res.filter_grasps(1).grasps
    assert [g.cls for g in one] == ["broccoli", "carrot", "bowl"]  # before: broccoli, bowl, bowl
    three = res.filter_grasps(3).grasps
    assert [g.cls for g in three].count("carrot") == 3  # before: 0 (bowl grasps took its slots)
    assert [g.cls for g in three].count("bowl") == 3
    assert [g.quality for g in three if g.cls == "bowl"] == [0.993, 0.99, 0.9899]


def test_grasp_source_match_tolerates_float32_conf_and_tie_breaks_by_box():
    import struct

    f32 = struct.unpack("f", struct.pack("f", 0.7589307443456806))[0]  # a float32 round trip
    dets = [Detection(bbox=[0, 0, 10, 10], cls="cup", conf=0.75),
            Detection(bbox=[100, 0, 10, 10], cls="cup", conf=0.75),  # same class AND conf
            Detection(bbox=[0, 0, 200, 200], cls="tray", conf=0.7589307443456806)]
    grasps = [Grasp(x=105, y=5, theta=0, width=5, quality=0.9, cls="cup", conf=0.75),
              Grasp(x=104, y=5, theta=0, width=5, quality=0.8, cls="cup", conf=0.75),
              Grasp(x=5, y=5, theta=0, width=5, quality=0.7, cls="cup", conf=0.75),
              Grasp(x=5, y=5, theta=0, width=5, quality=0.6, cls="tray", conf=f32)]
    kept = Result(task="grasp", model="g", detections=dets, grasps=grasps).filter_grasps(1).grasps
    # the two cups are told apart by the box containing the centre; the tray grasp (inside a cup
    # box, conf off by float32 rounding) still joins the tray
    assert [(g.cls, g.quality) for g in kept] == [("cup", 0.9), ("cup", 0.7), ("tray", 0.6)]


def test_class_aware_grasp_without_its_detection_and_class_agnostic_fallback():
    dets = [Detection(bbox=[0, 0, 100, 100], cls="box", conf=0.9)]
    # "cup" was filtered out of detections (filter_by_conf): its grasps stay one group, not the box's
    grasps = [Grasp(x=10, y=10, theta=0, width=5, quality=q, cls="cup", conf=0.3) for q in (0.5, 0.9)]
    grasps += [Grasp(x=50, y=50, theta=0, width=5, quality=q, cls="box", conf=0.9) for q in (0.4, 0.8)]
    kept = Result(task="grasp", model="g", detections=dets, grasps=grasps).filter_grasps(1).grasps
    assert [(g.cls, g.quality) for g in kept] == [("cup", 0.9), ("box", 0.8)]
    # class-agnostic grasps on automatic masks: grouped by the smallest containing mask box
    from visionserve.types import Mask

    masks = [Mask(rle="", bbox=[0, 0, 100, 100], conf=0.9), Mask(rle="", bbox=[0, 0, 20, 20], conf=0.9)]
    agn = [Grasp(x=10, y=10, theta=0, width=5, quality=q) for q in (0.1, 0.2)]
    agn += [Grasp(x=60, y=60, theta=0, width=5, quality=q) for q in (0.3, 0.4)]
    kept = Result(task="grasp", model="g", masks=masks, grasps=agn).filter_grasps(1).grasps
    assert [(g.x, g.quality) for g in kept] == [(10, 0.2), (60, 0.4)]


def test_result_parses_device():
    res = Result.from_json({"task": "grasp", "model": "grasp", "device": "gpu:0+trt"})
    assert res.device == "gpu:0+trt"
