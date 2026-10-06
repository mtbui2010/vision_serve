"""group_by_class splits the grasps per class like detections and masks (the JS SDK's groupByClass
does the same; both also run the shared cases in clients/testdata/postprocess_sync.json)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visionserve.types import Result  # noqa: E402


def test_group_by_class_carries_only_that_class_grasps():
    r = Result.from_json({
        "task": "grasp", "model": "grasp-rfdetr",
        "detections": [{"bbox": [0, 0, 10, 10], "class": "cup", "conf": 0.9},
                       {"bbox": [20, 0, 10, 10], "class": "bowl", "conf": 0.8}],
        "masks": [{"bbox": [20, 0, 10, 10], "conf": 0.95}, {"bbox": [1, 2, 3, 4], "conf": 0.5}],
        "grasps": [{"x": 25, "y": 5, "theta": 0, "width": 4, "quality": 0.8, "class": "bowl", "conf": 0.8},
                   {"x": 5, "y": 5, "theta": 0, "width": 4, "quality": 0.9, "class": "cup", "conf": 0.9},
                   {"x": 2, "y": 3, "theta": 0, "width": 4, "quality": 0.6},    # class-agnostic, in the cup box
                   {"x": 99, "y": 99, "theta": 0, "width": 4, "quality": 0.5}],  # in no box: ""
        "classifications": [{"class": "cup", "conf": 0.5}],
        "depth_map": [0, 1], "depth_width": 2, "depth_height": 1,
        "duration_ms": 9,
    })
    g = r.group_by_class()
    assert list(g) == ["cup", "bowl", ""]
    assert [x.quality for x in g["cup"].grasps] == [0.9, 0.6]
    assert [x.quality for x in g["bowl"].grasps] == [0.8]
    assert [x.quality for x in g[""].grasps] == [0.5]
    for grp in g.values():
        assert grp.classifications == [] and grp.depth_map == [] and grp.depth_width == 0
        assert grp.task == "grasp" and grp.duration_ms == 9
    grasps_only = Result.from_json({"task": "grasp", "model": "m", "grasps": [{"x": 1, "y": 1, "theta": 0, "width": 2, "quality": 0.3}]})
    assert list(grasps_only.group_by_class()) == [""]
