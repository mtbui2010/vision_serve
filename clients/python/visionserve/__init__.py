"""VisionServe Python client SDK.

A thin HTTP client for the VisionServe Go server (the runtime). This package is a
CLIENT only — it never performs inference itself; it talks to the server over HTTP
(default ``http://localhost:11435``).

Quickstart::

    from visionserve import Client

    c = Client()                       # http://localhost:11435
    c.health()                         # {"status": "ok"}
    c.list_models()                    # [ModelInfo, ...]
    c.load("rf-detr")
    res = c.predict("rf-detr", "cat.jpg")
    for d in res.detections:
        print(d.cls, d.conf, d.bbox)
"""

from .client import Client, VisionServeError
from .types import Classification, Detection, Grasp, Mask, Result, ModelInfo
from .postprocess import (
    CameraIntrinsics,
    backproject,
    camera_distance,
    get_depth_at_detection,
    grasp_distances,
    object_distances,
    select_target_grasp,
    select_target_object,
)

__all__ = [
    "Client",
    "VisionServeError",
    "Classification",
    "Detection",
    "Grasp",
    "Mask",
    "Result",
    "ModelInfo",
    "draw",
    "CameraIntrinsics",
    "backproject",
    "camera_distance",
    "get_depth_at_detection",
    "object_distances",
    "grasp_distances",
    "select_target_object",
    "select_target_grasp",
]


def __getattr__(name: str):  # noqa: N807 — PEP 562 module __getattr__
    """Lazily expose ``draw`` so that Pillow is NOT imported at package import time."""
    if name == "draw":
        from .visualize import draw as _draw

        return _draw
    raise AttributeError("module 'visionserve' has no attribute %r" % name)

__version__ = "0.1.0"
