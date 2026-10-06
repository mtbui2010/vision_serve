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

# Single source of the package version: pyproject.toml reads it (tool.setuptools.dynamic).
__version__ = "0.3.1"

from .client import Client, PreprocessResult, VisionServeError
from .sources import Frame, FrameSource, open_source
from .track import IoUTracker
from .types import Classification, Detection, FloatArray, Grasp, Mask, Result, ModelInfo
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
    "PreprocessResult",
    "Classification",
    "Detection",
    "FloatArray",
    "Grasp",
    "Mask",
    "Result",
    "ModelInfo",
    "Frame",
    "FrameSource",
    "open_source",
    "IoUTracker",
    "draw",
    "draw_prompts",
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
    """Lazily expose ``draw`` / ``draw_prompts`` so that Pillow is NOT imported at package
    import time."""
    if name in ("draw", "draw_prompts"):
        from . import visualize

        return getattr(visualize, name)
    raise AttributeError("module 'visionserve' has no attribute %r" % name)

