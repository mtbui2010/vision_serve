"""Client helpers that speak VisionServe's HTTP API.

VisionServe exposes ``POST /api/predict`` accepting either:

* JSON: ``{"model": "<name>", "image_base64": "<b64>", "prompt"/"box"/"point": ...}``
* multipart: form fields ``model=<name>``, ``image=<file>`` (+ optional prompt fields)

and returns the unified ``Result`` schema (see ``pkg/api/types.go``):

    {
      "task": "...", "model": "...", "device": "gpu:0+trt",
      "detections": [...], "masks": [...], "classifications": [...],
      "embeddings": [...], "depth_map": [...], "duration_ms": 12.3
    }

The baselines under ``eval/baselines`` deliberately mirror this request/response shape so
the *same* loadgen and accuracy clients can target VisionServe and every baseline unchanged.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Optional


def encode_image(path: str) -> str:
    """Read an image file and return its base64 string (for the JSON request branch)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def encode_image_bytes(raw: bytes) -> str:
    """Base64-encode already-loaded image bytes."""
    return base64.b64encode(raw).decode("ascii")


def build_json_request(
    model: str,
    image_b64: str,
    prompt: Optional[str] = None,
    box: Optional[str] = None,
    point: Optional[str] = None,
) -> dict[str, Any]:
    """Build the JSON body for ``POST /api/predict`` (VisionServe shape)."""
    body: dict[str, Any] = {"model": model, "image_base64": image_b64}
    if prompt:
        body["prompt"] = prompt
    if box:
        body["box"] = box
    if point:
        body["point"] = point
    return body


@dataclass
class PredictResult:
    """Lightweight view over the unified ``Result`` JSON."""

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return self.raw.get("task", "")

    @property
    def device(self) -> str:
        """EP/device the server actually used, e.g. ``gpu:0+trt`` — used for provenance."""
        return self.raw.get("device", "")

    @property
    def duration_ms(self) -> float:
        """Server-side inference duration the server reports (``Result.duration_ms``)."""
        return float(self.raw.get("duration_ms", 0.0))

    @property
    def classifications(self) -> list[dict[str, Any]]:
        return self.raw.get("classifications") or []

    @property
    def detections(self) -> list[dict[str, Any]]:
        return self.raw.get("detections") or []

    @property
    def masks(self) -> list[dict[str, Any]]:
        return self.raw.get("masks") or []

    def top1(self) -> Optional[str]:
        """Top-1 class label for classification tasks, or ``None`` if absent."""
        cls = self.classifications
        if not cls:
            return None
        return max(cls, key=lambda c: c.get("conf", 0.0)).get("class")


def parse_result(body: bytes | str) -> PredictResult:
    """Parse a ``/api/predict`` response body into a :class:`PredictResult`."""
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return PredictResult(raw=json.loads(body))


# Default VisionServe listen address (avoids clashing with Ollama's 11434).
DEFAULT_TARGET = "http://localhost:11435"
PREDICT_PATH = "/api/predict"
HEALTH_PATH = "/api/health"
