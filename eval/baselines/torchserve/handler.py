"""TorchServe custom handler skeleton (W1 baseline).

Serves the SAME ONNX export VisionServe uses, through onnxruntime, inside a TorchServe worker.
This keeps the runtime constant (engine-controlled) while measuring TorchServe's serving overhead.

The classification path is implemented; detection/segmentation decode is model-specific and
left as a TODO (we never emit fabricated boxes). Preprocess is delegated to the shared
``eval.baselines.preprocess`` so it matches the FastAPI+ORT baseline byte-for-byte.

Archive with ``torch-model-archiver`` (see README.md) pointing ``--serialized-file`` at the .onnx
and ``--handler`` at this file.
"""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np


class OnnxClassificationHandler:
    """Minimal TorchServe handler interface: ``initialize`` + ``handle``."""

    def __init__(self) -> None:
        self.session = None
        self.input_name = ""
        self.output_names: list[str] = []
        self.cfg = None
        self.labels: list[str] | None = None
        self.task = os.environ.get("TASK", "classification")
        self.model_name = os.environ.get("MODEL_NAME", "mobilenet-v3")

    def initialize(self, context: Any) -> None:
        import onnxruntime as ort  # imported lazily inside the worker

        from eval.baselines.preprocess import PreprocessConfig

        props = context.system_properties
        model_dir = props.get("model_dir", ".")
        # TorchServe places --serialized-file in model_dir; we expect the .onnx there.
        onnx_path = os.environ.get("ONNX_PATH") or self._find_onnx(model_dir)
        providers = os.environ.get(
            "EP", "CUDAExecutionProvider,CPUExecutionProvider"
        ).split(",")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.cfg = PreprocessConfig(
            width=int(os.environ.get("INPUT_W", "224")),
            height=int(os.environ.get("INPUT_H", "224")),
            letterbox=os.environ.get("LETTERBOX", "false").lower() in ("1", "true", "yes"),
        )
        labels_path = os.environ.get("LABELS_PATH")
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, encoding="utf-8") as f:
                self.labels = [ln.rstrip("\n") for ln in f]

    @staticmethod
    def _find_onnx(model_dir: str) -> str:
        for fn in os.listdir(model_dir):
            if fn.endswith(".onnx"):
                return os.path.join(model_dir, fn)
        raise FileNotFoundError(f"no .onnx found in TorchServe model_dir {model_dir!r}")

    def _read_image_bytes(self, row: Any) -> bytes:
        """Extract raw image bytes from a TorchServe request row."""
        data = row.get("data") or row.get("body")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):
            return data.encode("latin-1")
        if hasattr(data, "read"):
            return data.read()
        raise ValueError("unsupported TorchServe request payload")

    def handle(self, data: list[Any], context: Any) -> list[dict[str, Any]]:
        from eval.baselines.preprocess import preprocess

        results: list[dict[str, Any]] = []
        for row in data:
            raw = self._read_image_bytes(row)
            tensor, _meta = preprocess(raw, self.cfg)
            outputs = self.session.run(self.output_names, {self.input_name: tensor})
            results.append(self._postprocess(outputs))
        return results

    def _postprocess(self, outputs: list[np.ndarray]) -> dict[str, Any]:
        if self.task == "classification":
            logits = np.asarray(outputs[0]).reshape(-1)
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            idx = np.argsort(-probs)[:5]
            cls = [
                {
                    "class": self.labels[i] if self.labels and i < len(self.labels) else str(int(i)),
                    "conf": float(probs[i]),
                }
                for i in idx
            ]
            return {"task": self.task, "model": self.model_name, "classifications": cls}
        # TODO: requires per-model detection/segmentation decode (see fastapi_ort._postprocess).
        return {"task": self.task, "model": self.model_name, "detections": [], "masks": []}


_service = OnnxClassificationHandler()


def handle(data: Any, context: Any):
    """TorchServe entry point."""
    if data is None:
        return None
    if _service.session is None:
        _service.initialize(context)
    return _service.handle(data, context)


# silence "imported but unused" for the io import kept for future image-stream handling
_ = io
