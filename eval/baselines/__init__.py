"""Serving baselines for the W1 engine-controlled comparison.

All baselines load the SAME ONNX file VisionServe serves and expose ``POST /api/predict``
with VisionServe's request/response shape, so the speedup measured by the loadgen is
attributable to the serving layer, not the runtime.
"""
