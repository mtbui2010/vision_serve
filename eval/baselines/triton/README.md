# Triton (ONNX Runtime backend) baseline — W1 / W4

This is a model-repository **template**. Triton serves the **same** ONNX file VisionServe does,
via its ONNX Runtime backend with the CUDA EP — another engine-controlled W1 baseline, and the
reference point for the W4 *dynamic-batching* comparison (VisionServe is batch=1).

## Layout

```
model_repository/
  mobilenet_v3_onnx/
    config.pbtxt          # template below — EDIT shapes/names per model
    1/
      model.onnx          # <-- COPY the SAME .onnx VisionServe serves into here (not committed)
```

## Bring-up

```bash
# 1. Copy the exact ONNX export (do NOT re-export; reuse VisionServe's file)
cp ~/.visionserve/models/mobilenet-v3/model.onnx \
   eval/baselines/triton/model_repository/mobilenet_v3_onnx/1/model.onnx

# 2. Verify input/output names + shapes so config.pbtxt matches the graph
python -c "import onnxruntime as o; s=o.InferenceSession('eval/baselines/triton/model_repository/mobilenet_v3_onnx/1/model.onnx', providers=['CPUExecutionProvider']); \
print([(i.name,i.shape) for i in s.get_inputs()]); print([(o_.name,o_.shape) for o_ in s.get_outputs()])"
# TODO: paste the real names/shapes into config.pbtxt (left as placeholders below).

# 3. Launch
tritonserver --model-repository=eval/baselines/triton/model_repository

# 4. Drive with perf_analyzer (open-loop, CO-correct) for the W4 sweep:
perf_analyzer -m mobilenet_v3_onnx --concurrency-range 1:64:2 \
    --measurement-mode count_windows --percentile 99 \
    -i grpc -u localhost:8001
```

## Notes
- `perf_analyzer` speaks Triton's KServe protocol, **not** VisionServe's `/api/predict`. So for
  Triton we use `perf_analyzer` directly for latency/throughput (it is CO-correct) and record its
  output via the same provenance sidecar. The shared `loadgen/sweep.py` is for the HTTP
  `/api/predict` systems (VisionServe + FastAPI baselines).
- To exercise **dynamic batching** (the W4 contrast vs VisionServe batch=1), uncomment the
  `dynamic_batching` block and sweep `max_queue_delay_microseconds`.
- No measured numbers are committed here.
