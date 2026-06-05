# TorchServe baseline — W1

TorchServe is the canonical PyTorch production server; included as a W1 baseline (does **not**
share the ONNX file — it runs the ORT export through ORT *or* a torch model, your choice via the
handler). Use it to show VisionServe vs a mainstream Python serving framework.

## Archive + serve

```bash
# Option A (engine-controlled): serve the SAME .onnx via onnxruntime inside the handler.
torch-model-archiver \
    --model-name mobilenet_v3 \
    --version 1.0 \
    --serialized-file ~/.visionserve/models/mobilenet-v3/model.onnx \
    --handler eval/baselines/torchserve/handler.py \
    --export-path /tmp/ts_store -f

torchserve --start --model-store /tmp/ts_store --models mobilenet_v3=mobilenet_v3.mar \
    --ts-config eval/baselines/torchserve/config.properties

# Inference endpoint: POST http://localhost:8080/predictions/mobilenet_v3  (raw image bytes)
```

The handler exposes the prediction; to keep the **same** loadgen client, front it with the
`/api/predict` shape via `eval/loadgen` `--api torchserve` adapter, or hit TorchServe's native
`/predictions/<model>` endpoint directly (the loadgen scripts support a path override).

No measured numbers are committed here.
