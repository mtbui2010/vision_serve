#!/bin/sh
# Routes each format to the venv that has its framework. TensorFlow/tf2onnx and PyTorch cannot share
# one environment (tf2onnx pins protobuf ~=3.20 and numpy<2; torch/transformers need numpy 2).
set -e
case "${1:-}" in
  tensorflow|keras|tflite) PY=/opt/tfenv/bin/python ;;
  *)                       PY=/opt/torchenv/bin/python ;;
esac
exec "$PY" -m visionserve.convert "$@"
