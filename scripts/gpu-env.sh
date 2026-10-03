#!/usr/bin/env bash
# Source this to run VisionServe on the GPU (ONNX Runtime CUDA EP):
#     source scripts/gpu-env.sh && VISIONSERVE_TRACE=1 make serve
#
# Requirements it wires up:
#   1) ORT_DYLIB_PATH -> a CUDA-enabled libonnxruntime.so (one whose directory also
#      contains libonnxruntime_providers_cuda.so). The default CPU `onnxruntime` wheel
#      does NOT have this — you need an onnxruntime-gpu build.
#   2) LD_LIBRARY_PATH -> the matching cuDNN/cuBLAS/cudart. On many setups these live in
#      a conda env's pip wheels (nvidia-*-cu12), NOT in /usr — so the CUDA EP can't find
#      libcudnn.so.9 unless we add them here (this is the usual cause of a SILENT fallback
#      to CPU; run with VISIONSERVE_TRACE=1 to see it).
#   3) LD_LIBRARY_PATH -> TensorRT 10 (libnvinfer.so.10 + libnvonnxparser.so.10), so the
#      TensorRT EP can load when opted into (`serve --tensorrt` / VISIONSERVE_TENSORRT=1).
#      Optional: the default chain is CUDA → CPU, and without it the opt-in degrades to CUDA.
#      The `tensorrt-cu12-libs` pip wheel puts these under site-packages/tensorrt_libs/.
#
# Override the ORT lib by exporting VISIONSERVE_ORT_GPU=/path/to/libonnxruntime.so first.

# 1) Locate a CUDA-enabled ORT shared library.
if [ -z "${VISIONSERVE_ORT_GPU:-}" ]; then
  _cuda_so="$(find "$HOME" -xdev -name 'libonnxruntime_providers_cuda.so' 2>/dev/null | head -1)"
  if [ -n "$_cuda_so" ]; then
    VISIONSERVE_ORT_GPU="$(ls "$(dirname "$_cuda_so")"/libonnxruntime.so* 2>/dev/null | head -1)"
  fi
fi
if [ -z "${VISIONSERVE_ORT_GPU:-}" ] || [ ! -e "${VISIONSERVE_ORT_GPU:-/nonexistent}" ]; then
  echo "gpu-env: no CUDA-enabled libonnxruntime.so found." >&2
  echo "gpu-env: install onnxruntime-gpu (or download the official ORT GPU build) and set" >&2
  echo "gpu-env:   export VISIONSERVE_ORT_GPU=/path/to/libonnxruntime.so" >&2
  return 1 2>/dev/null || exit 1
fi
export ORT_DYLIB_PATH="$VISIONSERVE_ORT_GPU"

# 2) Add cuDNN/CUDA runtime libs. Locate the cuDNN 9 wheel DIRECTLY (do NOT rely on
# CONDA_PREFIX — the active env is often `base`, which has no GPU wheels). Override with
# VISIONSERVE_CUDA_LIBS=/path1:/path2 if your CUDA libs live elsewhere (e.g. /usr/local/cuda).
if [ -n "${VISIONSERVE_CUDA_LIBS:-}" ]; then
  _nv="$VISIONSERVE_CUDA_LIBS"
else
  # Find the nvidia/ wheel root that ships libcudnn.so.9. Prefer the ACTIVE conda env
  # (CONDA_PREFIX), then a known good env, then any env under miniconda.
  _nvroot=""
  for _cand in "${CONDA_PREFIX:-}" "$HOME/miniconda3/envs/label"; do
    [ -n "$_cand" ] || continue
    _d="$(ls -d "$_cand"/lib/python*/site-packages/nvidia 2>/dev/null | head -1)"
    if [ -n "$_d" ] && [ -e "$_d"/cudnn/lib/libcudnn.so.9 ]; then _nvroot="$_d"; break; fi
  done
  if [ -z "$_nvroot" ]; then
    _cudnn="$(find "$HOME"/miniconda3 -maxdepth 8 -name 'libcudnn.so.9' 2>/dev/null | head -1)"
    [ -n "$_cudnn" ] && _nvroot="$(dirname "$(dirname "$(dirname "$_cudnn")")")"
  fi
  if [ -n "$_nvroot" ]; then
    _nv="$(echo "$_nvroot"/*/lib | tr ' ' ':')"
  else
    echo "gpu-env: warning: libcudnn.so.9 not found — CUDA EP will fall back to CPU." >&2
    _nv=""
  fi
fi
# 3) Add TensorRT 10 libs (libnvinfer.so.10, libnvonnxparser.so.10). Same "find the wheel
# dir" dance as cuDNN above: the `tensorrt-cu12-libs` wheel installs them into
# site-packages/tensorrt_libs/. This is OPTIONAL — the default chain is CUDA → CPU, and
# TensorRT is used only with `--tensorrt` / VISIONSERVE_TENSORRT=1 (or a manifest that lists
# tensorrt), so a missing TensorRT is a warning, not an error. Override with
# VISIONSERVE_TRT_LIBS=/path/to/tensorrt/lib if TRT is installed outside a wheel
# (e.g. an NVIDIA .tar.gz install, or JetPack's /usr/lib/aarch64-linux-gnu).
if [ -n "${VISIONSERVE_TRT_LIBS:-}" ]; then
  _trt="$VISIONSERVE_TRT_LIBS"
else
  # Prefer the ACTIVE conda env, then the env that ships the ORT lib we just picked,
  # then any env under miniconda.
  _trt=""
  for _cand in "${CONDA_PREFIX:-}" "${ORT_DYLIB_PATH%/lib/python*}"; do
    [ -n "$_cand" ] || continue
    _d="$(ls -d "$_cand"/lib/python*/site-packages/tensorrt_libs 2>/dev/null | head -1)"
    if [ -n "$_d" ] && [ -e "$_d"/libnvinfer.so.10 ]; then _trt="$_d"; break; fi
  done
  if [ -z "$_trt" ]; then
    _nvinfer="$(find "$HOME"/miniconda3 -maxdepth 8 -name 'libnvinfer.so.10' 2>/dev/null | head -1)"
    [ -n "$_nvinfer" ] && _trt="$(dirname "$_nvinfer")"
  fi
  [ -z "$_trt" ] && echo "gpu-env: warning: libnvinfer.so.10 not found — the TensorRT opt-in (--tensorrt) would fall back to the CUDA EP." >&2
fi

export LD_LIBRARY_PATH="$(dirname "$ORT_DYLIB_PATH"):${_nv}${_trt:+:$_trt}:${LD_LIBRARY_PATH:-}"

# One concise line to stderr (never pollutes JSON on stdout, e.g. `make run`).
echo "gpu-env: GPU on (ORT=$(basename "$ORT_DYLIB_PATH"), cuDNN=$([ -n "${_nv:-}" ] && echo found || echo MISSING), TensorRT=$([ -n "${_trt:-}" ] && echo found || echo absent)); set VISIONSERVE_TRACE=1 for EP details" >&2
