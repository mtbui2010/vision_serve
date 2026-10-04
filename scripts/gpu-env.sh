#!/usr/bin/env bash
# Source this to run VisionServe on the GPU (ONNX Runtime CUDA EP):
#     source scripts/gpu-env.sh && VISIONSERVE_TRACE=1 make serve
#
# Requirements it wires up:
#   1) ORT_DYLIB_PATH -> a CUDA-enabled libonnxruntime.so (one whose directory also
#      contains libonnxruntime_providers_cuda.so) whose CUDA EP can actually load on THIS
#      machine. The default CPU `onnxruntime` wheel does NOT have a CUDA EP — you need an
#      onnxruntime-gpu build. Every candidate found under $HOME is checked (see below) and
#      the newest one that passes is used; the script prints which one and why the others
#      were skipped. A newer ORT is not always usable: onnxruntime-gpu 1.30 links the CUDA 13
#      runtime, which a driver that supports only CUDA 12.x refuses ("CUDA driver version is
#      insufficient"), and the EP then falls back to the CPU without failing.
#   2) LD_LIBRARY_PATH -> the matching cuDNN/cuBLAS/cudart. On many setups these live in
#      a conda env's pip wheels (nvidia-*-cu12), NOT in /usr — so the CUDA EP can't find
#      libcudnn.so.9 unless we add them here (this is the usual cause of a SILENT fallback
#      to CPU; run with VISIONSERVE_TRACE=1 to see it).
#   3) LD_LIBRARY_PATH -> TensorRT 10 (libnvinfer.so.10 + libnvonnxparser.so.10), so the
#      TensorRT EP can load when opted into (`serve --tensorrt` / VISIONSERVE_TENSORRT=1).
#      Optional: the default chain is CUDA → CPU, and without it the opt-in degrades to CUDA.
#      The `tensorrt-cu12-libs` pip wheel puts these under site-packages/tensorrt_libs/.
#
# A candidate passes when
#   - its libonnxruntime_providers_cuda.so links a CUDA runtime (libcudart.so.N) whose major
#     version N is not newer than the CUDA version the driver supports (`nvidia-smi` header;
#     CUDA minor-version compatibility covers a newer minor of the same major), and
#   - every library that provider needs resolves (`ldd`, with the cuDNN/CUDA wheel dirs of the
#     same conda env on the search path).
# The script never selects a library that fails the check, so `make serve` either runs on the
# CUDA EP or says it uses the CPU.
#
# Override the ORT lib by exporting VISIONSERVE_ORT_GPU=/path/to/libonnxruntime.so first (it
# is still checked, and a failing check is reported, but the override is honoured).

# Driver CUDA version ("12.8"), from the nvidia-smi header; empty when unknown.
_vs_drv_cuda=""
if command -v nvidia-smi >/dev/null 2>&1; then
  _vs_drv_cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
fi

# _vs_nvlibs <ort lib>: colon-separated lib dirs of the nvidia-* pip wheels that belong with
# this ORT lib: the same env's site-packages/nvidia first, then the active conda env, then a
# known-good env, then any env under ~/miniconda3 that ships libcudnn.so.9.
# VISIONSERVE_CUDA_LIBS=/path1:/path2 overrides (e.g. /usr/local/cuda/lib64).
_vs_nvlibs() {
  if [ -n "${VISIONSERVE_CUDA_LIBS:-}" ]; then echo "$VISIONSERVE_CUDA_LIBS"; return; fi
  local _root="" _cand _d _cudnn
  case "$1" in
    */site-packages/*)
      _d="${1%%/site-packages/*}/site-packages/nvidia"
      [ -e "$_d"/cudnn/lib/libcudnn.so.9 ] && _root="$_d" ;;
  esac
  if [ -z "$_root" ]; then
    for _cand in "${CONDA_PREFIX:-}" "$HOME/miniconda3/envs/label"; do
      [ -n "$_cand" ] || continue
      _d="$(ls -d "$_cand"/lib/python*/site-packages/nvidia 2>/dev/null | head -1)"
      if [ -n "$_d" ] && [ -e "$_d"/cudnn/lib/libcudnn.so.9 ]; then _root="$_d"; break; fi
    done
  fi
  if [ -z "$_root" ]; then
    _cudnn="$(find "$HOME"/miniconda3 -maxdepth 8 -name 'libcudnn.so.9' 2>/dev/null | head -1)"
    [ -n "$_cudnn" ] && _root="$(dirname "$(dirname "$(dirname "$_cudnn")")")"
  fi
  [ -n "$_root" ] && echo "$_root"/*/lib | tr ' ' ':'
}

# _vs_check <ort lib>: returns 0 when the lib's CUDA EP can load here; sets _vs_why (the
# reason, either way) and _vs_nv (the wheel lib dirs it was checked with).
_vs_check() {
  local _dir _prov _ldd _rt _missing
  _dir="$(dirname "$1")"
  _prov="$_dir/libonnxruntime_providers_cuda.so"
  _vs_nv="$(_vs_nvlibs "$1")"
  if [ ! -e "$_prov" ]; then _vs_why="no libonnxruntime_providers_cuda.so next to it"; return 1; fi
  if ! command -v ldd >/dev/null 2>&1; then _vs_why="cannot verify (no ldd)"; return 0; fi
  _ldd="$(LD_LIBRARY_PATH="$_dir:$_vs_nv:${LD_LIBRARY_PATH:-}" ldd "$_prov" 2>/dev/null)"
  _rt="$(echo "$_ldd" | sed -n 's/^[[:space:]]*libcudart\.so\.\([0-9][0-9]*\).*/\1/p' | head -1)"
  if [ -n "$_rt" ] && [ -n "$_vs_drv_cuda" ] && [ "$_rt" -gt "${_vs_drv_cuda%%.*}" ]; then
    _vs_why="built for CUDA $_rt, but the driver supports CUDA $_vs_drv_cuda (the CUDA EP would fail with 'driver version is insufficient' and fall back to CPU)"
    return 1
  fi
  _missing="$(echo "$_ldd" | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
  if [ -n "$_missing" ]; then _vs_why="missing ${_missing% }"; return 1; fi
  _vs_why="CUDA ${_rt:-?} runtime, driver supports CUDA ${_vs_drv_cuda:-unknown}, all CUDA EP libraries resolve"
  return 0
}

# 1) Pick the ORT shared library.
_vs_ort=""
if [ -n "${VISIONSERVE_ORT_GPU:-}" ]; then
  _vs_ort="$VISIONSERVE_ORT_GPU"
  if [ ! -e "$_vs_ort" ]; then
    echo "gpu-env: VISIONSERVE_ORT_GPU=$_vs_ort does not exist." >&2
    return 1 2>/dev/null || exit 1
  fi
  if _vs_check "$_vs_ort"; then
    echo "gpu-env: using VISIONSERVE_ORT_GPU=$_vs_ort ($_vs_why)" >&2
  else
    echo "gpu-env: WARNING: VISIONSERVE_ORT_GPU=$_vs_ort fails the check: $_vs_why — the CUDA EP will likely fall back to CPU (VISIONSERVE_TRACE=1 shows it)" >&2
  fi
else
  # Every CUDA-enabled ORT under $HOME, newest version first (libonnxruntime.so.1.26.0 sorts
  # by its version; an unversioned libonnxruntime.so.1 sorts last).
  _vs_cands="$(find "$HOME" -xdev -name 'libonnxruntime_providers_cuda.so' 2>/dev/null | while read -r _p; do
      ls "$(dirname "$_p")"/libonnxruntime.so* 2>/dev/null | sort -V | tail -1
    done | awk -F'libonnxruntime.so.' '{print $NF "\t" $0}' | sort -t"$(printf '\t')" -k1,1Vr | cut -f2)"
  if [ -z "$_vs_cands" ]; then
    echo "gpu-env: no CUDA-enabled libonnxruntime.so found under $HOME." >&2
  fi
  while IFS= read -r _c; do
    [ -n "$_c" ] || continue
    if _vs_check "$_c"; then
      _vs_ort="$_c"
      echo "gpu-env: chose $_c ($_vs_why)" >&2
      break
    fi
    echo "gpu-env: skip $_c: $_vs_why" >&2
  done <<EOF
$_vs_cands
EOF
  if [ -z "$_vs_ort" ]; then
    echo "gpu-env: no CUDA-enabled ORT lib whose CUDA EP can load on this driver (CUDA ${_vs_drv_cuda:-unknown})." >&2
    echo "gpu-env: install an onnxruntime-gpu build for CUDA ${_vs_drv_cuda%%.*}.x (plus its nvidia-*-cu${_vs_drv_cuda%%.*} wheels), or set" >&2
    echo "gpu-env:   export VISIONSERVE_ORT_GPU=/path/to/libonnxruntime.so" >&2
    unset -f _vs_nvlibs _vs_check
    return 1 2>/dev/null || exit 1
  fi
fi
export ORT_DYLIB_PATH="$_vs_ort"

# 2) cuDNN/CUDA runtime libs: the wheel dirs the chosen lib was checked with.
_nv="$_vs_nv"
[ -z "$_nv" ] && echo "gpu-env: warning: libcudnn.so.9 not found — CUDA EP will fall back to CPU." >&2

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
unset -f _vs_nvlibs _vs_check
