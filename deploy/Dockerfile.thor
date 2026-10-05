# deploy/Dockerfile.thor — VisionServe SERVER image for NVIDIA Jetson AGX Thor (JetPack 7, CUDA 13).
#
# STATUS: builds (docker buildx --platform linux/arm64, checked on an x86-64 host), NOT YET RUN ON A
# THOR. See website/docs/guides/edge.md ("Thor checklist") for what to verify on the device.
#
# Why a separate image from Dockerfile.edge (JetPack 6 / Orin):
#   - JetPack 7 is "SBSA-aligned": Thor uses the standard arm64 (SBSA) CUDA 13 packages and NGC
#     images, not the L4T-specific ones. The l4t-base / l4t-ml images stop at r36 (JetPack 6); there
#     are no r38 tags.
#   - Thor's GPU is Blackwell, compute capability 11.0 (sm_110 from CUDA 13 on). The ONNX Runtime
#     build must carry sm_110 kernels and link CUDA 13. Microsoft publishes no aarch64 CUDA build;
#     Jetson AI Lab publishes onnxruntime-gpu 1.24.0 for sbsa/cu130 (CUDA EP + TensorRT EP, kernels
#     for sm_87, sm_110, sm_120, sm_121). The libraries are taken out of that wheel.
#
#   docker buildx build --platform linux/arm64 -f deploy/Dockerfile.thor -t visionserve:thor --load .
#   # on the Thor (NVIDIA container runtime installed with JetPack 7):
#   docker run --rm --runtime nvidia --gpus all -p 11435:11435 \
#       -v ~/.visionserve_models:/root/.models visionserve:thor
#
# TensorRT: the base image has CUDA 13 + cuDNN but no TensorRT, so the opt-in TensorRT EP
# (--tensorrt) falls back to CUDA. For TensorRT use a base with TensorRT 10.13+ for arm64, e.g.
#   --build-arg BASE=nvcr.io/nvidia/tensorrt:25.08-py3   (much larger image)
# and re-measure accuracy under TensorRT (BUGS_TO_FIX.md #3).
#
# Every stage that RUNs anything runs on the BUILD platform (no QEMU needed); the arm64 stage only
# copies files.

ARG VERSION=docker-thor
ARG BASE=nvcr.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04
ARG ORT_WHEEL_INDEX=https://pypi.jetson-ai-lab.io/sbsa/cu130
ARG ORT_VERSION=1.24.0
# sha256 of onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl from the index above (2026-10-05).
ARG ORT_WHEEL_SHA256=012c10bef23a39f074730d158b72f797a7314f2695c65835a0669b57282422f6

############################
# Stage 1 — the Go binary for arm64 (cross-compiled, as in Dockerfile.edge)
############################
FROM --platform=$BUILDPLATFORM golang:1.22-bookworm AS build
ARG VERSION
WORKDIR /src
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross && \
    rm -rf /var/lib/apt/lists/*
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 GOOS=linux GOARCH=arm64 CC=aarch64-linux-gnu-gcc \
    go build -ldflags "-s -w -X visionserve/internal/cli.Version=${VERSION}" \
    -o /out/visionserve ./cmd/visionserve

############################
# Stage 2 — ONNX Runtime (CUDA 13, sm_110) out of the Jetson AI Lab wheel, on the build platform
############################
FROM --platform=$BUILDPLATFORM python:3.12-slim-bookworm AS ort
ARG ORT_WHEEL_INDEX
ARG ORT_VERSION
ARG ORT_WHEEL_SHA256
RUN set -eux; \
    pip download --no-deps --only-binary=:all: --platform linux_aarch64 --python-version 3.12 \
        --index-url "${ORT_WHEEL_INDEX}" "onnxruntime-gpu==${ORT_VERSION}" -d /tmp/whl; \
    whl="$(ls /tmp/whl/onnxruntime_gpu-*.whl)"; \
    echo "${ORT_WHEEL_SHA256}  ${whl}" | sha256sum -c -; \
    mkdir -p /export; \
    python - "$whl" <<'EOF'
import sys, zipfile, os
z = zipfile.ZipFile(sys.argv[1])
libs = [n for n in z.namelist() if n.startswith("onnxruntime/capi/libonnxruntime")]
assert any("providers_cuda" in n for n in libs), "the wheel has no CUDA execution provider"
for n in libs:
    with z.open(n) as src, open(os.path.join("/export", os.path.basename(n)), "wb") as dst:
        dst.write(src.read())
core = [os.path.basename(n) for n in libs if os.path.basename(n).startswith("libonnxruntime.so.")]
assert len(core) == 1, core
os.symlink(core[0], "/export/libonnxruntime.so")
print("ONNX Runtime libraries:", sorted(os.listdir("/export")))
EOF

############################
# Final — CUDA 13 + cuDNN 9 runtime (arm64/SBSA), ORT core + providers side by side
############################
FROM ${BASE} AS final
# Core + provider libraries from the same wheel, in one directory: ORT dlopens the providers from
# the directory libonnxruntime.so was loaded from.
COPY --from=ort /export/ /opt/onnxruntime/lib/
COPY --from=build /out/visionserve /usr/local/bin/visionserve
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64 \
    VISIONSERVE_MODELS=/root/.models
VOLUME ["/root/.models"]
EXPOSE 11435
ENTRYPOINT ["visionserve"]
CMD ["serve", "--addr", ":11435"]
