# deploy/Dockerfile — self-contained VisionServe SERVER image (x86_64 / linux-amd64).
#
# Two variants (--build-arg ORT_VARIANT):
#   cpu (default): slim Debian + ONNX Runtime CPU build.            ~141 MB
#   gpu           : CUDA 12.4 + cuDNN 9 + TensorRT 10 + ORT GPU.   ~3-5 GB
#
# The GPU variant includes libnvinfer.so.10 so the TensorRT EP is usable
# (TRT is the fastest ORT EP: ~2-4× over the plain CUDA EP).
#
# Usage:
#   # CPU image (default):
#   make docker
#   docker run --rm -p 11435:11435 -v "$PWD/models:/models" visionserve:latest
#
#   # GPU image (TensorRT EP):
#   make docker ORT_VARIANT=gpu
#   docker run --rm --gpus all -p 11435:11435 -v "$PWD/models:/models" visionserve:gpu
#
# Model weights are NOT baked in — mount them via the /models VOLUME.
# See deploy/README.md.

# ── pinned versions ──────────────────────────────────────────────────────────
# ORT 1.20.1: compatible with TensorRT 10.x (libnvinfer.so.10) and CUDA 12.
ARG ORT_VERSION=1.20.1
# TensorRT 10.6 GA — last GA release with libnvinfer.so.10, CUDA 12 compatible.
ARG TRT_VERSION=10.6.0
# Build variant: "cpu" or "gpu".
ARG ORT_VARIANT=cpu

############################
# Stage 1 — build the Go binary
############################
# yalue/onnxruntime_go requires CGO_ENABLED=1 (it uses the ORT C API via cgo).
# libonnxruntime.so is only needed at RUNTIME (dlopen), not at build time.
FROM golang:1.22-bookworm AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
ARG VERSION=docker
RUN CGO_ENABLED=1 go build \
        -ldflags "-s -w -X visionserve/internal/cli.Version=${VERSION}" \
        -o /out/visionserve ./cmd/visionserve

############################
# Stage 2a — fetch ONNX Runtime (CPU)
############################
FROM debian:bookworm-slim AS ort-cpu
ARG ORT_VERSION=1.20.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"; \
    curl -fSL "$url" -o /tmp/ort.tgz; \
    mkdir -p /opt/ort; \
    tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1; \
    rm /tmp/ort.tgz

############################
# Stage 2b — fetch ONNX Runtime (GPU, includes TensorRT + CUDA EP providers)
############################
FROM debian:bookworm-slim AS ort-gpu
ARG ORT_VERSION=1.20.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
# The GPU tgz ships libonnxruntime_providers_cuda.so + libonnxruntime_providers_tensorrt.so
# alongside libonnxruntime.so — all go into /opt/ort/lib/.
RUN set -eux; \
    url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"; \
    curl -fSL "$url" -o /tmp/ort.tgz; \
    mkdir -p /opt/ort; \
    tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1; \
    rm /tmp/ort.tgz; \
    ls /opt/ort/lib/

############################
# Stage 3a — CPU runtime (minimal Debian)
############################
FROM debian:bookworm-slim AS runtime-cpu
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

############################
# Stage 3b — GPU runtime: CUDA 12.4 + cuDNN 9 + TensorRT 10
############################
# Base provides CUDA runtime + cuDNN (needed by the ORT CUDA EP).
# We then install TensorRT 10.x from NVIDIA's package repo so that
# libnvinfer.so.10 is present for the ORT TensorRT EP.
#
# TensorRT 10 on CUDA 12: apt package tensorrt (meta) pulls in:
#   libnvinfer10, libnvinfer-plugin10, libnvonnxparsers10, libnvparsers10
# which are exactly what ORT's libonnxruntime_providers_tensorrt.so dlopen()s.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime-gpu
ARG TRT_VERSION=10.6.0
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*
# Add NVIDIA TensorRT package repository (same key/repo used by TensorRT DEB installer).
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg2 && \
    wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" && \
    dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    rm -rf /var/lib/apt/lists/*
# Install TensorRT libraries only (no Python bindings — keeps image leaner).
# libnvinfer-bin + libnvinfer-dev are not needed at runtime; only the shared libs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libnvinfer10 \
        libnvinfer-plugin10 \
        libnvonnxparsers10 && \
    rm -rf /var/lib/apt/lists/*

############################
# Final — select runtime + ORT by variant, wire everything together
############################
# ARG expands in FROM but NOT in COPY --from= (BuildKit limitation).
# Workaround: alias the chosen stage via FROM, then COPY from the literal alias.
FROM ort-${ORT_VARIANT} AS ort-selected
FROM runtime-${ORT_VARIANT} AS final

# ORT shared libraries (cpu: libonnxruntime.so only;
#                       gpu: + libonnxruntime_providers_cuda.so
#                             + libonnxruntime_providers_tensorrt.so
#                             + libonnxruntime_providers_shared.so)
COPY --from=ort-selected /opt/ort/lib/libonnxruntime*.so* /usr/local/lib/

# Server binary.
COPY --from=build /out/visionserve /usr/local/bin/visionserve

# ORT finds the core library at this path (dlopen by the binding).
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
# ORT provider plugins + TensorRT/CUDA libs must be on LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH=/usr/local/lib

RUN ldconfig

# Model weights are stored in ~/.visionserve/models (/root/.visionserve/models when
# running as root). Mount a named volume here to persist models across container restarts:
#   docker run -v visionserve:/root/.visionserve ...
VOLUME ["/root/.visionserve"]

EXPOSE 11435
ENTRYPOINT ["visionserve"]
# No --models flag needed: the binary defaults to ~/.visionserve/models.
CMD ["serve", "--addr", ":11435"]
