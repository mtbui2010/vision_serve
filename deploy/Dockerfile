# deploy/Dockerfile — self-contained VisionServe SERVER image (x86_64 / linux-amd64).
#
# Goal: `docker run -p 11435:11435 visionserve serve` just works — the Go binary and
# ONNX Runtime are bundled, no host setup. ONNX Runtime is downloaded (pinned) at build
# time and loaded at runtime via dlopen through ORT_DYLIB_PATH.
#
# Two variants are selectable with --build-arg ORT_VARIANT:
#   - cpu (default): slim Debian runtime + ONNX Runtime CPU build.
#   - gpu          : nvidia/cuda runtime + ONNX Runtime GPU build (CUDA/TensorRT EP).
#
# Examples:
#   docker build -f deploy/Dockerfile -t visionserve .
#   docker build -f deploy/Dockerfile --build-arg ORT_VARIANT=gpu -t visionserve:gpu .
#
# Model weights are NOT baked into the image (large, not committed to git). They live in
# the /models VOLUME — see deploy/README.md for how to populate it.

# Pinned ONNX Runtime version. v1.20.1 is a real, stable release with both
# linux-x64 (CPU) and linux-x64-gpu assets on the official GitHub releases page.
ARG ORT_VERSION=1.20.1
# Build variant: "cpu" (default) or "gpu".
ARG ORT_VARIANT=cpu

############################
# Stage 1 — build the Go binary
############################
# The yalue/onnxruntime_go binding links against the ONNX Runtime C API via cgo, so
# CGO_ENABLED=1 is required (CGO_ENABLED=0 fails with "build constraints exclude all Go
# files"). The golang image already ships gcc. libonnxruntime.so is only needed at
# RUNTIME (dlopen), not at build time, so we do not fetch it here.
FROM golang:1.22-bookworm AS build
WORKDIR /src

# Cache module downloads first for faster rebuilds.
COPY go.mod go.sum* ./
RUN go mod download

COPY . .

# Keep ldflags aligned with the Makefile (strip + inject version).
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
# CPU build asset: onnxruntime-linux-x64-<ver>.tgz
RUN set -eux; \
    url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"; \
    curl -fSL "$url" -o /tmp/ort.tgz; \
    mkdir -p /opt/ort; \
    tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1; \
    rm /tmp/ort.tgz; \
    ls -l /opt/ort/lib/libonnxruntime.so*

############################
# Stage 2b — fetch ONNX Runtime (GPU)
############################
FROM debian:bookworm-slim AS ort-gpu
ARG ORT_VERSION=1.20.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
# GPU build asset: onnxruntime-linux-x64-gpu-<ver>.tgz (CUDA + TensorRT execution providers).
RUN set -eux; \
    url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"; \
    curl -fSL "$url" -o /tmp/ort.tgz; \
    mkdir -p /opt/ort; \
    tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1; \
    rm /tmp/ort.tgz; \
    ls -l /opt/ort/lib/libonnxruntime.so*

############################
# Stage 3a — CPU runtime (slim Debian)
############################
FROM debian:bookworm-slim AS runtime-cpu
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

############################
# Stage 3b — GPU runtime (CUDA)
############################
# nvidia/cuda runtime provides the CUDA/cuDNN libraries the ORT GPU build links against.
# Requires nvidia-container-toolkit on the host and `docker run --gpus all`.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime-gpu
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

############################
# Final — pick runtime + ORT by variant
############################
# ARG expands in FROM but NOT in `COPY --from=<stage>` (BuildKit limitation). So we
# alias the chosen ORT download stage via a FROM, then COPY from that literal alias.
# BuildKit only builds the stages actually referenced, so the unused variant never runs.
FROM ort-${ORT_VARIANT} AS ort-selected

# `runtime-${ORT_VARIANT}` resolves to runtime-cpu or runtime-gpu.
FROM runtime-${ORT_VARIANT} AS final

# Bundle the matching ONNX Runtime shared libraries from the aliased stage.
COPY --from=ort-selected /opt/ort/lib/libonnxruntime.so* /usr/local/lib/

# Server binary.
COPY --from=build /out/visionserve /usr/local/bin/visionserve

# The binding dlopen()s this exact path at startup.
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib

RUN ldconfig

# Model weights + manifests are mounted here at runtime (not baked into the image).
VOLUME ["/models"]

EXPOSE 11435
ENTRYPOINT ["visionserve"]
CMD ["serve", "--addr", ":11435", "--models", "/models"]
