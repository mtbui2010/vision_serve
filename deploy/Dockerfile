# deploy/Dockerfile — image GPU (x86_64) cho VisionServe.
# Multi-stage: build Go binary, rồi chạy trên base CUDA kèm ONNX Runtime.

# --- Stage build ---
FROM golang:1.22-bookworm AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download || true
COPY . .
# Binding ORT dùng cgo (ORT C API) -> PHẢI CGO_ENABLED=1 (golang image đã có gcc).
# Lib libonnxruntime.so chỉ cần lúc RUNTIME (dlopen), không cần lúc build.
RUN CGO_ENABLED=1 go build -ldflags '-s -w' -o /out/visionserve ./cmd/visionserve

# --- Stage runtime (GPU) ---
# Cần ONNX Runtime build kèm CUDA/TensorRT EP. Bản nvidia/cuda chỉ có CUDA runtime;
# bạn phải mount/COPY libonnxruntime.so (build GPU) và set ORT_DYLIB_PATH.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# TODO: COPY libonnxruntime.so (bản có CUDA/TensorRT EP) vào /usr/local/lib/
#   COPY onnxruntime/lib/libonnxruntime.so /usr/local/lib/
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

COPY --from=build /out/visionserve /usr/local/bin/visionserve
# models/ thường mount volume lúc chạy (weights lớn, không bake vào image)
VOLUME ["/models"]
ENV VISIONSERVE_MODELS=/models
EXPOSE 11435

ENTRYPOINT ["visionserve"]
CMD ["serve", "--addr", ":11435"]
