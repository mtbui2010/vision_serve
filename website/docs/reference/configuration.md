# Configuration

VisionServe works with no configuration. Everything below is optional.

## Command line

```text
visionserve serve   [--addr 127.0.0.1:11435] [--models DIR] [--preload a,b]
                    [--idle-unload-seconds N] [--tensorrt]
visionserve run     MODEL IMAGE [--prompt "cat. dog."] [--box x,y,w,h] [--point x,y[,label]]
                    [--roi x,y,w,h] [--min-size %] [--max-size %] [--save | --save-as out.png]
                    [--tensorrt]
visionserve list    [--models DIR]             # installed + pullable models
visionserve pull    MODEL [--models DIR] [--force]
visionserve ps      [--addr ...]               # models loaded in a running server
visionserve rm      MODEL [--addr ...]         # unload a model from a running server
visionserve convert ...                        # convert a checkpoint to ONNX (Docker image)
visionserve version
```

| Flag | Meaning |
|---|---|
| `--addr` | Listen address. The default, `127.0.0.1:11435`, accepts connections from this machine only — the API has **no authentication**. Use `:11435` to listen on every interface, behind your own firewall or proxy. |
| `--models` | Model directory. Default: `$VISIONSERVE_MODELS`, else `~/.visionserve/models`. |
| `--preload` | Load these models at start-up instead of on first use. |
| `--idle-unload-seconds` | Override every model's idle timeout. `0` = never unload, `-1` = each manifest's own value (300 s by default). |
| `--tensorrt` | Try TensorRT before CUDA for every model whose chain uses CUDA. See [Model files and ONNX Runtime](../concepts/onnx.md#cpu-or-gpu-execution-providers). |

The source of truth for flags is `visionserve help` and
[`internal/cli`](https://github.com/mtbui2010/vision_serve/tree/main/internal/cli).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ORT_DYLIB_PATH` | — | Path to `libonnxruntime.so` (set in the Docker images). |
| `VISIONSERVE_MODELS` | `~/.visionserve/models` | Model directory (`--models` wins). |
| `VISIONSERVE_TRACE` | off | Log which execution provider each session really loaded on. |
| `VISIONSERVE_TENSORRT` | off | `1` = same as `--tensorrt`. |
| `VISIONSERVE_EP` | — | Replace every model's EP chain, e.g. `cpu` or `cuda` (CPU is still appended). Meant for benchmarking. |
| `VISIONSERVE_TRT_CACHE` | `~/.visionserve/trt-cache` | Where TensorRT keeps its built engines. |
| `VISIONSERVE_DETERMINISTIC` | off | `1` = deterministic GPU kernels: bit-identical GPU results run to run. Moves GPU outputs slightly once. |
| `VISIONSERVE_MAX_QUEUE` | auto | Requests per model, running + waiting, before new ones get `503`. Auto = `max(32, 2 × the model's sessions)`; `0` = unbounded. |
| `VISIONSERVE_POOL_THREADS` | auto | CPU threads per session of a session pool. `0` = ONNX Runtime's default. |
| `VISIONSERVE_VERIFY` | off | `strict` = cross-check every model's licence against the audited provenance ledger and enforce the SHA-256 / source pins. |
| `VISIONSERVE_CONVERT_IMAGE` | built-in | Docker image used by `visionserve convert`. |
| `VISIONSERVE_ORIGINS` | none (no CORS) | Web origins allowed to call the API from a browser, comma-separated, e.g. `http://localhost:5173,https://app.example.com` (like Ollama's `OLLAMA_ORIGINS`). The server then answers their CORS preflight and adds `Access-Control-Allow-Origin`; other origins get no CORS headers. `*` allows every origin and is logged as a warning: the API has no authentication, so any page a user opens could use it. Unset, pages can only call the server from their own origin (or through a proxy). |

## Per model: the manifest

Behaviour that belongs to one model — which files, how to prepare images, which execution
providers, thresholds, idle timeout, threads per role — lives in its `manifest.yaml`. See
[Manifest format](manifest.md).
