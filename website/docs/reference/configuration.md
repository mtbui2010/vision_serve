# Configuration

VisionServe works with no configuration. Everything below is optional.

## Command line

```text
visionserve serve   [--addr 127.0.0.1:11435] [--models DIR] [--preload a,b]
                    [--idle-unload-seconds N] [--tensorrt]
visionserve run     MODEL IMAGE [--prompt "cat. dog."] [--box x,y,w,h] [--point x,y[,label]]
                    [--box-threshold T] [--text-threshold T] [--roi x,y,w,h] [--dilate N]
                    [--min-size %] [--max-size %] [--method NAME] [--bg-max-area %]
                    [--fg-min-area %] [--grid-size N] [--gripper-min PX] [--gripper-max PX]
                    [--claim-threshold P] [--crop-temp T] [--template IMG ...]
                    [--depth FILE [--depth-dtype uint16|float32] [--depth-width W --depth-height H]]
                    [--save | --save-as out.png] [--tensorrt]
visionserve list    [--models DIR]             # installed + pullable models
visionserve pull    MODEL [--models DIR] [--force]
visionserve ps      [--addr ...]               # models loaded in a running server
visionserve rm      MODEL [--addr ...]         # unload a model from a running server
visionserve convert ...                        # convert a checkpoint to ONNX (Docker image)
visionserve inspect MODEL|FILE.onnx|DIR [--image PHOTO [--image-out F] [--prompt T]]
                    [--models DIR] [--tensorrt] [--json] [--report r.html]
                                               # model card: PASS/WARN/FAIL, files, ONNX I/O,
                                               # preprocessing, runtime (nothing loaded)
visionserve import  FILE.onnx --name N --task classification|detection|depth --license ID
                    [--labels F] [--input WxH] [--resize MODE] [--mean a,b,c --std a,b,c]
                    [--layout NCHW|NHWC] [--force] [--dry-run] [--models DIR] [--json]
                    [--report r.html]          # write a manifest for an ONNX file + install it
visionserve check   MODEL --images DIR [--labels FILE] [--reference SCRIPT.py | --checkpoint PATH]
                    [--server URL] [--models DIR] [--gpu] [--report FILE.html] [--json]
                                               # does the served model behave like training? (Docker image)
visionserve bench   MODEL [--images DIR | --size WxH] [--requests N] [--concurrency C] [--warmup N]
                    [--ep auto|cpu|cuda|tensorrt] [--server URL | --in-process] [--reload]
                    [--json] [--report FILE.html] [--prompt ... | --box ... | --point ...]
visionserve sensitivity MODEL --images DIR [--formats int8,fp16,...] [--threshold E] [--save FILE]
                    [--gpu] [--json] [--report FILE.html] [--python PY]
visionserve optimize MODEL --target jetson-orin|jetson-thor|cuda|cpu --images DIR [--labels COCO.json]
                    [--max-drop P] [--max-output-err E] [--install [--install-format F]] [--tensorrt] [--gpu]
                    [--sensitivity FILE] [--json] [--report FILE.html] [--python PY]
visionserve version
```

| Flag | Meaning |
|---|---|
| `--addr` | Listen address. The default, `127.0.0.1:11435`, accepts connections from this machine only — the API has **no authentication**. Use `:11435` to listen on every interface, behind your own firewall or proxy. |
| `--models` | Model directory. Default: `$VISIONSERVE_MODELS`, else `~/.visionserve/models`. |
| `--preload` | Load these models at start-up instead of on first use. |
| `--idle-unload-seconds` | Override every model's idle timeout. `0` = never unload, `-1` = each manifest's own value (300 s by default). |
| `--tensorrt` | Try TensorRT before CUDA for every model whose chain uses CUDA. See [Model files and ONNX Runtime](../concepts/onnx.md#cpu-or-gpu-execution-providers). |

`check` needs a running server (`--server`, default `$VISIONSERVE_HOST` or
`http://localhost:11435`) and prints a `PASS`, `WARN` or `FAIL` verdict first; it exits `0` for
PASS or WARN, `1` for FAIL and `2` for a usage or setup error. See
[Check it behaves like training](../guides/check-training.md).

`run` takes the options of `POST /api/predict` as flags named after the form fields
(`box_threshold` → `--box-threshold`; meanings in the [option table](../clients/python.md#every-option-at-a-glance)).
Two stand in for uploads: `--depth FILE` is a raw little-endian depth map (`uint16` unless
`--depth-dtype float32`; the image's size unless `--depth-width`/`--depth-height`), and
`--template IMG` (repeat it for several images) gives an `instance_detection` model its example
images directly, since there is no server to register a `template_name` with.

`inspect` and `import` share one output format: the first line is `PASS|WARN|FAIL: <reason>`,
then a short summary and the details; `--json` prints one object `{verdict, reason, summary,
details}` and nothing else; `--report` writes a self-contained HTML file. They exit 0 on PASS or
WARN, 1 on FAIL, 2 on a usage error. See [See what a model takes and returns](../guides/see-a-model.md) and
[Use a model you trained](../guides/use-your-model.md).

`bench`, `sensitivity` and `optimize` print a verdict line first (`PASS|WARN|FAIL: ...`), exit
`0` on PASS/WARN, `1` on FAIL and `2` on a usage or setup error, print one JSON object with
`--json`, and write a self-contained HTML page with `--report`. `sensitivity` and `optimize` run
the converter (Docker image, or a local Python with `--python`). See
[Measure speed](../guides/measure-speed.md) and
[Make it smaller and faster for Jetson](../guides/jetson.md).

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
| `VISIONSERVE_CONVERT_IMAGE` | built-in | Docker image used by `visionserve convert`, `check`, `sensitivity` and `optimize`. |
| `VISIONSERVE_HOST` | `http://localhost:11435` | Server that `visionserve check` (and the converter's verification) talks to (`--server` wins). |
| `VISIONSERVE_CONVERT_PYTHON` | — | A local Python with the converter installed: `sensitivity` and `optimize` run it instead of Docker (same as `--python`). |
| `VISIONSERVE_ORIGINS` | none (no CORS) | Web origins allowed to call the API from a browser, comma-separated, e.g. `http://localhost:5173,https://app.example.com` (like Ollama's `OLLAMA_ORIGINS`). The server then answers their CORS preflight and adds `Access-Control-Allow-Origin`; other origins get no CORS headers. `*` allows every origin and is logged as a warning: the API has no authentication, so any page a user opens could use it. Unset, pages can only call the server from their own origin (or through a proxy). |

## Per model: the manifest

Behaviour that belongs to one model — which files, how to prepare images, which execution
providers, thresholds, idle timeout, threads per role — lives in its `manifest.yaml`. See
[Manifest format](manifest.md).
