# Measure speed

## When you need this

You want to know how fast a model answers on a given machine: per photo (latency), per second
(throughput), how long it takes to start, how much memory it uses, and whether it really ran on
the GPU. Run it on the machine you deploy to. Numbers from another machine say little about it.

## The command

```console
$ visionserve bench my-detector --images ./photos --ep cuda
```

It loads the model in this process, like `visionserve run` (no server needed), sends a few
warm-up photos, then times 50 requests. `--images` is a folder of photos like your real ones;
without it, `bench` uses a synthetic 640 × 480 image. `--ep cuda` says you expect the GPU, so a
silent fall-back to the CPU becomes a `WARN`.

## Reading the result

```console
$ visionserve bench my-detector --images ./photos --ep cuda
bench: loading my-detector (in-process) ...
bench: 50 requests at concurrency 1 ...
PASS: p50 15.0 ms, 63.7 req/s on gpu:0

  device (EP used)         gpu:0
  cold load                4.69 s
  first request            963 ms
  latency p50 / p95 / p99  15.0 / 25.3 / 33.4 ms
  server-side p50 / p95    15.0 / 25.3 ms
  throughput               63.7 req/s at concurrency 1
  errors                   0 / 50
  process memory (RSS)     831 MB / 834 MB peak
  GPU memory               540 MB

Findings
  INFO  Numbers are for THIS machine, power mode and load; a shared or busy GPU inflates and jitters them.
...
```

| Line | In plain words |
|---|---|
| **device** | Where it really ran: `gpu:0` (CUDA), `gpu:0+trt` (TensorRT) or `cpu`. |
| **cold load** | Time to load the model: paid once when the server starts, or on the first request after the model was unloaded. |
| **first request** | The first photo is slower: the GPU picks its kernels and allocates memory. |
| **latency p50 / p95 / p99** | Time per photo. *p50* is the median (half the photos are faster); *p95* means 95 % are faster. Look at p95 if you have a deadline. |
| **throughput** | Photos per second, one at a time. Raise `--concurrency` to send several at once. |
| **memory** | RAM and GPU memory of this process. On a Jetson, where the GPU shares the RAM, `bench` reads `tegrastats` instead. |

The same model on the CPU of the same machine: `PASS: p50 263 ms, 3.25 req/s on cpu`, about 17
times slower.

### Through the server, as your clients see it

`--server URL` times whole HTTP requests to a running server, upload included:

```console
$ visionserve bench my-detector --images ./photos --server http://127.0.0.1:11770 \
      --concurrency 2
bench: loading my-detector (server http://127.0.0.1:11770) ...
bench: 50 requests at concurrency 2 ...
PASS: p50 30.8 ms, 60.6 req/s on gpu:0

  device (EP used)         gpu:0
  cold load                not measured
  first request            38.4 ms
  latency p50 / p95 / p99  30.8 / 43.8 / 46.3 ms
  server-side p50 / p95    14.4 / 26.8 ms
  throughput               60.6 req/s at concurrency 2
...
```

*Server-side* is the model's own time (14.4 ms); the rest of the 30.8 ms is the upload, decoding
the photo, and waiting behind the other request in flight. `--reload` unloads the model first,
to measure the cold load through the server too.

!!! tip "Large photos: the client can shrink them for you"
    The model shrinks every photo to its own input size (384 × 384 here) anyway, so uploading a
    12-megapixel photo mostly sends pixels the model throws away. The Python (0.2.0+) and
    JavaScript (0.1.4+) SDKs shrink a large photo before uploading it, by default, and map the
    answer back to your photo's pixels; see
    [Client-side resizing](../clients/python.md#client-side-resizing-on-by-default). `bench`
    itself sends your photos as they are, so it measures the upload of the size you give it.

## If it says WARN or FAIL

| Message | What to do |
|---|---|
| `WARN: CUDA was requested but the model ran on the CPU` | Your ONNX Runtime library has no CUDA support, or cannot load it. Point `ORT_DYLIB_PATH` at a GPU build (its folder must hold `libonnxruntime_providers_cuda.so`), make sure cuDNN 9 is installed, and run again with `VISIONSERVE_TRACE=1` to see why CUDA did not load. From source, `source scripts/gpu-env.sh` finds a working library for you. |
| `WARN: a GPU is present but the model ran on the CPU` | The same, when you did not pass `--ep`. |
| `WARN: TensorRT was requested but CUDA ran` | `libnvinfer.so.10` (TensorRT) is missing, or the engine failed to build. TensorRT is optional: CUDA is the default. |
| `WARN: requests ran on different devices` | Some requests ran on the GPU and some on the CPU. Run again with `VISIONSERVE_TRACE=1` to see which session fell back, and why. |
| `FAIL: N of M requests failed (first: ...)` | Read the first error. A prompt model (SAM, GroundingDINO) may need `--prompt`, `--box` or `--point`. |
| Next steps: `p99 is more than twice p50` | Something else is using the GPU or CPU, or the first request of each new photo size is slow. Raise `--warmup`, or use photos at your camera's resolution. |

On a Jetson, set the power mode first and write down which one you measured (`sudo nvpmodel -q`;
`sudo jetson_clocks` for stable clocks).

## Want the details?

- More measurements (three models, CPU vs GPU vs server), TensorRT numbers and every line of the
  output: [Jetson details, section 1](edge.md#1-bench-how-fast-is-it-on-this-machine).
- Making the model itself faster: [Make it smaller and faster for Jetson](jetson.md).
- `--json` prints one JSON object for scripts; `--report bench.html` writes a page with the
  latency histogram.

<small>Run on 5 October 2026 on an RTX A6000 shared with other jobs (CUDA, ONNX Runtime 1.26) and
its 48-core CPU, with the `visionserve` binary; 8 COCO val2017 photos (CC BY 2.0).</small>
