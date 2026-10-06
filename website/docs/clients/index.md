# Clients

VisionServe is a server: you start it once (`visionserve serve`) and then send it photos over
HTTP from your own program. This tab is about that second part. Every way of calling the server
ends up as the same `POST /api/predict` request, so pick whichever fits your program:

| | Python SDK | JavaScript / TypeScript SDK | Plain HTTP | `visionserve run` |
|---|---|---|---|---|
| What it is | `pip install visionserve` | `npm install visionserve` | `curl`, or any HTTP library | the Go binary, one shot |
| Needs a running server | yes | yes | yes | **no**: loads the model itself, answers, exits |
| Every `predict` option | **yes**, as keyword arguments | **yes**, in camelCase ([table](javascript.md#every-option)) | yes, as form fields | **yes**, as flags; `--depth FILE`, and `--template IMG` instead of `template_name` ([flags](../reference/configuration.md#command-line)) |
| Command-line client | `visionserve predict`: a flag for every option | `visionserve predict`: `prompt`, `box`, `point`, size filter | `curl` | is one |
| Prompt normalisation, `"object."` default | yes | yes (same rule) | no: sent as written | no: sent as written |
| Images you can pass | path, bytes, PIL, numpy | path (Node), bytes, `Blob` | file upload or base64 | path |
| Decodes masks / base64 arrays | yes (`Mask.to_ndarray`, `FloatArray`) | yes (`Mask.toMask`, `base64Arrays`) | you do it | prints the JSON |
| `503` `Retry-After` | `e.retry_after` | `e.retryAfter` | the header | no queue |
| Drawing | `draw()` / `Result.visualize()`: boxes, masks, grasps, depth, labels as a PIL image ([how](python.md#visualize-results)) | `toSVG()`: boxes and labels as an SVG overlay ([how](javascript.md#visualize-results)) | — | `--save` (PNG) |
| Page | [Python](python.md) | [JavaScript](javascript.md) | [Plain HTTP](http.md) | [Configuration](../reference/configuration.md) |

The Python SDK is the most complete one. Its keyword arguments have the **same names** as the
server's form fields, so its [option table](python.md#every-option-at-a-glance) is also the
reference for the other three.

`visionserve run` is handy for a quick look at one photo, but it loads the model for every call
(several seconds for GroundingDINO). For anything repeated, keep a server running and use a client.

## Install

=== "Python"

    ```bash
    pip install visionserve                # the client: standard library only, no dependencies
    pip install "visionserve[images]"      # + numpy and Pillow: arrays, PIL images, masks, drawing
    ```

    | Extra | Adds | You need it for |
    |---|---|---|
    | *(none)* | nothing | paths and bytes in, JSON-shaped results out |
    | `images` | `numpy`, `pillow` | numpy / PIL inputs, `Mask.to_ndarray`, `depth=`, `preprocess()`, `draw()` / `visualize()` |
    | `dev` | the above + `pytest` | running the SDK's tests |
    | `convert` | PyTorch, ONNX, transformers, … (several GB) | the checkpoint converter (`visionserve-convert`), not the client |

    From a checkout: `pip install -e "clients/python[images]"`. Python 3.8 or newer.

=== "JavaScript / TypeScript"

    ```bash
    npm install visionserve
    ```

    No runtime dependencies: it uses the built-in `fetch`, `FormData` and `Blob`, so it runs on
    Node 18+ and in browsers. From a checkout: `cd clients/js && npm install && npm run build`.

## Connect

Both SDKs talk to `http://…:11435` by default, the server's default port (one above Ollama's
11434, so the two can run side by side).

=== "Python"

    ```python
    from visionserve import Client

    c = Client()                                        # http://localhost:11435, 120 s timeout
    remote = Client("http://10.0.0.5:11435", timeout=30)  # another machine, shorter timeout
    print(c.health())                                   # {'status': 'ok'}
    ```

=== "JavaScript"

    ```ts
    import { Client } from "visionserve";

    const client = new Client();                        // http://127.0.0.1:11435, 120 s timeout
    const remote = new Client("http://10.0.0.5:11435", { timeoutMs: 30_000 });
    console.log(await client.health());                 // { status: 'ok' }
    ```

**Why `localhost` in Python and `127.0.0.1` in JavaScript?** The server listens on the IPv4
loopback address `127.0.0.1` by default. Python's `urllib` tries every address `localhost`
resolves to, so `localhost` works. Node 18's `fetch` may resolve `localhost` to the IPv6 address
`::1` first, where nothing is listening, and fail; the JS SDK therefore defaults to `127.0.0.1`
([`client.ts`](https://github.com/mtbui2010/vision_serve/blob/main/clients/js/src/client.ts#L224-L234)).
If you pass your own URL to the JS client, prefer `127.0.0.1` over `localhost` too.

**Another machine.** `visionserve serve` only accepts connections from the same machine unless
you start it with `--addr :11435` (the Docker images do). The API has no authentication: open it
only on a network you trust, or put it behind your own reverse proxy.

**Timeout.** The timeout covers one whole request: the upload, a model load if the model is
not in memory yet (GroundingDINO takes several seconds the first time) and the inference. The
default of 120 s is generous on purpose; lower it when you would rather fail fast, and call
`load()` at startup so the first request does not pay for the load.

**Environment variables.** Neither SDK reads any: pass the host and timeout explicitly (for
example from your own `os.environ["VISIONSERVE_URL"]`). The `VISIONSERVE_*` variables in
[Configuration](../reference/configuration.md) are read by the server.

## The two commands called `visionserve`

`pip install visionserve` and `npm install -g visionserve` each install a command-line client
named `visionserve`. It only talks to a running server (`predict`, `list`, `ps`, `load`,
`unload`, `health`). The Go binary is also called `visionserve`, and it is the one with `serve`,
`run`, `pull` and `convert`. When both are on your `PATH` the first one wins: check with
`which -a visionserve`, or call the Python one as `python -m visionserve`. Details:
[Python CLI](python.md#the-python-command-line-client).

## Where to next

- [Python](python.md): every `predict` option with real outputs, the `Result` object,
  [drawing results](python.md#visualize-results), [every helper](python.md#utilities), errors and
  retries, threads, and recipes (a folder of photos to CSV, masks to PNG, depth to numpy, CLIP
  ranking, grasping with a depth camera).
- [JavaScript / TypeScript](javascript.md): the same for Node and the browser.
- [Plain HTTP](http.md): `curl` and any other language.
- [HTTP API reference](../reference/api.md): endpoints, answer fields, status codes.
