# JavaScript / TypeScript

The JS SDK (`npm install visionserve`) has the same shape as the [Python SDK](python.md) with
zero runtime dependencies: it uses the built-in `fetch`, `FormData` and `Blob`, so it runs on
Node 18+ and in browsers. It is smaller, though: `predict` sends only `prompt`, `box` and
`point`. The other options are plain form fields you can [send yourself](#options-the-sdk-does-not-expose).

!!! note "How the examples on this page were run"
    With the SDK in this repository (package version 0.1.2) from TypeScript sources
    (`node --import tsx`, Node 20), against the same server as the Python page: one RTX A6000,
    CUDA, port 11680 (the runs pointed the default host there). Snippets that use
    `readFile`, `Result`, `filterBySize`, `toSVG` or `VisionServeError` import them from
    `"node:fs/promises"` / `"visionserve"`. Photos: `cat.jpg`, `dogs.jpg`, `elephant.jpg` are COCO val2017 #177015,
    #372819 and #564133 (CC BY 2.0, [credits](python.md#photos-used-on-this-page)).

## Client

```ts
import { Client } from "visionserve";

const client = new Client();                                   // http://127.0.0.1:11435
const remote = new Client("http://10.0.0.5:11435", { timeoutMs: 30_000 });
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `host` (1st) | `string` | `"http://127.0.0.1:11435"` | Base URL; trailing `/` removed. Prefer `127.0.0.1` to `localhost` ([why](index.md#connect)). |
| `opts.timeoutMs` | `number`, milliseconds | `120000` | Abort a request that takes longer; you get a `VisionServeError` without a status. |

There is no `base64_arrays` option: depth maps and embeddings always come as JSON numbers.

## `predict(model, image, opts)`

```ts
const res = await client.predict("rf-detr", "dogs.jpg");
console.log(res.task, res.device, res.durationMs.toFixed(1));
for (const d of res.detections.slice(0, 3)) console.log(d.cls, d.conf.toFixed(3), d.bbox.map((v) => +v.toFixed(1)));

const gd = await client.predict("grounding-dino", "cat.jpg", { prompt: "cat. laptop." });
for (const d of gd.detections) console.log("gd", d.cls, d.conf.toFixed(3), d.bbox.map((v) => +v.toFixed(1)));

const sam = await client.predict("mobile-sam", "cat.jpg", { box: [310, 175, 310, 195] });
const m = sam.masks[0]!;
const bits = m.toMask(640, 480);                               // Uint8Array, row-major, 1 = mask
let n = 0;
for (const b of bits) n += b;
console.log("sam", m.conf.toFixed(3), m.bbox, "pixels", n);

const pts = await client.predict("mobile-sam", "elephant.jpg", { point: [[470, 195, 1], [470, 300, 0]] });
console.log("points", pts.masks[0]!.bbox);
```

```text
detection gpu:0 22.1
person 0.937 [ 441.7, 31.7, 84.6, 131.9 ]
dog 0.923 [ 216, 227.2, 57.4, 92.5 ]
dog 0.902 [ 281.1, 108.6, 33.3, 79.2 ]
gd cat 0.901 [ 311.6, 182.5, 303.5, 181.9 ]
gd laptop 0.786 [ 7.5, 172.9, 310.4, 246.3 ]
sam 0.998 [ 320, 186, 290, 177 ] pixels 29453
points [ 335, 147, 218, 160 ]
```

| Option | Type | Meaning |
|---|---|---|
| `prompt` | `string` | Phrases separated by `.`: `"cat. laptop."`. Sent as written: unlike Python, commas are **not** turned into `.`, and there is no `"object."` default, so `grounding-dino` without a prompt is a 400. |
| `box` | `number[]` or `number[][]` | `[x, y, w, h]` in photo pixels, or a list (one mask per box). |
| `point` | `number[]` or `number[][]` | `[x, y]` or `[x, y, label]` (1 = on the object, 0 = not), or a list (one mask together). |

The option names are the server's own (`prompt`, `box`, `point`); there is no camelCase /
snake_case mapping. Their meaning is the same as in Python:
[prompt](python.md#prompt-text), [box](python.md#box-boxes), [point](python.md#point-points).

**Images**: a file path (`string`, Node only, read with `node:fs`), encoded bytes
(`Uint8Array`, a Node `Buffer`, `ArrayBuffer`) or a `Blob` / `File`. They are sent unchanged:
there is no pixel-array input, so encode raw frames to JPEG or PNG first. The server applies the
EXIF orientation of a JPEG, as described for [Python](python.md#images).

```ts
const bytes = new Uint8Array(await readFile("dogs.jpg"));
console.log("bytes", (await client.predict("rf-detr", bytes)).detections.length);
console.log("blob", (await client.predict("rf-detr", new Blob([bytes]))).detections.length);
```

```text
bytes 7
blob 7
```

### Options the SDK does not expose

Any other key in `opts` is ignored, silently:

```ts
const ignored = await client.predict("rf-detr", "dogs.jpg", { min_size: 1 } as any);
console.log("min_size via opts (ignored):", ignored.detections.length);
```

```text
min_size via opts (ignored): 7
```

The server reads every option as a form field with the name from the
[Python table](python.md#every-option-at-a-glance) (`min_size`, `roi`, `box_threshold`,
`dilate`, `method`, …), so build the form yourself and turn the answer into a `Result`:

```ts
import { Result } from "visionserve";

const form = new FormData();
form.append("model", "rf-detr");
form.append("min_size", "1");                     // % of the photo's area
form.append("roi", "200,100,260,240");            // x,y,w,h
form.append("image", new Blob([bytes]), "dogs.jpg");
const raw = await (await fetch("http://127.0.0.1:11435/api/predict", { method: "POST", body: form })).json();
console.log("raw form:", raw.detections.length, raw.detections[0]);
const res2 = Result.fromJSON(raw);                // same helpers as a predict() answer
```

```text
raw form: 3 {
  bbox: [
    226.03216871619225,
    138.73076558113098,
    41.86750799417496,
    89.99318361282349
  ],
  class: 'dog',
  conf: 0.9382620444893054
}
```

Numbers are sent as text: `String(0.35)`. Integer fields (`grid_size`, `dilate`) must be whole
numbers: `"2.5"` is read as 0, which means "default". Check `res.ok` yourself when you call
`fetch` directly; the body of an error is `{"error": "..."}`.

## Result

`Result` mirrors the Python one with camelCase names:

| Field | Type |
|---|---|
| `task`, `model`, `device` | `string` |
| `durationMs` | `number` (server time) |
| `detections` | `Detection[]`: `bbox` (`[x, y, w, h]`, photo pixels), `cls` (the JSON `class`), `conf` |
| `masks` | `Mask[]`: `rle`, `bbox`, `conf`; `toMask(width, height)` → `Uint8Array` (row-major, 1 inside), `toMask2D(width, height)` → `boolean[][]` |
| `grasps` | `Grasp[]`: `x`, `y`, `theta`, `width`, `quality`, `cls`, `conf` |
| `classifications` | `Classification[]`: `cls`, `conf` |
| `depthMap`, `depthWidth`, `depthHeight` | `number[]` (row-major, model resolution), `number`, `number` |
| `embeddings` | `number[][]` |

The JSON field `hint` is not kept. Helpers, all client side and returning a new `Result`:
`filterByConf(min, max)`, `sortByConf(desc)`, `topK(k)`, `nms(iou)`, `groupByClass()`. Module
functions: `filterBySize(result, { minSize, maxSize, imageWidth, imageHeight })` (fractions 0–1
of the area when both image sizes are given, else pixels²; also `client.filterBySize`),
`getDepthAtDetection(depthResult, detResult, mode)` and `toSVG(result, width, height)`, an SVG
string to lay over an `<img>`.

```ts
const big = filterBySize(res, { minSize: 0.01, imageWidth: 640, imageHeight: 426 });
console.log("filterBySize >=1%:", big.detections.length, "of", res.detections.length);

const depth = await client.predict("midas", "dogs.jpg");
console.log("depth", depth.depthWidth, depth.depthHeight, depth.depthMap.length, Array.isArray(depth.depthMap));

console.log(toSVG(res, 640, 426).slice(0, 160));
```

```text
filterBySize >=1%: 6 of 7
depth 256 256 65536 true
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="426"><rect x="441.66123390197754" y="31.666212290525433" width="84.62957382202148" height="131.91945
```

!!! warning "`getDepthAtDetection` and a depth model's resolution"
    A `midas` depth map is 256 × 256 whatever the photo's size, while boxes are in photo pixels.
    `getDepthAtDetection` reads the map at the box's pixel positions without scaling, so for a
    model's depth map it reads the wrong pixels (or none). Use it with a depth map that has the
    photo's size, or scale the boxes first. (The Python helper takes `image_size=` for this.)

## Errors

Non-2xx answers and network failures reject with `VisionServeError`; `e.status` is the HTTP
status, or `undefined` when no answer came back. There is no `retryAfter`: on a 503, wait about a
second (the server's `Retry-After`) and retry, as in the
[Python helper](python.md#errors-and-retries). A wrong `box` / `point` length throws a plain
`Error` before anything is sent.

```ts
try {
  await client.predict("no-such-model", "dogs.jpg");
} catch (e) {
  if (e instanceof VisionServeError) console.log("error", e.status, e.message);
}
try {
  await client.predict("grounding-dino", "cat.jpg");
} catch (e) {
  if (e instanceof VisionServeError) console.log("error", e.status, e.message);
}
try {
  await new Client("http://127.0.0.1:11435", { timeoutMs: 50 }).predict("grounding-dino", "cat.jpg", { prompt: "cat." });
} catch (e) {
  if (e instanceof VisionServeError) console.log("timeout", e.status, e.message);
}
```

```text
error 404 POST /api/predict -> 404: lifecycle: model not found: "no-such-model" is not in the registry
error 400 POST /api/predict -> 400: grounding-dino requires a text prompt, e.g. --prompt "cat. remote."
timeout undefined failed to reach VisionServe at http://127.0.0.1:11680/api/predict: This operation was aborted
```

A timeout reads "failed to reach … aborted", the same wording as a server that is down. (The
message shows the test server's port.)

## Node and the browser

In **Node** (18 or newer) everything above works, including file paths. Other calls:
`health()`, `listModels()`, `ps()`, `load(model)`, `unload(model)`. There is no `preprocess`,
`tokenize`, `explain`, templates or `infer_tensor` method; use `fetch` as above.

In a **browser**, pass a `File` from an `<input type="file">`, a `Blob` from a canvas
(`canvas.toBlob`), or bytes; a path string does not work there. The server sends no CORS headers,
so the browser only lets a page read the answers when the page comes from the **same origin** as
the API: serve your page and proxy `/api/` to VisionServe from the same host and port (a
development-server proxy, or nginx / Caddy in production), and pass that origin as the host:

```ts
// page served from https://myapp.example, which proxies /api/ to VisionServe
const client = new Client(window.location.origin);
const file = (document.querySelector("#photo") as HTMLInputElement).files![0]!;
const res = await client.predict("rf-detr", file);
```

(This snippet was not run in a browser; the same call with a `Blob` was run in Node above.)
Without a same-origin proxy, the request fails with "failed to reach VisionServe … Failed to
fetch".

## The JS command-line client

`npm install -g visionserve` (or `npx visionserve`) installs a `visionserve` command with the same
commands as the [Python one](python.md#the-python-command-line-client): `predict` (alias `run`),
`list` (`models`, `ls`), `ps`, `load`, `unload` (`rm`), `health`, with `--host`, `--timeout`
(seconds) and `--version`. `predict` takes `--prompt`, `--box`, `--point`, and `--min-size` /
`--max-size` in percent, which this CLI applies **on the client** after the answer arrives.
`--save` writes an SVG (`<stem>.js.<model>.<task>.svg`), `--save-as PATH` names it, `--compact`
and `--quiet` as in Python.

```bash
visionserve predict rf-detr dogs.jpg --min-size 1 --compact
visionserve predict grounding-dino cat.jpg --prompt "cat. laptop." --save-as cat.svg --compact
visionserve predict nope dogs.jpg; echo "exit=$?"
```

```text
predict: model=rf-detr task=detection device=gpu:0  client=117.2ms server=30.0ms  (6 detections)
{"task":"detection","model":"rf-detr","device":"gpu:0","detections":[{"bbox":[441.66123390197754,...
predict: model=grounding-dino task=open_vocab device=gpu:0  client=191.2ms server=114.8ms  (2 detections)
saved: cat.svg
{"task":"open_vocab","model":"grounding-dino","device":"gpu:0","detections":[{"bbox":[311.63137435913086,...
error: POST /api/predict -> 404: lifecycle: model not found: "nope" is not in the registry
exit=1
```
