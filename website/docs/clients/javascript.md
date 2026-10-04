# JavaScript / TypeScript

The JS SDK (`npm install visionserve`) has the same shape as the [Python SDK](python.md) with
zero runtime dependencies: it uses the built-in `fetch`, `FormData` and `Blob`, so it runs on
Node 18+ and in browsers. `predict` takes every option the Python `predict` sends, in camelCase
(`boxThreshold` is sent as the form field `box_threshold`), and normalises prompts the same way.

!!! note "How the examples on this page were run"
    With the SDK in this repository (package version 0.1.3) from TypeScript sources
    (`node --import tsx`, Node 20), against a server on the CPU (`CUDA_VISIBLE_DEVICES=`, so the
    results say `device: "cpu"`), port 11698; the runs pointed the default host there. The
    503 example used a second server started with `VISIONSERVE_MAX_QUEUE=1` on port 11699.
    Snippets that use `readFile`, `Result`, `filterBySize`, `getDepthAtDetection`, `toSVG`,
    `normalizePrompt` or `VisionServeError` import them from `"node:fs/promises"` /
    `"visionserve"`. Photos: `cat.jpg`, `dogs.jpg`, `elephant.jpg`, `food.jpg` and
    `living-room.jpg` are COCO val2017 #177015, #372819, #564133, #389381 and #29596 (CC BY 2.0,
    [credits](python.md#photos-used-on-this-page)). Timings are CPU timings.

## Client

```ts
import { Client } from "visionserve";

const client = new Client();                                   // http://127.0.0.1:11435
const remote = new Client("http://10.0.0.5:11435", { timeoutMs: 30_000 });
const arrays = new Client("http://127.0.0.1:11435", { base64Arrays: true });
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `host` (1st) | `string` | `"http://127.0.0.1:11435"` | Base URL; trailing `/` removed. Prefer `127.0.0.1` to `localhost` ([why](index.md#connect)). |
| `opts.timeoutMs` | `number`, milliseconds | `120000` | Abort a request that takes longer (the whole request, reading the answer included); you get a `VisionServeError` without a status. |
| `opts.base64Arrays` | `boolean` | `false` | Ask for depth maps and embeddings as base64 float32 ([below](#base64arrays)). |

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
detection cpu 197.5
person 0.937 [ 441.7, 31.7, 84.6, 131.9 ]
dog 0.922 [ 216, 227.2, 57.4, 92.5 ]
dog 0.903 [ 281, 108.6, 33.3, 79.2 ]
gd cat 0.904 [ 311.6, 182.6, 303.4, 181.9 ]
gd laptop 0.778 [ 7.5, 172.9, 310.4, 246.3 ]
sam 0.998 [ 320, 186, 290, 177 ] pixels 29455
points [ 335, 147, 218, 160 ]
```

### Every option

Each option is sent as the server's form field in snake_case; the meanings, defaults and the
models that read them are in the [Python option table](python.md#every-option-at-a-glance).
Leave an option out (or pass `undefined` / `null`) for the server's default.

| Option | Sent as | Type |
|---|---|---|
| [`prompt`](python.md#prompt-text) | `prompt` | `string`, normalised ([below](#prompts)) |
| [`box`](python.md#box-boxes) | `box` | `[x, y, w, h]` in photo pixels, or a list (one mask per box) |
| [`point`](python.md#point-points) | `point` | `[x, y]` or `[x, y, label]` (1 = on the object, 0 = not), or a list (one mask together) |
| [`boxThreshold`](python.md#box_threshold), [`textThreshold`](python.md#text_threshold) | `box_threshold`, `text_threshold` | `number` in (0, 1) |
| [`minSize`, `maxSize`](python.md#min_size-and-max_size) | `min_size`, `max_size` | `number`, % of the photo's area (filtered by the server) |
| [`roi`](python.md#roi-region-of-interest) | `roi` | `[x, y, w, h]`, pixels or 0–1 fractions |
| [`dilate`](python.md#dilate) | `dilate` | integer pixels |
| [`method`](python.md#method) | `method` | `string` |
| [`bgMaxArea`, `fgMinArea`](python.md#bg_max_area-and-fg_min_area) | `bg_max_area`, `fg_min_area` | `number`, % of the photo's area |
| [`gridSize`](python.md#grid_size) | `grid_size` | integer |
| [`depth`](#depth) | file part `depth` (+ `depth_dtype`, `depth_width`, `depth_height`) | `Uint16Array` or `Float32Array` (or raw bytes with `depthDtype`) |
| `depthDtype`, `depthWidth`, `depthHeight` | `depth_dtype`, `depth_width`, `depth_height` | `"uint16"` / `"float32"`; integers (default: the photo's size) |
| [`gripperMin`, `gripperMax`](python.md#gripper_min-and-gripper_max) | `gripper_min`, `gripper_max` | `number`, photo pixels |
| [`claimThreshold`](python.md#claim_threshold) | `claim_threshold` | `number` in (0, 1) |
| [`cropTemp`](python.md#crop_temp) | `crop_temp` | `number` > 0 |
| [`templateName`](#templates) | `template_name` | `string` |

Mistakes are thrown before anything is sent: a wrong `box` / `point` / `roi` length throws an
`Error`; a value that is not a finite number, a non-integer `gridSize` or `dilate` (the server
would read `2.5` as 0, the default) and an unknown key throw a `TypeError`. The last one
catches the snake_case spelling:

```ts
const big = await client.predict("rf-detr", "dogs.jpg", {
  minSize: 1,                    // % of the photo's area, applied by the server
  roi: [200, 100, 260, 240],     // x, y, w, h: run on this crop only
});
console.log("minSize + roi:", big.detections.length, big.detections[0]!.cls, big.detections[0]!.bbox.map((v) => +v.toFixed(1)));

const strict = await client.predict("grounding-dino", "cat.jpg", { prompt: "cat. laptop.", boxThreshold: 0.8 });
console.log("boxThreshold 0.8:", strict.detections.map((d) => d.cls));

const grown = await client.predict("mobile-sam", "cat.jpg", { box: [310, 175, 310, 195], dilate: 10 });
console.log("dilate 10:", grown.masks[0]!.bbox);

const grasps = await client.predict("grasp-rfdetr", "food.jpg", { gripperMin: 40, gripperMax: 80 });
console.log("grasps:", grasps.grasps.length, grasps.grasps.every((g) => g.width >= 40 && g.width <= 80));

try {
  await client.predict("rf-detr", "dogs.jpg", { min_size: 1 } as any);
} catch (e) {
  console.log((e as Error).name + ":", (e as Error).message);
}
```

```text
minSize + roi: 3 dog [ 226, 138.7, 41.9, 90 ]
boxThreshold 0.8: [ 'cat' ]
dilate 10: [ 310, 176, 310, 197 ]
grasps: 60 true
TypeError: predict(): unknown option "min_size" (did you mean "minSize"?)
```

The JS SDK has no [`max_grasps_per_object`](python.md#max_grasps_per_object): it returns every
grasp the server sent (60 here, up to 20 per object), where Python keeps the best 3 per object
by default.

For the two-stage naming models, `claimThreshold` and `cropTemp` behave as on the Python page:

```ts
const words = "cup. book. remote. lamp.";
for (const extra of [{}, { claimThreshold: 0.05 }, { cropTemp: 0.005 }]) {
  const res = await client.predict("rfdetr-dualhead-dec1", "living-room.jpg", { prompt: words, method: "dual", ...extra });
  console.log(JSON.stringify(extra), res.detections.slice(0, 3).map((d) => [d.cls, +d.conf.toFixed(3)]));
}
```

```text
{} [ [ 'cup', 0.088 ], [ 'lamp', 0.081 ], [ 'cup', 0.074 ] ]
{"claimThreshold":0.05} [ [ 'cup', 0.088 ], [ 'lamp', 0.081 ], [ 'cup', 0.119 ] ]
{"cropTemp":0.005} [ [ 'cup', 0.144 ], [ 'lamp', 0.123 ], [ 'cup', 0.118 ] ]
```

### Prompts

The prompt is normalised exactly as in Python ([details](python.md#prompt-text)): `,` and `|`
become the `.` phrase separator, a single phrase gets a trailing `.`, CLIP / SigLIP towers get the
text unchanged, and the GroundingDINO family (`grounding-dino`, `grounded-sam`, `grasp-gd`,
`gdino-siglip*`) gets `"object."` when you give no prompt. `normalizePrompt(model, prompt)`
returns what `predict` sends (`null` = no prompt field). Both SDKs run the same test cases
(`clients/testdata/normalize_prompt.json`).

```ts
const gd = await client.predict("grounding-dino", "cat.jpg");          // no prompt: "object."
console.log("no prompt:", gd.detections.map((d) => d.cls));
const two = await client.predict("grounding-dino", "cat.jpg", { prompt: "cat, laptop" });
console.log("cat, laptop:", two.detections.map((d) => d.cls));
console.log(normalizePrompt("grounding-dino", "cat, laptop"), normalizePrompt("grounding-dino", ""),
  normalizePrompt("siglip-text", "a cat, sitting"), normalizePrompt("rf-detr", undefined));
```

```text
no prompt: [ 'object' ]
cat, laptop: [ 'cat', 'laptop' ]
cat. laptop object. a cat, sitting null
```

### Depth

`depth` is a depth image from a camera, pixel-aligned with the photo; only `background` reads
it ([details](python.md#depth-an-aligned-depth-image)). Pass a `Uint16Array` (sent as `uint16`,
for example millimetres; 0 = no reading) or a `Float32Array` (sent as `float32`; ≤ 0 or NaN =
no reading), row-major, with `depthWidth` and `depthHeight`, or neither when it has the photo's
size. Raw little-endian bytes (`Uint8Array`, `ArrayBuffer`, `Blob`) are sent unchanged and need
`depthDtype`. A length that does not match `depthWidth × depthHeight` throws before sending.

```ts
const [w, h] = [640, 428];                                     // living-room.jpg
// A made-up "sensor" frame, only to show the call: a floor plane 3 m away at the top of the
// frame, closer towards the bottom, in uint16 millimetres (0 would mean "no reading").
const depthMm = new Uint16Array(w * h);
for (let y = 0; y < h; y++) depthMm.fill(3000 - 4 * y, y * w, (y + 1) * w);
const res = await client.predict("background", "living-room.jpg", {
  method: "depth",
  depth: depthMm,
  depthWidth: w,
  depthHeight: h,
});
let n = 0;
for (const b of res.masks[0]!.toMask(w, h)) n += b;
console.log(res.masks.length, "mask covering", Math.round((100 * n) / (w * h)) + "% of the photo");
```

```text
1 mask covering 100% of the photo
```

Everything in this made-up frame lies on one plane, so all of it is "surface", as in the Python
example.

### Templates

`templateName` names example images registered with `POST /api/templates`, for
`instance_detection` models. The SDK has no method for the template endpoints; use `fetch`:

```ts
// Register one example crop under a name (the SDK has no method for this endpoint).
const crop = new Blob([await readFile("dog-crop.png")]);
const form = new FormData();
form.append("name", "dog");
form.append("images", crop, "dog-crop.png");
console.log(await (await fetch(client.host + "/api/templates", { method: "POST", body: form })).json());

const res = await client.predict("owlv2_base_patch16", "dogs.jpg", { templateName: "dog" });
console.log(res.task, res.detections.length, "detections");
await fetch(client.host + "/api/templates/dog", { method: "DELETE" });
```

```text
{ count: 1, name: 'dog' }
instance_detection 10 detections
```

(`dog-crop.png` is the crop `(216, 227)`–`(274, 320)` of `dogs.jpg`, as on the
[Python page](python.md#template_name).)

### `base64Arrays`

With `new Client(host, { base64Arrays: true })` every request carries `encoding=base64`, the
server sends `depth_map` and `embeddings` as base64 float32 bytes, and the SDK decodes them.
`depthMap` and `embeddings` are still plain `number[]`, so nothing else changes; they hold the
exact float32 values (with JSON numbers you get the shortest decimal of each float32, which
`Math.fround` turns back into the same value). It is about half the bytes and cheaper to parse.

```ts
const plain = await client.predict("midas", "dogs.jpg");
const fast = await new Client(HOST, { base64Arrays: true }).predict("midas", "dogs.jpg");
console.log(fast.depthMap.length, fast.depthWidth, fast.depthHeight, Array.isArray(fast.depthMap));
console.log("same float32 values:", fast.depthMap.every((v, i) => v === Math.fround(plain.depthMap[i]!)));
```

```text
65536 256 256 true
same float32 values: true
```

### Images

A file path (`string`, Node only, read with `node:fs`), encoded bytes (`Uint8Array`, a Node
`Buffer`, `ArrayBuffer`) or a `Blob` / `File`. They are sent unchanged: there is no pixel-array
input, so encode raw frames to JPEG or PNG first. The server applies the EXIF orientation of a
JPEG, as described for [Python](python.md#images).

```ts
const bytes = new Uint8Array(await readFile("dogs.jpg"));
console.log("bytes", (await client.predict("rf-detr", bytes)).detections.length);
console.log("blob", (await client.predict("rf-detr", new Blob([bytes]))).detections.length);
```

```text
bytes 7
blob 7
```

## Result

`Result` mirrors the Python one with camelCase names:

| Field | Type |
|---|---|
| `task`, `model`, `device` | `string` |
| `hint` | `string`: the server's setup recommendation (for example about TensorRT), `""` when there is none |
| `durationMs` | `number` (server time) |
| `detections` | `Detection[]`: `bbox` (`[x, y, w, h]`, photo pixels), `cls` (the JSON `class`), `conf` |
| `masks` | `Mask[]`: `rle`, `bbox`, `conf`; `toMask(width, height)` → `Uint8Array` (row-major, 1 inside), `toMask2D(width, height)` → `boolean[][]` |
| `grasps` | `Grasp[]`: `x`, `y`, `theta`, `width`, `quality`, `cls`, `conf` |
| `classifications` | `Classification[]`: `cls`, `conf` |
| `depthMap`, `depthWidth`, `depthHeight` | `number[]` (row-major, the model's resolution), `number`, `number` |
| `embeddings` | `number[][]` |

Helpers, all client side and returning a new `Result`: `filterByConf(min, max)`,
`sortByConf(desc)`, `topK(k)`, `nms(iou)`, `groupByClass()`. Module functions:
`filterBySize(result, { minSize, maxSize, imageWidth, imageHeight })` (fractions 0–1 of the area
when both image sizes are given, else pixels²; also `client.filterBySize`),
`getDepthAtDetection(depthResult, detResult, { imageWidth, imageHeight, mode })` and
`toSVG(result, width, height)`, an SVG string to lay over an `<img>`.

`getDepthAtDetection` reads a depth model's map under each detection (or each mask when there
are none). The map has the model's resolution (256 × 256 for `midas`) whatever the photo's
size, so it needs the photo's `imageWidth` and `imageHeight` to scale the boxes onto it, and
throws without them. `mode` is `"median"` (default), `"mean"`, `"min"` or `"max"`; a box that
misses the map gives `null`. The values are relative inverse depth in `[0, 1]` (larger =
closer), not metres, as with Python's `get_depth_at_detection(..., image_size=(w, h))`.

```ts
const res = await client.predict("rf-detr", "dogs.jpg");
const big = filterBySize(res, { minSize: 0.01, imageWidth: 640, imageHeight: 426 });
console.log("filterBySize >=1%:", big.detections.length, "of", res.detections.length);

const depth = await client.predict("midas", "dogs.jpg");
console.log("depth", depth.depthWidth, depth.depthHeight, depth.depthMap.length);
const near = getDepthAtDetection(depth, res, { imageWidth: 640, imageHeight: 426 });
res.detections.slice(0, 3).forEach((d, i) => console.log(d.cls, near[i]!.toFixed(3)));

console.log(JSON.stringify(res.hint), toSVG(res, 640, 426).slice(0, 120));
```

```text
filterBySize >=1%: 6 of 7
depth 256 256 65536
person 0.309
dog 0.697
dog 0.400
"" <svg xmlns="http://www.w3.org/2000/svg" width="640" height="426"><rect x="441.66203022003174" y="31.665799677371975" wid
```

The dog in front (0.697) is closer to the camera than the person behind it (0.309).

## Errors

Non-2xx answers and network failures reject with `VisionServeError`:

- `e.status`: the HTTP status ([what each means](python.md#errors-and-retries)), or `undefined`
  when no answer came back.
- `e.retryAfter`: on a `503` (the model's queue is full), the server's `Retry-After` in seconds;
  `undefined` otherwise.
- the message: `"… timed out after N ms …"` when `timeoutMs` passed, `"failed to reach
  VisionServe at …"` when the server could not be reached.

```ts
try {
  await client.predict("no-such-model", "dogs.jpg");
} catch (e) {
  if (e instanceof VisionServeError) console.log("error", e.status, e.message);
}
try {
  await new Client(HOST, { timeoutMs: 50 }).predict("grounding-dino", "cat.jpg", { prompt: "cat." });
} catch (e) {
  if (e instanceof VisionServeError) console.log("timeout", e.status, e.message);
}
```

```text
error 404 POST /api/predict -> 404: lifecycle: model not found: "no-such-model" is not in the registry
timeout undefined POST /api/predict: timed out after 50 ms waiting for VisionServe at http://127.0.0.1:11698
```

(`HOST` is the server's URL; the message shows the test server's port.) Three requests at once
to a server that admits one request per model (`VISIONSERVE_MAX_QUEUE=1`):

```ts
const busy = new Client("http://127.0.0.1:11699");
const results = await Promise.allSettled(
  [1, 2, 3].map(() => busy.predict("grounding-dino", "cat.jpg", { prompt: "cat." })),
);
for (const r of results) {
  if (r.status === "fulfilled") console.log("ok", r.value.detections.length, "detections");
  else if (r.reason instanceof VisionServeError) {
    console.log("refused", r.reason.status, "retryAfter", r.reason.retryAfter, "-", r.reason.message);
  }
}
```

```text
ok 1 detections
refused 503 retryAfter 1 - POST /api/predict -> 503: lifecycle: model overloaded: "grounding-dino" already has 1 requests running or waiting (limit 1 per model; set VISIONSERVE_MAX_QUEUE to change it)
refused 503 retryAfter 1 - POST /api/predict -> 503: lifecycle: model overloaded: "grounding-dino" already has 1 requests running or waiting (limit 1 per model; set VISIONSERVE_MAX_QUEUE to change it)
```

A retry helper that waits as long as the server asks; with it, the same three requests all
succeed:

```ts
/** predict(), retried while the model's queue is full (503), waiting as long as the server asks. */
async function predictWithRetry(model: string, image: string, opts = {}, tries = 5): Promise<Result> {
  for (let i = 1; ; i++) {
    try {
      return await busy.predict(model, image, opts);
    } catch (e) {
      if (!(e instanceof VisionServeError) || e.status !== 503 || i === tries) throw e;
      await new Promise((r) => setTimeout(r, (e.retryAfter ?? 1) * 1000));
    }
  }
}
const all = await Promise.all([1, 2, 3].map(() => predictWithRetry("grounding-dino", "cat.jpg", { prompt: "cat." })));
console.log(all.map((r) => r.detections.length));
```

```text
[ 1, 1, 1 ]
```

## Node and the browser

In **Node** (18 or newer) everything above works, including file paths. Other calls:
`health()`, `listModels()`, `ps()`, `load(model)`, `unload(model)`. There is no `preprocess`,
`tokenize`, `explain`, templates or `infer_tensor` method; use `fetch` as in the
[templates example](#templates).

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
`--max-size` in percent, which this CLI applies **on the client** after the answer arrives; the
other options are only in the SDK (the Python CLI has a flag for each). `--save` writes an SVG
(`<stem>.js.<model>.<task>.svg`), `--save-as PATH` names it, `--compact` and `--quiet` as in
Python.

```bash
visionserve predict rf-detr dogs.jpg --min-size 1 --compact
visionserve predict grounding-dino cat.jpg --prompt "cat. laptop." --save-as cat.svg --compact
visionserve predict nope dogs.jpg; echo "exit=$?"
```

```text
{"task":"detection","model":"rf-detr","device":"cpu","detections":[{"bbox":[441.66203022003174,...
predict: model=rf-detr task=detection device=cpu  client=294.5ms server=195.5ms  (6 detections)
{"task":"open_vocab","model":"grounding-dino","device":"cpu","detections":[{"bbox":[311.6482353210449,...
predict: model=grounding-dino task=open_vocab device=cpu  client=2326.5ms server=2238.5ms  (2 detections)
saved: cat.svg
error: POST /api/predict -> 404: lifecycle: model not found: "nope" is not in the registry
exit=1
```
