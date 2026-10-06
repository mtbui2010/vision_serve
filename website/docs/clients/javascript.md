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
| `opts.resize` | `"auto"`, `"off"` or `number` | `"auto"` | Shrink a photo larger than the model can use before uploading it ([below](#client-side-resizing-on-by-default)), with the loopback rule on localhost; a number is a longest side in pixels. |
| `opts.jpeg` | `boolean` | `true` | Send a shrunk photo as JPEG (else PNG); a photo that is not shrunk is always sent unchanged. |
| `opts.jpegQuality` | `number` 1–100 | `90` | Its JPEG quality. |
| `opts.codec` | `ImageCodec` or `null` | the browser's, else `sharp` if installed, else none | What decodes and shrinks photos; `null` turns resizing off. |

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

The JS `predict` has no [`max_grasps_per_object`](python.md#max_grasps_per_object): it returns
every grasp the server sent (60 here, up to 20 per object), where Python keeps the best 3 per
object by default. `res.filterGrasps(3)` keeps the same 3 ([Grasps and robot
helpers](#grasps-and-robot-helpers)), and `toSVG` draws the best 3 per object unless told otherwise.

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
instance_detection 4 detections
```

(`dog-crop.png` is the crop `(215, 230)`–`(272, 319)` of `dogs.jpg`, as on the
[Python page](python.md#template_name). The four boxes are the four dogs: by default a template
match must score above 0.9; pass `boxThreshold` to change that.)

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
`Buffer`, `ArrayBuffer`) or a `Blob` / `File`. They are sent unchanged unless client-side
resizing shrinks or re-encodes them (next section); there is no pixel-array input, so encode raw
frames to JPEG or PNG first. The server applies the EXIF orientation of a
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

### Client-side resizing (on by default)

The same feature as in the [Python SDK](python.md#client-side-resizing-on-by-default), with the
same rules and defaults: with `resize: "auto"` the client fetches the per-model size hints of
`GET /api/models` once, shrinks a photo larger than its model can use (measured on the region
when `roi` is given), sends it as JPEG at `jpegQuality` 90, scales `box`, `point` and a pixel
`roi` into the sent photo, and maps every box, mask and grasp back. `res.clientResize` says what
was sent and why (`reason`: `"hint"`, `"resize=N"`, or the loopback rule's message; `null`: your
bytes, untouched); a mask decodes at your photo's size with `toMask(width, height)` either way.
Only a shrunk photo is re-encoded: models without a hint, a photo already small enough, and
requests with `depth`, `dilate`, `gripperMin` / `gripperMax` or `templateName` go out unchanged.
The [loopback rule](python.md#the-loopback-rule) applies too: with a `host` on this machine
(`localhost`, 127.0.0.0/8, `::1`; `isLoopback(host)` is exported) a photo is shrunk only when
that at least halves its sides. `resize`, `jpeg` and `jpegQuality` are also `predict` options
for one call.

The SDK has no image decoder of its own, so the work needs a codec:

| Where | Codec | Notes |
|---|---|---|
| browser, web worker | `createImageBitmap` + `OffscreenCanvas` (built in) | EXIF-rotated like the server. An un-rotated photo is decoded straight at the target size (`createImageBitmap`'s `resizeWidth` / `resizeHeight`, `resizeQuality: "high"`), the browser's equivalent of Pillow's draft decode; a rotated one is decoded whole and scaled on the canvas, because browsers need not agree on whether resizing comes before the rotation. The canvas JPEG encoder chooses its own colour subsampling (usually 4:2:0), which the [Python measurements](python.md#client-side-resizing-on-by-default) show costs an embedding model more than 4:4:4; transparent pixels become black. Use `jpeg: false` or `resize: "off"` where that matters. |
| Node with `sharp` installed by your application | `sharp` (libvips) | Lanczos-3 resize, JPEG 4:4:4; it shrinks a JPEG on load (libjpeg DCT scaling, like Pillow's draft mode) by itself. `sharp` is not a dependency of this package. |
| Node without `sharp` | none | Photos are sent exactly as given, without an error or a warning: the results are the same, the upload is larger. |

`codec: null` disables it; `codec: myCodec` plugs in your own (an object with
`probe(bytes)` and `transcode(bytes, width, height, { jpeg, quality })`). The examples on this
page ran in Node without `sharp`, so they upload the files unchanged; the size hint and the
mapping are tested with a stand-in codec in `clients/js/tests/resize.test.ts`, and the decision
rules (target size, ROI region, which hosts are loopback) run the same shared cases as Python
(`clients/testdata/client_resize.json`).

## Result

`Result` mirrors the Python one with camelCase names:

| Field | Type |
|---|---|
| `task`, `model`, `device` | `string` |
| `hint` | `string`: the server's setup recommendation (for example about TensorRT), `""` when there is none |
| `durationMs` | `number` (server time) |
| `detections` | `Detection[]`: `bbox` (`[x, y, w, h]`, photo pixels), `cls` (the JSON `class`), `conf` |
| `masks` | `Mask[]`: `rle`, `bbox`, `conf`; `toMask(width, height)` → `Uint8Array` (row-major, 1 inside), `toMask2D(width, height)` → `boolean[][]` |
| `grasps` | `Grasp[]`: `x`, `y`, `theta`, `width`, `quality`, `cls`, `conf`; `pose` (`[x, y, width, theta]`), `contacts()` (`[[x0, y0], [x1, y1]]`, the jaw points), `contactsFlat()` |
| `classifications` | `Classification[]`: `cls`, `conf` |
| `depthMap`, `depthWidth`, `depthHeight` | `number[]` (row-major, the model's resolution), `number`, `number` |
| `embeddings` | `number[][]` |
| `clientResize` | `ClientResize` or `null`: `originalWidth/Height`, `sentWidth/Height`, `jpegQuality` (`null` = PNG or not re-encoded), `resized`, `reason` (client side only) |

Helpers, all client side and returning a new `Result`: `filterByConf(min, max)`,
`sortByConf(desc)`, `topK(k)`, `nms(iou)`, `filterGrasps(k)`, `groupByClass()`. `toJSON()` gives
the server's JSON (`JSON.stringify(res)` calls it). Module functions:
`filterBySize(result, { minSize, maxSize, imageWidth, imageHeight })` (fractions 0–1 of the area
when both image sizes are given, else pixels²; also `client.filterBySize`),
`getDepthAtDetection(depthResult, detResult, { imageWidth, imageHeight, mode })`,
`toSVG(result, width, height, opts)` ([drawing](#visualize-results)) and the
[robot helpers](#grasps-and-robot-helpers).

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

console.log(JSON.stringify(res.hint), toSVG(res, 640, 426).slice(0, 87));
```

```text
filterBySize >=1%: 6 of 7
depth 256 256 65536
person 0.309
dog 0.697
dog 0.400
"" <svg xmlns="http://www.w3.org/2000/svg" width="640" height="426" viewBox="0 0 640 426">
```

The dog in front (0.697) is closer to the camera than the person behind it (0.309).

## Visualize results

`toSVG(result, width, height, opts)` returns an SVG string with what Python's
[`draw()`](python.md#visualize-results) puts on the photo, in the same style: the mask pixels as
translucent colours, boxes with `class conf%` labels, a box and `mask conf%` for each mask that no
detection labels, grasps, and the classification labels in the top-left corner. It does not
contain the photo: lay it over an `<img>`, or put the photo in as an `<image>` for one file
(below). It needs no image library and no DOM, so it runs the same in Node and in a browser.
The examples in this section and in [Utilities](#utilities) were run with the SDK 0.2.0 in Node 20
against a server on one NVIDIA RTX A6000 (CUDA), port 11835.

`width` and `height` are the photo's **own** size in pixels (the frame every coordinate of the
result is in; `probeHeader(bytes)` reads it from the file header), not the size it is shown at.
The SVG has that size and a `viewBox="0 0 width height"`, so it scales to any CSS size.

| Result | What `toSVG` draws |
|---|---|
| detections (`detection`, `open_vocab`, `instance_detection`, ...) | A box and the label `class conf%` per detection. |
| masks (`segmentation`, Grounded-SAM, `grasp`, ...) | The mask's pixels in its colour (all masks in one embedded PNG); a mask whose box equals a detection's box takes that detection's class and colour, any other also gets a box and `mask conf%`. |
| grasps (`grasp`, `grasp-gd`, ...) | The best 3 grasps per object as a gripper glyph: the closing line between the two jaw points, a jaw plate at each end, a dot at the centre and `class q0.97`, coloured red (low quality) → yellow → green (high). |
| classifications | The labels in the top-left corner on a dark band. |
| `depth`, `embed` | Nothing: an empty `<svg>` (Python's `draw()` renders depth maps; `toSVG` does not). |

| Option | Default | What it does |
|---|---|---|
| `alpha` | `0.45` | Opacity of the mask colours. |
| `masks` | `true` | `false`: no mask pixels, only their boxes. |
| `maskBoxes` | `true` | `false`: masks as colours only, without their box and label (for the dozens of masks of an automatic-mask result); a target mask keeps its red box. |
| `maxGraspsPerObject` | `3` | Grasps drawn per object, grouped as [`filterGrasps`](#grasps-and-robot-helpers) does; `null` or `0` draws every grasp. |
| `targetGrasp` | none | A grasp of the result (the same object, e.g. from `selectTargetGrasp`), drawn in red on top. |
| `targetBox` | none | A `Detection` / `Mask` of the result (matched by identity or box) or `[x, y, w, h]`: drawn in red with a line twice as thick; a box that is none of the result's items is drawn on its own. |
| `colorBy` | `"class"` | `"class"`: one colour per class name, the same as Python's (a stable hash of the name into 16 colours; two classes of one picture never share one). `"index"`: item `i` gets colour `i` of the server's 8 colours. Masks without a class use their index either way. |
| `fontSize`, `lineWidth` | scaled | Label size and line width in the photo's pixels; by default they grow with the photo's shorter side (14 and 2 on a 640 × 426 photo), as in Python. |

The label text is black or white, whichever reads better on its colour. The class colours are
those of Python's `draw()`: both SDKs run the same cases (`clients/testdata/class_colors.json`).

```ts
import { readFile } from "node:fs/promises";
import { Client, probeHeader, toSVG } from "visionserve";

const client = new Client();
const runs: Array<[string, string, object]> = [
  ["rf-detr", "dogs.jpg", {}],
  ["grounding-dino", "cat.jpg", { prompt: "cat. laptop." }],
  ["grounded-sam", "dogs.jpg", { prompt: "dog." }],
  ["mobile-sam", "cat.jpg", { box: [310, 175, 310, 195] }],
  ["efficientnet-b0", "elephant.jpg", {}],
  ["grasp-rfdetr", "food.jpg", {}],
  ["midas", "dogs.jpg", {}],
];
for (const [model, photo, opts] of runs) {
  const bytes = new Uint8Array(await readFile(photo));
  const { width, height } = probeHeader(bytes)!;              // the photo's own size
  const res = await client.predict(model, bytes, opts);
  const svg = toSVG(res, width, height);
  const count = (tag: string) => (svg.match(new RegExp("<" + tag + " ", "g")) ?? []).length;
  console.log(model.padEnd(16), res.task.padEnd(15), "image", count("image"), "rect", count("rect"), "text", count("text"), "line", count("line"));
}
```

```text
rf-detr          detection       image 0 rect 14 text 7 line 0
grounding-dino   open_vocab      image 0 rect 4 text 2 line 0
grounded-sam     open_vocab      image 1 rect 8 text 4 line 0
mobile-sam       segmentation    image 1 rect 2 text 1 line 0
efficientnet-b0  classification  image 0 rect 5 text 5 line 0
grasp-rfdetr     grasp           image 1 rect 15 text 12 line 27
midas            depth           image 0 rect 0 text 0 line 0
```

Each label is a `<rect>` band and a `<text>`, so a box counts two `<rect>`s; the one `<image>` holds
every mask. `grasp-rfdetr` returns 60 grasps, and 9 are drawn (3 objects × 3, each a glyph of 3
lines). With the wrong size, a mask cannot be placed: its run counts do not add up to
`width × height`, so it gets its box only.

The masks are an 8-bit palette PNG written by the SDK itself (a small deflate encoder, no
`node:zlib`, no canvas), so `toSVG` stays synchronous and the same everywhere. Masks are long
runs of one value, so the PNG is small: the four Grounded-SAM dogs above make a 7.4 KB SVG, 52
automatic masks of `food.jpg` 32 KB; a 4000 × 3000 picture with 10 large (synthetic) masks
took 0.2 s and 143 KB in Node.

### In Node: one SVG file with the photo

```ts
import { readFile, writeFile } from "node:fs/promises";
import { Client, probeHeader, toSVG } from "visionserve";

const client = new Client();
const bytes = new Uint8Array(await readFile("dogs.jpg"));
const { width: w, height: h } = probeHeader(bytes)!;          // the photo's size, no decoder needed
const res = await client.predict("grounded-sam", bytes, { prompt: "dog. bench." });
const overlay = toSVG(res, w, h);                              // "<svg ... viewBox=...>masks, boxes, labels</svg>"
console.log(w, h, res.detections.map((d) => d.cls), (overlay.length / 1024).toFixed(1), "KB");

// One standalone SVG file: the photo as an <image>, the overlay on top. Open it in a browser.
const photo = `<image href="data:image/jpeg;base64,${Buffer.from(bytes).toString("base64")}" width="${w}" height="${h}"/>`;
await writeFile("dogs-drawn.svg", overlay.replace(">", ">" + photo));
```

```text
640 426 [ 'dog', 'dog', 'dog', 'dog', 'bench', 'bench' ] 7.4 KB
```

`dogs-drawn.svg` opens in any browser: rendered in headless Chrome it shows the four dogs and the
two benches filled with their class colour, each with one labelled box, as Python's `draw()`
picture of the same call. To get a PNG or JPEG, convert it with a tool such as `rsvg-convert`, or
render it in a headless browser.

### Grasps and a chosen target

```ts
import { readFile, writeFile } from "node:fs/promises";
import { Client, probeHeader, selectTargetGrasp, selectTargetObject, toSVG } from "visionserve";

const client = new Client();
const bytes = new Uint8Array(await readFile("food.jpg"));
const { width: w, height: h } = probeHeader(bytes)!;
const g = await client.predict("grasp-rfdetr", bytes);         // every grasp the server sends
console.log(g.detections.length, "objects,", g.masks.length, "masks,", g.grasps.length, "grasps");
const obj = selectTargetObject(g, { cls: "broccoli" })!;
const best = selectTargetGrasp(g.grasps, { cls: "broccoli" })!;
console.log(obj.bbox.map(Math.round), "| grasp", best.pose.map((v) => +v.toFixed(2)));
const svg = toSVG(g, w, h, { targetBox: obj, targetGrasp: best, maxGraspsPerObject: 1 });
const photo = `<image href="data:image/jpeg;base64,${Buffer.from(bytes).toString("base64")}" width="${w}" height="${h}"/>`;
await writeFile("food-grasp.svg", svg.replace(">", ">" + photo));
console.log((svg.match(/<g>/g) ?? []).length, "grasps drawn");
```

```text
3 objects, 3 masks, 60 grasps
[ 373, 278, 105, 129 ] | grasp [ 424, 332.5, 76.53, 3.02 ]
3 grasps drawn
```

The same object and grasp as the [Python example](python.md#grasps). In the picture the broccoli
box and its grasp are red, the bowl and carrot keep their colours with one grasp each. A grasp is
counted with the detection it was planned on (same class and confidence), not with the box its
centre falls in: the bowl's best grasp lies inside the carrot's box and is still drawn as the
bowl's. `targetGrasp` is matched by identity and must be one of the grasps drawn.

### In a browser: an overlay on the `<img>`

```html
<div style="position: relative; display: inline-block">
  <img id="photo" src="dogs.jpg" style="display: block; width: 480px">
  <div id="overlay" style="position: absolute; inset: 0"></div>
</div>
```

```ts
const img = document.querySelector("#photo") as HTMLImageElement;
const blob = await (await fetch(img.src)).blob();
const res = await client.predict("grounded-sam", blob, { prompt: "dog." });
// The photo's own pixels; the SVG's viewBox does the scaling to the 480-px <img>.
document.querySelector("#overlay")!.innerHTML = toSVG(res, img.naturalWidth, img.naturalHeight)
  .replace("<svg ", '<svg style="width: 100%; height: 100%" ');
```

(The page itself was not run. The overlay part was checked in headless Chrome: a 640-pixel
Grounded-SAM result laid over the photo shown 480 pixels wide puts the masks and boxes on the
dogs.)

### Many masks

Automatic masks (no prompt) can be dozens of overlapping regions. Their boxes and labels then
hide the photo; `maskBoxes: false` keeps only the colours:

```ts
import { Client, toSVG } from "visionserve";

const client = new Client();
const auto = await client.predict("mobile-sam", "food.jpg");   // no prompt: automatic masks
const [w, h] = [640, 543];
const boxes = toSVG(auto, w, h);                               // a box + label per mask
const fills = toSVG(auto, w, h, { maskBoxes: false, alpha: 0.6 });  // colour only
console.log(auto.masks.length, "masks:", (boxes.length / 1024).toFixed(1), "KB with boxes,", (fills.length / 1024).toFixed(1), "KB without");
```

```text
52 masks: 48.3 KB with boxes, 32.0 KB without
```

### Masks on a canvas

To paint masks your own way, `Mask.toMask(width, height)` gives one byte per pixel (1 = inside),
row by row. Mix a colour into the pixels of a canvas with it. The function below works on any
RGBA buffer; here it ran in Node on a white buffer, and in a browser you pass
`ctx.getImageData(0, 0, w, h).data`:

```ts
import { Client, Result } from "visionserve";

/** Tint every mask of `res` into an RGBA pixel buffer (row-major, 4 bytes per pixel). */
function paintMasks(rgba: Uint8ClampedArray, res: Result, w: number, h: number, alpha = 0.5) {
  const colours = [[255, 59, 59], [255, 165, 0], [50, 205, 50], [0, 191, 255]];
  res.masks.forEach((m, i) => {
    const [r, g, b] = colours[i % colours.length]!;
    const bits = m.toMask(w, h);                          // 1 inside, row-major
    for (let k = 0; k < bits.length; k++) {
      if (!bits[k]) continue;
      rgba[4 * k] = rgba[4 * k]! * (1 - alpha) + r! * alpha;
      rgba[4 * k + 1] = rgba[4 * k + 1]! * (1 - alpha) + g! * alpha;
      rgba[4 * k + 2] = rgba[4 * k + 2]! * (1 - alpha) + b! * alpha;
    }
  });
}

const client = new Client();
const res = await client.predict("grounded-sam", "dogs.jpg", { prompt: "dog." });
const [w, h] = [640, 426];
const rgba = new Uint8ClampedArray(w * h * 4).fill(255);  // stand-in for a canvas's pixels
paintMasks(rgba, res, w, h);
let tinted = 0;
for (let k = 0; k < w * h; k++) if (rgba[4 * k + 1] !== 255 || rgba[4 * k] !== 255) tinted++;
console.log(res.masks.length, "masks,", tinted, "pixels tinted");
```

```text
4 masks, 9217 pixels tinted
```

In a browser:

```ts
const canvas = document.querySelector("canvas")!;
const ctx = canvas.getContext("2d")!;
ctx.drawImage(img, 0, 0);                                     // canvas.width/height = the photo's size
const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
paintMasks(pixels.data, res, canvas.width, canvas.height);
ctx.putImageData(pixels, 0, 0);
```

(Not run in a browser either; the `paintMasks` part is the code run above.)

## Utilities

These helpers work on results in your program and never call the server.

| Where | Helper | What it does | Use it when |
|---|---|---|---|
| `Result` | `filterByConf(minConf = 0, maxConf = 1)` | Keeps detections, masks and classifications whose `conf` is in the range. | Hiding weak guesses. |
| module | `filterBySize(result, { minSize, maxSize, imageWidth, imageHeight })` (also `client.filterBySize`) | Keeps detections and masks by box area: **fractions** (0–1) of the photo when both sizes are given, else pixels². `0` = no limit. | Dropping tiny or huge boxes. |
| `Result` | `sortByConf(descending = true)`, `topK(k)` | Orders by `conf`; keeps the `k` best of each list. | Only the best few. |
| `Result` | `nms(iouThreshold = 0.5)` | Greedy non-maximum suppression on detections (masks are kept). | Merged results with overlapping boxes. |
| `Result` | `filterGrasps(maxPerObject)` | Keeps the best `maxPerObject` grasps per object: a grasp goes with the detection it was planned on (same class and confidence), a class-agnostic one with the smallest box containing its centre. `null` / `0` keeps all. | Python's `max_grasps_per_object` (default 3). |
| `Result` | `groupByClass()` | `{ label: Result }` with only that class's detections, masks and grasps; a mask joins the detection with the same box, other masks go under `""`; a class-agnostic grasp joins the detection box it lies in. Classifications, depth map and embeddings are per photo, so every group has them empty. Same as Python's `group_by_class()`. | Handling each class apart. |
| `Result` | `toJSON(opts)`, `Result.fromJSON(obj)` | The server's JSON (`class`, `depth_map`, `duration_ms`, ...) and back: `Result.fromJSON(JSON.parse(JSON.stringify(res)))` deep-equals `res`. `{ encoding: "base64" }` writes depth maps and embeddings as base64 float32. | Saving and reloading results. |
| `Grasp` | `pose`, `contacts()`, `contactsFlat()` | `[x, y, width, theta]`; the two jaw points `[[x0, y0], [x1, y1]]` (centre ∓ width / 2 along θ) or `[x0, y0, x1, y1]`. | Sending a grasp to a robot. |
| `Mask` | `toMask(width, height)`, `toMask2D(width, height)` | The mask as a row-major `Uint8Array` (1 = inside), or `boolean[height][width]`. | Area, union, painting pixels. |
| module | `getDepthAtDetection(depth, detResult, { imageWidth, imageHeight, mode, depthScale })` | The depth under each box: a depth model's value (relative 0–1, larger = closer), or metres from a camera's depth image. | "Which object is nearer?" |
| module | `backproject(u, v, z, K)`, `cameraDistance(u, v, z, K)` | A pixel at depth `z` as a 3D point `[X, Y, Z]` in the camera frame, and its distance from the camera. `K` is `{ fx, fy, cx, cy }` or `[fx, fy, cx, cy]`. | Pixels to robot coordinates. |
| module | `objectDistances(depth, detResult, K, { mode, depthScale })`, `graspDistances(depth, grasps, K, { window, depthScale })` | Camera → object / grasp distance in metres from a metric depth image. | Reach checks. |
| module | `selectTargetObject(result, opts)`, `selectTargetGrasp(grasps, opts)` (and `…Index` versions returning `[item, index]`) | The best object / grasp by class, confidence, size, closeness to a point, gripper opening and camera distance. | Picking what the robot goes for. |
| module | `toSVG(result, width, height, opts)` | Masks, boxes, grasps and labels as an SVG string ([above](#visualize-results)). | Drawing. |
| module | `probeHeader(bytes)` | `{ width, height, isJpeg, orientation }` from a JPEG or PNG header, EXIF rotation applied, or `null`. | The photo's size without an image library. |
| module | `targetSize(width, height, { maxSide, maxShortSide, region })` | The size a photo would be shrunk to. | Planning uploads. |
| `Client` | `usefulSide(model)` | `[maxSide, maxShortSide]`: the model's size hint, `null` = none. | Knowing how big a photo is worth sending. |
| module | `ClientResize` (`res.clientResize`) | What was uploaded: `originalWidth/Height`, `sentWidth/Height`, `jpegQuality`, `reason`, `resized`, `toSentX/Y()`, `toOriginalX/Y()`. | Checking the client-side resize. |
| module | `normalizePrompt(model, prompt)`, `isLoopback(host)` | The prompt `predict` sends; whether a host is this machine. | Debugging prompts and resizing. |

The grasp and robot helpers are a port of Python's (`visionserve/postprocess.py`): the same
maths and defaults, checked by the same cases in both SDKs (`clients/testdata/postprocess_sync.json`,
generated from the Python code).

### Filters and depth

```ts
import { Client, filterBySize, getDepthAtDetection } from "visionserve";

const client = new Client();
const res = await client.predict("rf-detr", "dogs.jpg");
const [W, H] = [640, 426];
console.log(res.detections.length, "detections");
console.log(res.filterByConf(0.8).detections.map((d) => d.cls));
console.log(filterBySize(res, { minSize: 0.02, imageWidth: W, imageHeight: H }).detections.map((d) => d.cls));
console.log(filterBySize(res, { maxSize: 3000 }).detections.length, "boxes of at most 3000 px²");
console.log(res.topK(3).detections.map((d) => +d.conf.toFixed(2)));
console.log(Object.fromEntries(Object.entries(res.groupByClass()).map(([k, v]) => [k, v.detections.length])));
console.log(res.nms(0.5).detections.length, "after nms(0.5)");

const depth = await client.predict("midas", "dogs.jpg");
const near = getDepthAtDetection(depth, res, { imageWidth: W, imageHeight: H, mode: "median" });
console.log(near.slice(0, 4).map((v) => +v!.toFixed(3)));
```

```text
7 detections
[ 'person', 'dog', 'dog', 'person', 'dog', 'dog' ]
[ 'person', 'person', 'bench' ]
1 boxes of at most 3000 px²
[ 0.94, 0.92, 0.9 ]
{ person: 2, dog: 4, bench: 1 }
7 after nms(0.5)
[ 0.309, 0.697, 0.401, 0.301 ]
```

The filters return a new `Result`, so they chain (`res.filterByConf(0.5).topK(2)`).

### Masks: area and union

```ts
import { Client } from "visionserve";

const client = new Client();
const res = await client.predict("grounded-sam", "dogs.jpg", { prompt: "dog." });
const [W, H] = [640, 426];
const masks = res.masks.map((m) => m.toMask(W, H));          // Uint8Array, 1 inside, row-major
const area = (bits: Uint8Array) => bits.reduce((s, b) => s + b, 0);
console.log("areas:", masks.map(area));
const union = new Uint8Array(W * H);
for (const bits of masks) bits.forEach((b, k) => { if (b) union[k] = 1; });
console.log("union:", area(union), "px =", ((100 * area(union)) / (W * H)).toFixed(2), "% of the photo");
const grid = res.masks[0]!.toMask2D(W, H);                    // boolean[H][W]
console.log(grid.length, grid[0]!.length, grid[150]![295]);
```

```text
areas: [ 1297, 2417, 3380, 2123 ]
union: 9217 px = 3.38 % of the photo
426 640 true
```

### Sizes and the client-side resize

```ts
import { Client, ClientResize, targetSize } from "visionserve";

const client = new Client();
console.log(await client.usefulSide("grounding-dino"), await client.usefulSide("mobile-sam"));
console.log(targetSize(4000, 3000, { maxSide: 1333 }), targetSize(4000, 3000, { maxShortSide: 800 }));
const res = await client.predict("grounding-dino", "dogs.jpg", { prompt: "dog.", resize: 320 });
console.log(res.clientResize);              // null: Node without sharp has no codec, sent as given
const cr = new ClientResize(640, 426, 320, 213, 90, "resize=320");   // what a codec would report
console.log(cr.resized, cr.toSentX(640), cr.toOriginalX(160), cr.toOriginalY(100));
```

```text
[ null, 1600 ] [ null, null ]
[ 1333, 1000 ] [ 1067, 800 ]
null
true 320 320 200
```

### Grasps and robot helpers

A metric depth image from an RGB-D camera, aligned with the photo, is `{ data, width, height }`:
an integer array (`Uint16Array`) is read as millimetres, a float array (`Float32Array`,
`number[]`) as metres, `0` = no reading; `depthScale` (metres per unit) overrides that. A depth
model's answer (`midas`) has no scale, so the distance helpers refuse it.

```ts
import { Client, graspDistances, objectDistances, selectTargetGrasp, selectTargetObject } from "visionserve";

const client = new Client();
const g = await client.predict("grasp-rfdetr", "food.jpg");
const top = g.filterGrasps(3);                                  // the best 3 per object, as Python's default
console.log(g.grasps.length, "grasps ->", top.grasps.length, "after filterGrasps(3)");
const best = selectTargetGrasp(top.grasps)!;                    // highest quality
console.log(best.cls, "pose [x, y, width, theta]:", best.pose.map((v) => +v.toFixed(2)));
console.log("jaw contacts:", best.contacts().map((p) => p.map((v) => +v.toFixed(1))));
const fit = selectTargetGrasp(top.grasps, { gripperMin: 60, gripperMax: 80 })!;   // what the gripper can open to
console.log("for a 60-80 px opening:", fit.cls, fit.width.toFixed(1), "q", fit.quality.toFixed(2));
console.log(Object.fromEntries(Object.entries(top.groupByClass()).map(([k, r]) => [k, r.grasps.length])));

// Metric depth from an RGB-D camera, aligned with the photo: here a made-up tilted table,
// 0.9 m away at the top of the frame and 0.6 m at the bottom, in uint16 millimetres.
const [w, h] = [640, 543];
const depth = { data: new Uint16Array(w * h), width: w, height: h };
for (let y = 0; y < h; y++) depth.data.fill(Math.round(900 - (300 * y) / h), y * w, (y + 1) * w);
const K = { fx: 600, fy: 600, cx: 320, cy: 271.5 };            // or [fx, fy, cx, cy]
console.log("object distances (m):", objectDistances(depth, g, K).map((v) => +v!.toFixed(3)));
console.log("grasp distances (m):", graspDistances(depth, top.grasps.slice(0, 3), K).map((v) => +v!.toFixed(3)));
const near = selectTargetObject(g, { depth, intrinsics: K, targetDistance: 0.7 })!;
console.log("object nearest 0.7 m:", "cls" in near ? near.cls : "mask", near.bbox.map(Math.round));
```

```text
60 grasps -> 9 after filterGrasps(3)
bowl pose [x, y, width, theta]: [ 274, 228.5, 11.18, 0.46 ]
jaw contacts: [ [ 269, 226 ], [ 279, 231 ] ]
for a 60-80 px opening: broccoli 76.5 q 0.97
{ broccoli: 3, carrot: 3, bowl: 3 }
object distances (m): [ 0.727, 0.818, 0.81 ]
grasp distances (m): [ 0.731, 0.731, 0.732 ]
object nearest 0.7 m: broccoli [ 373, 278, 105, 129 ]
```

The depth frame is made up, only to show the calls. `selectTargetObject` scores the candidates on
`conf`, `area`, `near` (closeness to `nearPoint`, a pixel or `"center"`) and `distance`
(closeness of the camera distance to `targetDistance`); without `weights` it uses the most
specific one given (distance, else near, else conf), with `weights: { conf: 1, area: 1 }` a mix.
`selectTargetGrasp` does the same with `quality`, `near` (`targetPoint`), `distance` and `width`
(an opening in the middle of `gripperMin`–`gripperMax`). Both return the result's own object, or
`null` when nothing passes `cls`, `minConf` or the gripper limits; the defaults are Python's
([`select_target_object`](python.md#utilities)).

### JSON

`JSON.stringify(res)` writes the server's JSON (`class`, `duration_ms`, empty fields left out),
so a saved result reads back with `Result.fromJSON`, and the file is the same as one saved from a
plain HTTP call. A photo the client shrank keeps its `clientResize` as an extra `client_resize`
field.

```ts
import { deepStrictEqual } from "node:assert/strict";
import { Client, Result } from "visionserve";

const client = new Client();
const res = await client.predict("grounded-sam", "dogs.jpg", { prompt: "dog." });
const text = JSON.stringify(res);                              // the server's wire format
const obj = JSON.parse(text);
console.log(Object.keys(obj), Object.keys(obj.detections[0]));
deepStrictEqual(Result.fromJSON(obj), res);                    // reads back the same
console.log("round trip ok,", text.length, "bytes");
const arrays = await client.predict("midas", "dogs.jpg");
console.log(Object.keys(arrays.toJSON({ encoding: "base64" })));   // depth as base64 float32
```

```text
[ 'task', 'model', 'device', 'detections', 'masks', 'duration_ms' ] [ 'bbox', 'class', 'conf' ]
round trip ok, 2419 bytes
[
  'task',
  'model',
  'device',
  'depth_width',
  'depth_height',
  'duration_ms',
  'depth_map_base64'
]
```

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
(`canvas.toBlob`), or bytes; a path string does not work there. By default the server sends no
CORS headers, so the browser only lets a page read the answers when the page comes from the
**same origin** as the API. Two ways to call it from a page:

- **Allow your page's origin** on the server: `VISIONSERVE_ORIGINS=http://localhost:5173
  visionserve serve` (a comma-separated list, like Ollama's `OLLAMA_ORIGINS`; see
  [configuration](../reference/configuration.md#environment-variables)). The page then calls
  `new Client("http://127.0.0.1:11435")` directly. `*` allows every origin, so any web page a
  user opens could use the API, which has no authentication: list your origins instead.
- **Proxy** `/api/` to VisionServe from your page's own host and port (a development-server
  proxy, or nginx / Caddy in production), and pass that origin as the host:

```ts
// page served from https://myapp.example, which proxies /api/ to VisionServe
const client = new Client(window.location.origin);
const file = (document.querySelector("#photo") as HTMLInputElement).files![0]!;
const res = await client.predict("rf-detr", file);
```

(This snippet was not run in a browser; the same call with a `Blob` was run in Node above.)
Without either, the request fails with "failed to reach VisionServe … Failed to fetch".

## The JS command-line client

`npm install -g visionserve` (or `npx visionserve`) installs a `visionserve` command with the same
commands as the [Python one](python.md#the-python-command-line-client): `predict` (alias `run`),
`list` (`models`, `ls`), `ps`, `load`, `unload` (`rm`), `health`, with `--host`, `--timeout`
(seconds) and `--version`. `predict` takes `--prompt`, `--box`, `--point`, `--min-size` /
`--max-size` in percent, which this CLI applies **on the client** after the answer arrives, and
`--resize auto|off|N`, `--no-jpeg`, `--jpeg-quality Q` (resizing needs `sharp` in Node); the
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
