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
| `grasps` | `Grasp[]`: `x`, `y`, `theta`, `width`, `quality`, `cls`, `conf` |
| `classifications` | `Classification[]`: `cls`, `conf` |
| `depthMap`, `depthWidth`, `depthHeight` | `number[]` (row-major, the model's resolution), `number`, `number` |
| `embeddings` | `number[][]` |
| `clientResize` | `ClientResize` or `null`: `originalWidth/Height`, `sentWidth/Height`, `jpegQuality` (`null` = PNG or not re-encoded), `resized`, `reason` (client side only) |

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

## Visualize results

The JS SDK has one drawing helper: `toSVG(result, width, height)`. It returns an SVG string
with boxes and labels, to lay over the photo. It does not decode or change the photo, so it
cannot paint masks; you do that yourself (below). The examples in this section and in
[Utilities](#utilities) were run in Node 20 against a server on one NVIDIA RTX A6000 (CUDA),
port 11820, with the SDK in this repository.

| Result | What `toSVG` draws |
|---|---|
| `task` `detection` or `open_vocab` (Grounded-SAM too) | A box and the label `class conf%` per detection. |
| `task` `segmentation` | The **box** of each mask and `mask conf%`; not the mask's pixels. |
| `task` `classification` | The labels in the top-left corner. |
| `grasp`, `instance_detection`, `depth`, `embed` | Nothing: an empty `<svg>`. |

The boxes are written in the photo's pixels and the SVG has no `viewBox`, so pass the photo's
**own** size as `width` and `height` (not the size it is shown at).

```ts
import { Client, toSVG } from "visionserve";

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
  const res = await client.predict(model, photo, opts);
  const svg = toSVG(res, 640, 480);
  const count = (tag: string) => (svg.match(new RegExp("<" + tag, "g")) ?? []).length;
  console.log(model.padEnd(16), res.task.padEnd(15), "rect", count("rect"), "text", count("text"));
}
```

```text
rf-detr          detection       rect 7 text 7
grounding-dino   open_vocab      rect 2 text 2
grounded-sam     open_vocab      rect 4 text 4
mobile-sam       segmentation    rect 1 text 1
efficientnet-b0  classification  rect 0 text 5
grasp-rfdetr     grasp           rect 0 text 0
midas            depth           rect 0 text 0
```

(`grasp-rfdetr` returns 3 detections, 3 masks and 60 grasps here, and `toSVG` draws none of
them, because its task is `grasp`.)

### In Node: one SVG file with the photo

`probeHeader(bytes)` reads a JPEG's or PNG's size from its header (EXIF rotation applied), so
you need no image library:

```ts
import { readFile, writeFile } from "node:fs/promises";
import { Client, probeHeader, toSVG } from "visionserve";

const client = new Client();
const bytes = new Uint8Array(await readFile("dogs.jpg"));
const { width: w, height: h } = probeHeader(bytes)!;          // the photo's size, no decoder needed
const res = await client.predict("rf-detr", bytes);
const overlay = toSVG(res, w, h);                              // "<svg ...>boxes + labels</svg>"
console.log(w, h, (overlay.match(/<rect/g) ?? []).length, "boxes");

// One standalone SVG file: the photo as an <image>, the overlay on top. Open it in a browser.
const photo = `<image href="data:image/jpeg;base64,${Buffer.from(bytes).toString("base64")}" width="${w}" height="${h}"/>`;
await writeFile("dogs-boxes.svg", overlay.replace(">", ">" + photo));
```

```text
640 426 7 boxes
```

`dogs-boxes.svg` opens in any browser. To get a PNG or JPEG, convert it with a tool such as
`rsvg-convert`, or render it in a headless browser.

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
const res = await client.predict("rf-detr", blob);
const [w, h] = [img.naturalWidth, img.naturalHeight];        // the photo's own pixels
// viewBox + 100 % size: the overlay follows the <img> however large it is shown.
document.querySelector("#overlay")!.innerHTML = toSVG(res, w, h)
  .replace("<svg ", `<svg viewBox="0 0 ${w} ${h}" style="width: 100%; height: 100%" `);
```

(The page itself was not run. The overlay part was checked in headless Chrome: with the
`viewBox`, the boxes of a 640-pixel result sit on the dogs of the photo shown 480 pixels wide.)

### Masks on a canvas

`Mask.toMask(width, height)` gives one byte per pixel (1 = inside), row by row. Mix a colour into
the pixels of a canvas with it. The function below works on any RGBA buffer; here it ran in Node
on a white buffer, and in a browser you pass `ctx.getImageData(0, 0, w, h).data`:

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

### Draw it yourself: masks, boxes and grasps as SVG

For grasps, or your own style, write the SVG elements yourself. A mask becomes one 1-pixel-high
`<rect>` per run of pixels in a row; a grasp is the line between its two jaws:

```ts
import { readFile, writeFile } from "node:fs/promises";
import { Client, probeHeader } from "visionserve";

const client = new Client();
const bytes = new Uint8Array(await readFile("food.jpg"));
const { width: w, height: h } = probeHeader(bytes)!;
const res = await client.predict("grasp-rfdetr", bytes);
const parts: string[] = [`<image href="data:image/jpeg;base64,${Buffer.from(bytes).toString("base64")}" width="${w}" height="${h}"/>`];
res.masks.forEach((m) => {                                   // masks: one 1-px-high rect per run of a row
  const bits = m.toMask(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!bits[y * w + x] || bits[y * w + x - 1] && x > 0) continue;
      let e = x;
      while (e < w && bits[y * w + e]) e++;
      parts.push(`<rect x="${x}" y="${y}" width="${e - x}" height="1" fill="#00e0ff" opacity="0.4"/>`);
    }
});
for (const d of res.detections) {                            // boxes + labels
  const [x, y, bw, bh] = d.bbox as [number, number, number, number];
  parts.push(`<rect x="${x}" y="${y}" width="${bw}" height="${bh}" fill="none" stroke="yellow" stroke-width="3"/>`);
  parts.push(`<text x="${x + 4}" y="${y + 18}" fill="yellow" font-family="sans-serif" font-size="16">${d.cls} ${d.conf.toFixed(2)}</text>`);
}
for (const g of res.grasps) {                                // grasps: a line between the two jaws
  const dx = (Math.cos(g.theta) * g.width) / 2, dy = (Math.sin(g.theta) * g.width) / 2;
  parts.push(`<line x1="${g.x - dx}" y1="${g.y - dy}" x2="${g.x + dx}" y2="${g.y + dy}" stroke="red" stroke-width="3"/>`);
}
await writeFile("food-custom.svg", `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">${parts.join("")}</svg>`);
console.log(res.masks.length, "masks,", res.detections.length, "boxes,", res.grasps.length, "grasps");
```

```text
3 masks, 3 boxes, 60 grasps
```

The file shows the three masks in light blue, yellow boxes with labels, and 60 red grasp lines
(the JS SDK keeps every grasp the server sends). Every row of a mask adds elements, so the file
grows with the masks: 390 KB here, 300 KB of it the embedded photo. For many or large masks, use
a canvas instead.

## Utilities

These helpers work on results in your program and never call the server.

| Where | Helper | What it does | Use it when |
|---|---|---|---|
| `Result` | `filterByConf(minConf = 0, maxConf = 1)` | Keeps detections, masks and classifications whose `conf` is in the range. | Hiding weak guesses. |
| module | `filterBySize(result, { minSize, maxSize, imageWidth, imageHeight })` (also `client.filterBySize`) | Keeps detections and masks by box area: **fractions** (0–1) of the photo when both sizes are given, else pixels². `0` = no limit. | Dropping tiny or huge boxes. |
| `Result` | `sortByConf(descending = true)`, `topK(k)` | Orders by `conf`; keeps the `k` best of each list. | Only the best few. |
| `Result` | `nms(iouThreshold = 0.5)` | Greedy non-maximum suppression on detections (masks are kept). | Merged results with overlapping boxes. |
| `Result` | `groupByClass()` | `{ label: Result }`; a mask joins the detection with the same box, other masks go under `""`. Each group keeps the result's depth map, embeddings and all grasps. | Handling each class apart. |
| `Result` | `Result.fromJSON(obj)` | Builds a `Result` from the server's JSON (`class`, `depth_map`, ...). | Results saved from an HTTP call. |
| `Mask` | `toMask(width, height)`, `toMask2D(width, height)` | The mask as a row-major `Uint8Array` (1 = inside), or `boolean[height][width]`. | Area, union, painting pixels. |
| module | `getDepthAtDetection(depthResult, detResult, { imageWidth, imageHeight, mode })` | The depth model's value under each box (relative 0–1, larger = closer). | "Which object is nearer?" |
| module | `toSVG(result, width, height)` | Boxes and labels as an SVG string ([above](#visualize-results)). | Drawing. |
| module | `probeHeader(bytes)` | `{ width, height, isJpeg, orientation }` from a JPEG or PNG header, EXIF rotation applied, or `null`. | The photo's size without an image library. |
| module | `targetSize(width, height, { maxSide, maxShortSide, region })` | The size a photo would be shrunk to. | Planning uploads. |
| `Client` | `usefulSide(model)` | `[maxSide, maxShortSide]`: the model's size hint, `null` = none. | Knowing how big a photo is worth sending. |
| module | `ClientResize` (`res.clientResize`) | What was uploaded: `originalWidth/Height`, `sentWidth/Height`, `jpegQuality`, `reason`, `resized`, `toSentX/Y()`, `toOriginalX/Y()`. | Checking the client-side resize. |
| module | `normalizePrompt(model, prompt)`, `isLoopback(host)` | The prompt `predict` sends; whether a host is this machine. | Debugging prompts and resizing. |

Not in the JS SDK (use the [Python SDK](python.md#utilities) or write a few lines): grasp
helpers (`filterGrasps`, a grasp's pose or jaw points; the SVG example above has the jaw maths), camera and
robot helpers (back-projection, distances in metres, target selection), and a `toJSON` in the
server's format: `JSON.stringify(res)` writes the object's own names (`cls`, `depthMap`, ...),
which `Result.fromJSON` does not read back.

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

### JSON

```ts
import { Client, Result } from "visionserve";

const client = new Client();
const res = await client.predict("grounded-sam", "dogs.jpg", { prompt: "dog." });
console.log(Object.keys(JSON.parse(JSON.stringify(res))).slice(0, 6));   // the object's own names
const back = Result.fromJSON({ task: "detection", model: "m", detections: [{ bbox: [1, 2, 3, 4], class: "dog", conf: 0.9 }] });
console.log(back.detections[0]!.cls, back.detections[0]!.bbox);
```

```text
[ 'task', 'model', 'detections', 'masks', 'grasps', 'classifications' ]
dog [ 1, 2, 3, 4 ]
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
