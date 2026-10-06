# visionserve (JavaScript / TypeScript client)

A lightweight TypeScript/JavaScript **client** SDK for the [VisionServe](../../) HTTP
server. It talks to the Go runtime over REST — it does **not** run inference itself.

It is the sibling of the [Python client](../python/) and mirrors its API. Transport uses
only built-in globals (`fetch`, `FormData`, `Blob`), so it has **zero runtime
dependencies** and runs on **Node >= 18** and in modern browsers. (Passing a file *path*
to `predict()` uses `node:fs` and is Node-only; in the browser pass bytes or a `Blob`.)

## Contents

- [Install](#install)
- [Usage](#usage)
  - [Image inputs](#image-inputs)
  - [Client-side resizing (on by default)](#client-side-resizing-on-by-default)
  - [Prompt options](#prompt-options-opts)
  - [Detection](#detection)
  - [Segmentation](#segmentation)
  - [Open-vocab / Grounded-SAM](#open-vocab--grounded-sam)
  - [Depth estimation](#depth-estimation)
  - [Classification](#classification)
  - [CLIP embeddings](#clip-embeddings)
  - [Grasp detection](#grasp-detection)
  - [Result schema](#result-schema)
- [CLI](#cli)
- [Post-processing](#post-processing)
- [Grasp and robot helpers](#grasp-and-robot-helpers)
- [Size filtering](#size-filtering)
- [Visualization](#visualization)
- [JSON](#json)
- [Other API](#other-api)
- [Develop](#develop)

## Install

```bash
npm install visionserve          # from npm
# or from source:
cd clients/js && npm install && npm run build
```

Start the server in another terminal:

```bash
make serve                       # listens on :11435
```

## Usage

```ts
import { Client } from "visionserve";
const client = new Client("http://127.0.0.1:11435");
```

### Image inputs

`predict(model, image, opts?)` accepts three image types — choose whichever fits your
pipeline:

```ts
import fs from "node:fs";

// 1. File path — simplest (Node only)
const res = await client.predict("rf-detr", "photo.jpg");

// 2. Uint8Array / ArrayBuffer — already-encoded PNG/JPEG bytes (Node + browser)
const bytes = new Uint8Array(fs.readFileSync("photo.jpg"));
const res2 = await client.predict("rf-detr", bytes);

// 3. Blob — from browser <input> or fetch (browser + Node)
const blob = await fetch("photo.jpg").then((r) => r.blob());
const res3 = await client.predict("rf-detr", blob);
```

`ArrayBuffer` is also accepted and behaves identically to `Uint8Array`.

### Client-side resizing (on by default)

A model resizes every photo to its own small input (RF-DETR: 560 × 560), so most of a
12-megapixel upload is thrown away on the server. By default (`resize: "auto"`) the client
shrinks a photo larger than the model can use to the size the server advertises for that model
(`GET /api/models`: `max_useful_side` / `max_useful_short_side`, 2 × the model's input; fetched
once and cached), uploads it as JPEG (`jpeg: true`, `jpegQuality: 90`), and maps every box,
mask and grasp back to the **original** photo's pixels. `res.clientResize` says what was sent
and why (`reason`; `null` = the bytes you passed, untouched). Only a shrunk photo is
re-encoded. Models whose output needs the full photo (masks, OCR, grasping, templates) have no
hint and always get it. With the server on this machine (`localhost`, 127.0.0.0/8, `::1`) a
photo is shrunk only when that at least halves its sides (the loopback rule: on localhost a
mild shrink costs more to decode than it saves).

```ts
new Client(host, { resize: "off" });                 // send every photo as given
new Client(host, { resize: 1280, jpeg: false });     // longer side 1280 px, PNG
await client.predict("rf-detr", "photo.jpg", { resize: "off" });   // per call
```

Decoding needs a codec, and the SDK ships none: in a **browser** it uses `createImageBitmap` +
`OffscreenCanvas`; in **Node** it uses [`sharp`](https://www.npmjs.com/package/sharp) if your
application has installed it (`npm install sharp`; it is never installed by this package; it
shrinks JPEGs on load), and otherwise sends photos exactly as given (no error, no warning: results are the same, the upload
is larger). Pass `codec: null` to disable resizing, or your own `ImageCodec`. The browser's JPEG
encoder subsamples colour (4:2:0); the Python SDK and `sharp` use 4:4:4. The measured effect on
bytes, latency and accuracy is in the
[Python docs](../../website/docs/clients/python.md#client-side-resizing-on-by-default).

### Detection

```ts
// RF-DETR (COCO-80) or RT-DETR
const det = await client.predict("rf-detr", "photo.jpg");
for (const d of det.detections) {
  console.log(d.cls, d.conf.toFixed(3), d.bbox); // bbox = [x, y, w, h] in original pixels
}
```

### Segmentation

```ts
// Box-prompted — decode the column-major RLE mask
const seg = await client.predict("mobile-sam", "photo.jpg", { box: [34, 58, 120, 240] });
const mask = seg.masks[0]?.toMask(640, 480);    // row-major Uint8Array; 1 = inside mask
const mask2d = seg.masks[0]?.toMask2D(640, 480); // boolean[][]

// No prompt → Automatic Mask Generator (segment everything)
const amg = await client.predict("mobile-sam", "photo.jpg");
console.log(`found ${amg.masks.length} masks`);

// EfficientSAM and SAM2 — same interface
const seg2 = await client.predict("efficient-sam", "photo.jpg", { box: [34, 58, 120, 240] });
```

### Open-vocab / Grounded-SAM

```ts
// GroundingDINO — text → boxes
const gd = await client.predict("grounding-dino", "photo.jpg", { prompt: "cat. remote." });
for (const d of gd.detections) console.log(d.cls, d.conf.toFixed(3), d.bbox);

// Grounded-SAM — text → boxes → masks
const gs = await client.predict("grounded-sam", "photo.jpg", { prompt: "cat. remote." });
console.log(gs.detections.map((d) => d.cls), "→", gs.masks.length, "masks");
```

### Depth estimation

```ts
const dep = await client.predict("depth-anything-v2", "photo.jpg");
// dep.depthMap is a number[], row-major, length = dep.depthWidth × dep.depthHeight
const depth2d: number[][] = [];
for (let y = 0; y < dep.depthHeight; y++) {
  depth2d.push(Array.from(dep.depthMap.slice(y * dep.depthWidth, (y + 1) * dep.depthWidth)));
}
```

### Classification

```ts
const cls = await client.predict("efficientnet-b0", "photo.jpg");
for (const c of cls.classifications) console.log(c.cls, c.conf.toFixed(3));
```

### CLIP embeddings

```ts
const emb = await client.predict("clip", "photo.jpg");
// emb.embeddings[0] is a number[] of length 512
const vec = emb.embeddings[0];
const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
const unit = vec.map((v) => v / norm);  // L2-normalize before cosine similarity
```

### Grasp detection

```ts
// Class-agnostic grasps (whole image)
const grasp = await client.predict("grasp", "bin.jpg");
for (const g of grasp.grasps) {
  console.log(`q=${g.quality.toFixed(3)}  x=${g.x.toFixed(1)} y=${g.y.toFixed(1)}`
            + `  θ=${g.theta.toFixed(3)}  w=${g.width.toFixed(1)}`);
}

// Class-aware grasps — text-prompted detector (grasp-gd)
const graspGd = await client.predict("grasp-gd", "table.jpg", { prompt: "mug. bottle." });
for (const g of graspGd.grasps) console.log(g.cls, g.quality.toFixed(3));
```

`Result.grasps` is `Grasp[]` where each `Grasp` has `{ x, y, theta, width, quality, cls, conf }`,
plus `pose` (`[x, y, width, theta]`) and `contacts()` (the two jaw points). The server can send up
to 20 grasps per object; `res.filterGrasps(3)` keeps the best 3 per object (Python's default).

### Prompt options (`opts`)

`predict()` takes every option the Python SDK's `predict()` sends. Each is sent as the
server form field of the same name in snake_case (`boxThreshold` → `box_threshold`); leave
it out (or `undefined` / `null`) for the server's default. An unknown key (for example
`min_size` instead of `minSize`) throws a `TypeError` before anything is sent.

| Option | Server field | For | Format |
|--------|--------------|-----|--------|
| `prompt` | `prompt` | open-vocab text | `"cat. remote."` (normalised, see below) |
| `box` | `box` | SAM box | `[x, y, w, h]` or a list `[[...], [...]]` |
| `point` | `point` | SAM point | `[x, y]` / `[x, y, label]` or a list (label 1=fg, 0=bg) |
| `boxThreshold`, `textThreshold` | `box_threshold`, `text_threshold` | GroundingDINO family | number in (0, 1) |
| `minSize`, `maxSize` | `min_size`, `max_size` | every model with boxes/masks | % of the image area, applied by the **server** |
| `roi` | `roi` | every model | `[x, y, w, h]`, pixels or 0–1 fractions |
| `dilate` | `dilate` | masks | integer pixels, `> 0` grow, `< 0` shrink |
| `method` | `method` | `background`, `rfdetr-textalign*` | `"auto"`, `"depth"`, `"sam"`, `"cv"`, `"automask"` / `"exact"`, `"dual"` |
| `bgMaxArea`, `fgMinArea` | `bg_max_area`, `fg_min_area` | `background` (sam/automask) | % of the image area |
| `gridSize` | `grid_size` | automask | integer (server max 64) |
| `depth` (+ `depthDtype`, `depthWidth`, `depthHeight`) | file part `depth` + `depth_dtype`, `depth_width`, `depth_height` | `background` (depth/auto) | `Uint16Array` (sent as `uint16`) or `Float32Array` (`float32`), row-major; raw little-endian bytes need `depthDtype`. Size defaults to the image's |
| `gripperMin`, `gripperMax` | `gripper_min`, `gripper_max` | grasp models | jaw opening in image pixels |
| `claimThreshold` | `claim_threshold` | `rfdetr-textalign*`, `method: "dual"` | probability; `>= 1` never claims |
| `cropTemp` | `crop_temp` | SigLIP crop namers | softmax temperature, lower = more decisive |
| `templateName` | `template_name` | `instance_detection` models | a set registered with `POST /api/templates` |

Boxes and points are in **original-image** coordinates, matching the server and the
Python client. Numbers must be finite; `gridSize`, `dilate`, `depthWidth` and `depthHeight`
must be integers (the server would read `2.5` as 0, the default).

```ts
const res = await client.predict("grounding-dino", "kitchen.jpg", {
  prompt: "cup, bowl",          // sent as "cup. bowl"
  boxThreshold: 0.4,
  roi: [0, 0.5, 1, 0.5],        // bottom half (fractions)
  minSize: 0.5,                 // drop boxes under 0.5% of the image
});
```

**Prompts** are normalised exactly as in the Python SDK (`normalizePrompt(model, prompt)`
shows what is sent): `,` and `|` become the `.` phrase separator, a single phrase gets a
trailing `.`, CLIP / SigLIP towers get the text unchanged, and the GroundingDINO family
(`grounding-dino`, `grounded-sam`, `grasp-gd`, `gdino-siglip*`) gets `"object."` when no
prompt is given. Both SDKs run the shared cases in `clients/testdata/normalize_prompt.json`.

**`new Client(host, { base64Arrays: true })`** asks the server for depth maps and
embeddings as base64 float32 (`encoding=base64`): about half the bytes and much cheaper to
parse. The SDK decodes them, so `depthMap` and `embeddings` stay plain `number[]` holding
the exact float32 values.

### Result schema

Every task returns the same unified `Result`:

```ts
class Result {
  task: string;                  // "detection" | "segmentation" | "open_vocab" |
                                 // "depth" | "classification" | "embedding" | "grasp" | ...
  model: string;
  device: string;                // "cpu" | "gpu:0" | "gpu:0+trt"
  hint: string;                  // the server's setup recommendation, "" if none
  detections: Detection[];       // { bbox: [x,y,w,h], cls: string, conf: number }
  masks: Mask[];                 // { rle, bbox, conf } — column-major RLE
  classifications: Classification[]; // { cls: string, conf: number } — top-K
  grasps: Grasp[];               // { x, y, theta, width, quality, cls, conf }
  depthMap: number[];            // flat row-major, length depthWidth×depthHeight (MODEL resolution)
  depthWidth: number;
  depthHeight: number;
  embeddings: number[][];        // one 512-d vector per image (CLIP)
  durationMs: number;
}
```

`Mask.toMask(width, height)` decodes the column-major RLE into a row-major `Uint8Array`
(`1` = inside the mask); `Mask.toMask2D(width, height)` returns a `boolean[][]`. Pass the
**original** image width/height; when the client shrank the photo, the mask is decoded at the
sent size (`mask.rleSize`) and scaled up (nearest neighbour).

## CLI

Installing the package globally (or running it via `npx`) exposes a `visionserve`
command — a thin HTTP client over the same REST API. It does **not** run inference; it
talks to a running VisionServe server (the Go binary `visionserve serve`, default
`http://127.0.0.1:11435`). Zero runtime dependencies (built-in `fetch`/`FormData`),
**Node >= 18**.

```bash
npm install -g visionserve   # adds the `visionserve` command
# or run without installing:
npx visionserve --help
```

### Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `predict <model> <image> [flags]` | `run` | Run a model on an image, print the unified result as JSON |
| `list` | `models`, `ls` | List available models |
| `ps` | | List loaded models |
| `load <model>` | | Load a model into memory |
| `unload <model>` | `rm` | Unload a model |
| `health` | | Check server health |

### Global flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host <url>` | `http://127.0.0.1:11435` | Server base URL |
| `--timeout <sec>` | `120` | Per-request timeout in seconds |
| `-h`, `--help` | | Show help |
| `--version` | | Print the client version |

`list` and `ps` also accept `--json`.

### `predict` flags

| Flag | Description |
|------|-------------|
| `--prompt "<text>"` | Open-vocab text prompt, e.g. `"cat. remote."` (GroundingDINO / grasp-gd) |
| `--box x,y,w,h` | SAM box prompt(s) in **original** image pixels; multiple separated by `;` |
| `--point x,y[,l]` | SAM point prompt(s); label `1`=fg, `0`=bg; multiple separated by `;` |
| `--min-size PCT` / `--max-size PCT` | Drop objects whose bbox area is below/above PCT% of the image (applied **client-side**; requires a PNG/JPEG so the image size can be read) |
| `--resize auto\|off\|N` | Client-side resizing (default `auto`; needs `sharp` in Node, else the file is sent as is) |
| `--no-jpeg` / `--jpeg-quality Q` | Send a shrunk photo as PNG / JPEG quality (default 90) |
| `--save` | Save an annotated SVG with an auto name `<stem>.js.<model>.<task>.svg` |
| `--save-as PATH` | Save the annotated SVG to this exact path |
| `--compact` | Print result JSON on a single line (default: pretty) |
| `--quiet` | Suppress the stderr summary line |

Notes:

- The JS client does **not** support `--gripper-min`/`--gripper-max` (those are Python/Go
  only).
- `--save` writes an **SVG**, not a raster. The source image is embedded as a base64
  background so the SVG is viewable standalone. The overlay is `toSVG`'s: masks, boxes,
  grasps (the best 3 per object) and labels.
- Image-size sniffing supports **PNG and JPEG only**; if the size can't be determined,
  `--save` is skipped with a warning.

### Output

`predict` prints the unified result as JSON to **stdout** (pipe-friendly; field names
match the server wire schema: it is `Result.toJSON()`, `class`, empty fields omitted). A
one-line summary goes to **stderr**:

```
predict: model=rf-detr task=detection device=gpu:0  client=42.1ms server=12.3ms  (12 detections)
```

where `client` is the wall-clock time around the `predict()` round-trip and `server` is
the server's `duration_ms` (inference only). Both are measured **before** the SVG is built
or saved. The auto filename is `<stem>.<client_type>.<model>.<task>.svg` with `client_type`
`js`, so Python/JS/Go outputs never collide.

### Examples

```bash
# Start the server (Go binary) first, then:
npx visionserve predict rf-detr cat.jpg
npx visionserve predict grounding-dino cat.jpg --prompt "cat. remote." --save
npx visionserve predict mobile-sam dog.jpg --box 50,40,200,180 --save-as dog.svg
npx visionserve --host http://10.0.0.5:11435 list --json
npx visionserve ps
```

## Post-processing

All methods return a **new** `Result`; the original is not modified.

```typescript
import { Client, getDepthAtDetection } from "visionserve";

const client = new Client();
let result = await client.predict("rf-detr", imageBytes);

// Filter, sort, NMS
result = result.filterByConf(0.5)
               .nms(0.45)
               .sortByConf();

// Top-5
const top5 = result.topK(5);

// Group by class: each group has only that class's detections, masks and grasps
const byClass = result.groupByClass();
for (const [cls, r] of Object.entries(byClass)) {
  console.log(`${cls}: ${r.detections.length} detections`);
}

// Depth fusion: the depth map is at the model's resolution (e.g. 256×256), the boxes are in
// image pixels, so pass the image's size and the boxes are scaled onto the map.
const depth = await client.predict("midas", imageBytes);
const depths = getDepthAtDetection(depth, result, { imageWidth: 640, imageHeight: 480 });
result.detections.forEach((det, i) => {
  console.log(`${det.cls}: depth=${depths[i]?.toFixed(2) ?? "N/A"}`);
});
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `filterByConf` | `(minConf, maxConf?)` | Keep predictions with conf in `[minConf, maxConf]` |
| `sortByConf` | `(descending?)` | Sort predictions by confidence (default descending) |
| `topK` | `(k)` | Retain top-k predictions by confidence |
| `nms` | `(iouThreshold?)` | Greedy NMS on detections |
| `filterGrasps` | `(maxPerObject)` | Keep the best grasps per object (grouped by the detection each grasp was planned on) |
| `groupByClass` | `()` | `Record<string, Result>` keyed by class label: that class's detections, masks (by matching box, else `""`) and grasps; classifications, depth map and embeddings left empty in every group (as Python's `group_by_class`) |
| `toJSON` | `(opts?)` | The server's wire JSON (`class`, snake_case); `JSON.stringify(res)` uses it, `{ encoding: "base64" }` for base64 arrays |

`getDepthAtDetection(depthResult, detResult, { imageWidth, imageHeight, mode? })` (exported
from the top-level `visionserve` package, implemented in `filter.ts`) returns
`(number | null)[]` — one depth value per detection (or per mask when there are no
detections), or `null` when the box falls outside the depth map. `imageWidth` /
`imageHeight` are the size of the original image the boxes refer to and are required: the
boxes are scaled by `depthWidth / imageWidth` and `depthHeight / imageHeight`. `mode` is
`"median"` (default), `"mean"`, `"min"` or `"max"`. The values are the server's relative
inverse depth (`[0, 1]`, larger = closer), not metres; `0` counts as a value. Same rule as
the Python `get_depth_at_detection(..., image_size=(W, H))`.

## Grasp and robot helpers

A port of the Python SDK's `postprocess.py` (same maths and defaults; both SDKs run the shared
cases in `clients/testdata/postprocess_sync.json`, generated from the Python code):

```ts
import { selectTargetGrasp, selectTargetObject, graspDistances, objectDistances, backproject } from "visionserve";

const g = await client.predict("grasp-rfdetr", "table.jpg");
const obj = selectTargetObject(g, { cls: "cup", nearPoint: "center", imageWidth: 640, imageHeight: 480 });
const grasp = selectTargetGrasp(g.grasps, { cls: "cup", gripperMin: 30, gripperMax: 80 });
console.log(grasp?.pose, grasp?.contacts());          // [x, y, width, theta], jaw points

// Metres need a METRIC depth image from an RGB-D camera, aligned with the photo:
const depth = { data: depthMm /* Uint16Array, millimetres */, width: 640, height: 480 };
const K = { fx: 600, fy: 600, cx: 320, cy: 240 };     // or [fx, fy, cx, cy]
objectDistances(depth, g, K);                          // camera -> object, metres (null = no reading)
graspDistances(depth, g.grasps, K);                    // camera -> grasp centre
selectTargetObject(g, { depth, intrinsics: K, targetDistance: 0.6 });
backproject(320, 200, 0.6, K);                         // pixel + depth -> [X, Y, Z] in the camera frame
```

`selectTargetObject` scores `conf`, `area`, `near` (`nearPoint`) and `distance`
(`targetDistance`); `selectTargetGrasp` scores `quality`, `near` (`targetPoint`), `distance` and
`width` (within `gripperMin`–`gripperMax`). Without `weights` the most specific criterion given is
used (distance, else near, else the model's score); `weights: { conf: 1, area: 1 }` mixes them.
`selectTargetObjectIndex` / `selectTargetGraspIndex` also return the index. A depth model's result
(`midas`) is relative and has no scale, so the distance helpers refuse it.

## Size filtering

Keep only objects whose bbox area is within a range. Available as a standalone function
or as a method on `Client` (both are equivalent).

```ts
import { filterBySize } from "visionserve";

const res = await client.predict("rf-detr", "image.jpg");

// Absolute mode — area in pixels² (client-side filtering on received results)
const big = filterBySize(res, { minSize: 5000 });
const mid = filterBySize(res, { minSize: 500, maxSize: 50000 });

// Relative mode — fraction of image area (0.0–1.0), supply imageWidth + imageHeight
const rel = filterBySize(res, {
  minSize: 0.005,    // at least 0.5% of image area
  maxSize: 0.9,      // at most 90% of image area
  imageWidth: 1280,
  imageHeight: 720,
});

// Via Client method:
const filtered = client.filterBySize(res, { minSize: 500 });
```

This filters the answer on the client. To have the **server** drop small or large objects
(before grasp planning, for grasp models), pass `minSize` / `maxSize` (percent of the image
area) to `predict()` instead.

## Visualization

`toSVG(result, width, height, opts?)` returns an SVG string that draws a result over its photo, in
the style of the Python SDK's `draw()`: mask pixels as translucent colours (one embedded PNG,
encoded by the SDK itself), boxes with `class conf%` labels, grasps (the best 3 per object) and
classification labels. Zero runtime dependencies, synchronous, the same in the browser and in Node.

`width` / `height` are the ORIGINAL photo's size (the frame of every coordinate;
`probeHeader(bytes)` reads it), not the display size. The SVG has a `viewBox`, so it scales to any
CSS size:

```ts
import { toSVG, selectTargetGrasp } from "visionserve";

const res = await client.predict("grasp-rfdetr", "image.jpg");
const svg = toSVG(res, 1280, 720, {
  targetGrasp: selectTargetGrasp(res.grasps), // drawn in red, on top
  maxGraspsPerObject: 3,                      // default
});

// In HTML, over the <img>, at any size:
// <div style="position:relative; display:inline-block">
//   <img src="image.jpg" style="width:480px">
//   <div style="position:absolute; inset:0">${svg with style="width:100%;height:100%"}</div>
// </div>
```

| Option | Default | |
|--------|---------|---|
| `alpha` | `0.45` | Mask opacity |
| `masks` / `maskBoxes` | `true` / `true` | Draw mask pixels / a box + `mask conf%` for masks no detection labels |
| `maxGraspsPerObject` | `3` | `null` draws every grasp |
| `targetGrasp`, `targetBox` | none | Highlight in red |
| `colorBy` | `"class"` | One colour per class name (the same as Python's); `"index"`: the server's 8 colours by position |
| `fontSize`, `lineWidth` | scaled with the photo | Label size and line width in photo pixels |

Depth and embedding results draw nothing (an empty `<svg>`).

## JSON

`JSON.stringify(res)` writes the server's wire format (`class`, `duration_ms`, empty fields
omitted), so `Result.fromJSON(JSON.parse(JSON.stringify(res)))` deep-equals `res`. A photo the
client shrank keeps its `clientResize` as an extra `client_resize` field (the server never sends
it). `res.toJSON({ encoding: "base64" })` writes depth maps and embeddings as base64 float32, like
`encoding=base64` answers.

## Other API

```ts
await client.health();      // { status: "ok" }
await client.listModels();  // ModelInfo[]  (name, task, license, state, maxUsefulSide, maxUsefulShortSide)
await client.ps();          // only loaded models
await client.load("rf-detr");
await client.unload("rf-detr");
```

## Errors

Every failure is a `VisionServeError`:

- `status`: the HTTP status, or `undefined` when no answer came back (server down, timeout).
- `retryAfter`: on a `503` (the model's queue is full), the server's `Retry-After` in
  seconds; wait that long and retry. `undefined` otherwise.
- a timeout reads `"GET /api/health: timed out after 50 ms waiting for VisionServe at …"`;
  an unreachable server reads `"failed to reach VisionServe at …"`. The timeout covers the
  whole request, including reading the answer.

```ts
import { VisionServeError } from "visionserve";

try {
  await client.predict("rf-detr", "photo.jpg");
} catch (e) {
  if (e instanceof VisionServeError && e.status === 503) {
    await new Promise((r) => setTimeout(r, (e.retryAfter ?? 1) * 1000));
    // ...retry
  } else throw e;
}
```

Malformed options (a wrong `box` length, a non-integer `gridSize`, an unknown option key)
throw a plain `Error` / `TypeError` before anything is sent.

## Develop

```bash
npm install
npm run build      # tsc → dist/
npm test           # node:test with a mocked fetch (no server needed)
npm run typecheck  # tsc --noEmit
```

## Changelog

### 0.2.0

- **`toSVG` draws everything Python's `draw()` does on a photo**: mask PIXELS (one palette PNG
  in an `<image>`, from a tiny synchronous deflate encoder in the SDK: a few KB per photo, no
  `node:zlib`, no canvas), grasp glyphs (the best 3 per object; `maxGraspsPerObject`,
  `targetGrasp`), `instance_detection` and any task with detections / masks / grasps /
  classifications (it used to dispatch on `task` and drew nothing for `grasp` results), and
  `targetBox`, `maskBoxes`, `masks`, `alpha`, `fontSize`, `lineWidth`. **Changed look:** a
  `viewBox` (the SVG scales to any CSS size), colours by class by default (`colorBy: "class"`,
  the Python SDK's 16 class colours; `"index"` for the old palette by position), labels on a
  coloured band with black or white text, `conf` as a whole percentage, sizes scaled with the
  photo. The CLI's `--save` therefore draws grasps too.
- **JSON in the server's format**: `toJSON()` on `Result`, `Detection`, `Mask`, `Grasp` and
  `Classification`, so `JSON.stringify(res)` writes `class`, `duration_ms`, ... and
  `Result.fromJSON(JSON.parse(JSON.stringify(res)))` deep-equals `res` (`clientResize` is kept as
  `client_resize`); `toJSON({ encoding: "base64" })`. The CLI prints `res.toJSON()` (it now
  includes `hint` when the server sends one).
- **`groupByClass()` as Python's `group_by_class()`** (**breaking**): a group holds only its
  class's grasps (it held every grasp), and has no classifications, depth map or embeddings (it
  had copies of the whole result's).
- **Grasp and robot helpers**, ported from Python with shared test cases: `Result.filterGrasps`,
  `Grasp.pose` / `contacts()` / `contactsFlat()`, `backproject`, `cameraDistance`,
  `objectDistances`, `graspDistances`, `selectTargetObject(Index)`, `selectTargetGrasp(Index)`;
  `getDepthAtDetection` also takes a metric depth image (`{ data, width, height }`, `depthScale`).

### 0.1.4

- **Client-side resizing, on by default** (a user decision: it changes what is uploaded). With
  `resize: "auto"` the client shrinks a photo larger than the model can use to the server's
  hint (`GET /api/models`: `max_useful_side` / `max_useful_short_side`), sends it as JPEG
  (`jpeg: true`, `jpegQuality: 90`) and maps boxes, masks and grasps back to original pixels;
  `Result.clientResize` records it, with a `reason`. Only shrunk photos are re-encoded. With
  a loopback `host`, only a shrink to half the sides or less is done (`isLoopback`). `resize: "off"` restores the old uploads exactly; also
  per call (`predict(..., { resize, jpeg, jpegQuality })`) and in the CLI (`--resize`,
  `--no-jpeg`, `--jpeg-quality`). It needs a codec: the browser's, or `sharp` in Node when
  installed (optional; otherwise photos are sent as given). New exports: `ClientResize`,
  `ImageCodec`, `browserCodec`, `sharpCodec`, `probeHeader`, `targetSize`, `isLoopback`.
- `ModelInfo.maxUsefulSide` / `maxUsefulShortSide`; `Mask.rleSize`.

### 0.1.3

- `predict()` takes every option the Python SDK sends (`boxThreshold`, `textThreshold`,
  `minSize`, `maxSize`, `roi`, `method`, `dilate`, `bgMaxArea`, `fgMinArea`, `gridSize`,
  `gripperMin`/`gripperMax`, `claimThreshold`, `cropTemp`, `templateName`, `depth` with
  `depthDtype`/`depthWidth`/`depthHeight`) and `Client` takes `base64Arrays`. An unknown
  option key now throws a `TypeError` (it used to be ignored silently).
- Prompts are normalised like Python's (`"cat, dog"` → `"cat. dog"`), and the
  GroundingDINO family defaults to `"object."`: `grounding-dino` without a prompt no longer
  fails with a 400. `normalizePrompt` is exported.
- `VisionServeError.retryAfter` (seconds, from a 503's `Retry-After`); a timeout says
  "timed out after N ms" instead of "failed to reach", and also covers reading the body.
- `Result.hint` keeps the server's setup recommendation.
- **Breaking:** `getDepthAtDetection(depth, det, { imageWidth, imageHeight, mode? })` takes
  the image size and scales the boxes onto the model-sized depth map (it used to read the
  map at image-pixel positions, i.e. the wrong pixels), and counts `0` as a depth value. The
  old third argument (`mode` as a string) is replaced by the options object.
