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
- [Size filtering](#size-filtering)
- [Visualization](#visualization)
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
const client = new Client("http://localhost:11435");
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
// dep.depthMap is a Float32Array, length = dep.depthWidth × dep.depthHeight
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

`Result.grasps` is `Grasp[]` where each `Grasp` has `{ x, y, theta, width, quality, cls, conf }`.

### Prompt options (`opts`)

| Field | For | Format |
|-------|-----|--------|
| `prompt` | open-vocab text | `"cat. remote."` |
| `box` | SAM box | `[x, y, w, h]` or a list `[[...], [...]]` |
| `point` | SAM point | `[x, y]` / `[x, y, label]` or a list (label 1=fg, 0=bg) |

Boxes and points are in **original-image** coordinates, matching the server and the
Python client.

### Result schema

Every task returns the same unified `Result`:

```ts
class Result {
  task: string;                  // "detection" | "segmentation" | "open_vocab" |
                                 // "depth" | "classification" | "embedding" | "grasp" | ...
  model: string;
  device: string;                // "cpu" | "gpu:0" | "gpu:0+trt"
  detections: Detection[];       // { bbox: [x,y,w,h], cls: string, conf: number }
  masks: Mask[];                 // { rle, bbox, conf } — column-major RLE
  classifications: Classification[]; // { cls: string, conf: number } — top-K
  grasps: Grasp[];               // { x, y, theta, width, quality, cls, conf }
  depthMap: Float32Array;        // flat row-major, length depthWidth×depthHeight
  depthWidth: number;
  depthHeight: number;
  embeddings: number[][];        // one 512-d vector per image (CLIP)
  durationMs: number;
}
```

`Mask.toMask(width, height)` decodes the column-major RLE into a row-major `Uint8Array`
(`1` = inside the mask); `Mask.toMask2D(width, height)` returns a `boolean[][]`. Pass the
**original** image width/height the mask was produced against.

## CLI

Installing the package globally (or running it via `npx`) exposes a `visionserve`
command — a thin HTTP client over the same REST API. It does **not** run inference; it
talks to a running VisionServe server (the Go binary `visionserve serve`, default
`http://localhost:11435`). Zero runtime dependencies (built-in `fetch`/`FormData`),
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
| `--host <url>` | `http://localhost:11435` | Server base URL |
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
| `--save` | Save an annotated SVG with an auto name `<stem>.js.<model>.<task>.svg` |
| `--save-as PATH` | Save the annotated SVG to this exact path |
| `--compact` | Print result JSON on a single line (default: pretty) |
| `--quiet` | Suppress the stderr summary line |

Notes:

- The JS client does **not** support `--gripper-min`/`--gripper-max` (those are Python/Go
  only).
- `--save` writes an **SVG**, not a raster. The source image is embedded as a base64
  background so the SVG is viewable standalone. The overlay draws detections, masks, and
  classifications but **not** grasp glyphs — for grasp models the grasp data is in the JSON
  output (a note is printed to stderr).
- Image-size sniffing supports **PNG and JPEG only**; if the size can't be determined,
  `--save` is skipped with a warning.

### Output

`predict` prints the unified result as JSON to **stdout** (pipe-friendly; field names
match the server wire schema: `class`, empty arrays omitted, includes `grasps` and
`device`). A one-line summary goes to **stderr**:

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

// Group by class
const byClass = result.groupByClass();
for (const [cls, r] of Object.entries(byClass)) {
  console.log(`${cls}: ${r.detections.length} detections`);
}

// Depth fusion
const depth = await client.predict("midas", imageBytes);
const depths = getDepthAtDetection(depth, result);
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
| `groupByClass` | `()` | Returns `Record<string, Result>` keyed by class label |

`getDepthAtDetection(depthResult, detResult, mode?)` (exported from the top-level
`visionserve` package, implemented in `filter.ts`) returns `(number | null)[]` — one
depth value per detection/mask, or `null` when the box falls outside the depth map.
`mode` is `"median"` (default) or `"mean"`.

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
// Note: pass min_size/max_size to predict() for server-side filtering (% of image area)
const serverFiltered = await client.predict("rf-detr", "image.jpg",
  { minSize: 0.5, maxSize: 80 });   // 0.5%–80% of image area
```

## Visualization

`toSVG` returns a ready-to-embed SVG string with annotation overlays. Zero runtime
dependencies — works in the browser and in Node.

```ts
import { toSVG } from "visionserve";

const res = await client.predict("rf-detr", "image.jpg");
const svg = toSVG(res, 1280, 720);  // width, height of the original image

// In HTML — position the SVG over the <img>:
// <div style="position:relative; display:inline-block">
//   <img src="image.jpg" width="1280" height="720">
//   <svg style="position:absolute;top:0;left:0;pointer-events:none"
//        [innerHTML]="svg"></svg>
// </div>
```

What `toSVG` draws per task:

| Task | SVG content |
|------|-------------|
| `detection` / `open_vocab` | Colored `<rect>` boxes + `<text>` `"class conf%"` labels |
| `segmentation` | Colored `<rect>` bbox outlines + `<text>` confidence labels |
| `classification` | Stacked `<text>` lines with top-K `"class conf%"` |
| `depth` / `embed` | Empty `<svg>` (no meaningful pixel annotation) |

## Other API

```ts
await client.health();      // { status: "ok" }
await client.listModels();  // ModelInfo[]  (name, task, license, state)
await client.ps();          // only loaded models
await client.load("rf-detr");
await client.unload("rf-detr");
```

## Develop

```bash
npm install
npm run build      # tsc → dist/
npm test           # node:test with a mocked fetch (no server needed)
npm run typecheck  # tsc --noEmit
```
