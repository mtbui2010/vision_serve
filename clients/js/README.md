# visionserve (JavaScript / TypeScript client)

A lightweight TypeScript/JavaScript **client** SDK for the [VisionServe](../../) HTTP
server. It talks to the Go runtime over REST — it does **not** run inference itself.

It is the sibling of the [Python client](../python/) and mirrors its API. Transport uses
only built-in globals (`fetch`, `FormData`, `Blob`), so it has **zero runtime
dependencies** and runs on **Node >= 18** and in modern browsers. (Passing a file *path*
to `predict()` uses `node:fs` and is Node-only; in the browser pass bytes or a `Blob`.)

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

// Detection — RF-DETR or RT-DETR
const det = await client.predict("rf-detr", "image.jpg");
for (const d of det.detections) {
  console.log(d.cls, d.conf.toFixed(3), d.bbox); // bbox = [x, y, w, h] in original pixels
}

// Segmentation — MobileSAM / EfficientSAM / SAM2 with a box prompt (original-image coords)
const seg = await client.predict("mobile-sam", "image.jpg", { box: [34, 58, 120, 240] });
const flat = seg.masks[0]?.toMask(640, 480); // row-major Uint8Array, 1 = inside mask

// Open-vocab segmentation — Grounded-SAM (text → boxes → masks)
const gs = await client.predict("grounded-sam", "image.jpg", { prompt: "cat. remote." });
console.log(gs.detections.map((d) => d.cls), "→", gs.masks.length, "masks");

// Depth estimation
const dep = await client.predict("depth-anything-v2", "image.jpg");
// dep.depthMap is a Float32Array of length dep.depthWidth * dep.depthHeight

// Classification — top-K ImageNet predictions
const cls = await client.predict("efficientnet-b0", "image.jpg");
for (const c of cls.classifications) {
  console.log(c.cls, c.conf.toFixed(3));
}
```

### Image inputs

`predict(model, image, opts?)` accepts:

| Input | Notes |
|-------|-------|
| `string` | path to an image file on disk (Node only) |
| `Uint8Array` / `ArrayBuffer` | already-encoded image bytes (PNG/JPEG) |
| `Blob` | e.g. from a browser `<input type="file">` or `fetch` |

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
                                 // "depth" | "classification" | "embed" | ...
  model: string;
  detections: Detection[];       // { bbox: [x,y,w,h], cls: string, conf: number }
  masks: Mask[];                 // { rle, bbox, conf } — column-major RLE
  classifications: Classification[]; // { cls: string, conf: number } — top-K
  depthMap: number[];            // flat row-major float array, size depthWidth×depthHeight
  depthWidth: number;
  depthHeight: number;
  embeddings: number[][];        // one embedding vector per image
  durationMs: number;
}
```

`Mask.toMask(width, height)` decodes the column-major RLE into a row-major `Uint8Array`
(`1` = inside the mask); `Mask.toMask2D(width, height)` returns a `boolean[][]`. Pass the
**original** image width/height the mask was produced against.

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

// Absolute mode — area in pixels²
const big = filterBySize(res, { minSize: 5000 });
const mid = filterBySize(res, { minSize: 500, maxSize: 50000 });

// Relative mode — fraction of image area (0.0–1.0), supply imageWidth + imageHeight
const rel = filterBySize(res, {
  minSize: 0.01,     // at least 1% of image area
  maxSize: 0.5,      // at most 50% of image area
  imageWidth: 1280,
  imageHeight: 720,
});

// Via Client method:
const filtered = client.filterBySize(res, { minSize: 500 });
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
