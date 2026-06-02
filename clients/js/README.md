# visionserve (JavaScript / TypeScript client)

A lightweight TypeScript/JavaScript **client** SDK for the [VisionServe](../../) HTTP
server. It talks to the Go runtime over REST — it does **not** run inference itself.

It is the sibling of the [Python client](../python/) and mirrors its API. Transport uses
only built-in globals (`fetch`, `FormData`, `Blob`), so it has **zero runtime
dependencies** and runs on **Node >= 18** and in modern browsers. (Passing a file *path*
to `predict()` uses `node:fs` and is Node-only; in the browser pass bytes or a `Blob`.)

## Install

```bash
npm install visionserve          # once published to npm
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

// Detection
const det = await client.predict("rf-detr", "image.jpg");
for (const d of det.detections) {
  console.log(d.cls, d.conf.toFixed(3), d.bbox); // bbox = [x, y, w, h] in original pixels
}

// Segmentation — MobileSAM with a box prompt (original-image coords)
const seg = await client.predict("mobile-sam", "image.jpg", { box: [34, 58, 120, 240] });
const flat = seg.masks[0]?.toMask(640, 480); // row-major Uint8Array, 1 = inside mask

// Open-vocab segmentation — Grounded-SAM (text → boxes → masks)
const gs = await client.predict("grounded-sam", "image.jpg", { prompt: "cat. remote." });
console.log(gs.detections.map((d) => d.cls), "→", gs.masks.length, "masks");
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
  task: string;            // "detection" | "segmentation" | "open_vocab" | ...
  model: string;
  detections: Detection[]; // { bbox: [x,y,w,h], cls: string, conf: number }
  masks: Mask[];           // { rle, bbox, conf } — column-major RLE
  durationMs: number;
}
```

`Mask.toMask(width, height)` decodes the column-major RLE into a row-major `Uint8Array`
(`1` = inside the mask); `Mask.toMask2D(width, height)` returns a `boolean[][]`. Pass the
**original** image width/height the mask was produced against.

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
