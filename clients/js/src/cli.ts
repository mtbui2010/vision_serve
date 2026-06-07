#!/usr/bin/env node
/**
 * Command-line interface for the VisionServe TypeScript/JavaScript client.
 *
 * This is a thin CLI over {@link Client} — it performs NO inference itself, it
 * talks to a running VisionServe server (the Go runtime) over HTTP (default
 * `http://localhost:11435`). Start the server first with `visionserve serve`
 * (the Go binary), then use this CLI to drive it from Node.
 *
 * Installed as the `visionserve` command (see `package.json` "bin"):
 *
 *   npx visionserve predict rf-detr cat.jpg
 *   npx visionserve predict grounding-dino kitchen.jpg --prompt "cat. remote."
 *   npx visionserve predict mobile-sam dog.jpg --box 50,40,200,180 --save
 *   npx visionserve list
 *   npx visionserve ps
 *
 * Design notes:
 *   - `predict` prints the result JSON to **stdout** (pipe-friendly) and a
 *     one-line summary (model/task/device + timings) to **stderr**.
 *   - The reported duration splits into `client` (wall-clock around the
 *     `predict()` HTTP round-trip) and `server` (the `duration_ms` the server
 *     measured for inference only). Both are captured BEFORE the SVG is built or
 *     saved, so visualization cost never inflates the reported latency.
 *   - `--save` writes an annotated **SVG** with an auto name
 *     `<stem>.js.<model>.<task>.svg` (the source image is embedded so the SVG is
 *     viewable standalone). `--save-as PATH` overrides the name.
 */

import { parseArgs } from "node:util";
import { readFile, writeFile } from "node:fs/promises";
import * as path from "node:path";

import { Client, VisionServeError } from "./client.js";
import { Result, ModelInfo } from "./types.js";
import { toSVG } from "./visualize.js";

const CLIENT_TYPE = "js";
const DEFAULT_HOST = "http://localhost:11435";

const USAGE = `visionserve — VisionServe JavaScript client CLI

Drive a running VisionServe server over HTTP (run \`visionserve serve\` first;
this CLI does no inference itself).

Usage:
  visionserve <command> [options]

Commands:
  predict <model> <image>   run inference; print result JSON to stdout
  list                      list models in the registry (alias: models, ls)
  ps                        list models currently loaded in server memory
  load <model>              load a model into server memory
  unload <model>            unload a model from server memory (alias: rm)
  health                    check that the server is reachable

Global options:
  --host <url>      server base URL (default ${DEFAULT_HOST})
  --timeout <sec>   per-request timeout in seconds (default 120)
  -h, --help        show this help
  --version         print the client version

predict options:
  --prompt <text>   open-vocab text prompt, e.g. "cat. remote." (GroundingDINO / grasp-gd)
  --box <x,y,w,h>   SAM box prompt(s) in ORIGINAL image pixels (multiple separated by ';')
  --point <x,y[,l]> SAM point prompt(s); label 1=fg 0=bg (multiple separated by ';')
  --min-size <pct>  drop objects whose bbox area is below pct% of the image (client-side)
  --max-size <pct>  drop objects whose bbox area is above pct% of the image (client-side)
  --save            save an annotated SVG: <stem>.js.<model>.<task>.svg
  --save-as <path>  save the annotated SVG to this exact path (implies --save)
  --compact         print result JSON on a single line (default: pretty)
  --quiet           suppress the stderr summary line

list / ps options:
  --json            print as JSON instead of a table

Examples:
  visionserve predict rf-detr cat.jpg
  visionserve predict grounding-dino kitchen.jpg --prompt 'cat. remote.' --save
  visionserve predict mobile-sam dog.jpg --box 50,40,200,180 --save-as out.svg
  visionserve list --json
  visionserve ps --host http://10.0.0.5:11435
`;

// --------------------------------------------------------------------------- //
// Wire serialization (match pkg/api/types.go exactly: `class`, omitempty arrays)
// --------------------------------------------------------------------------- //
function resultToWire(res: Result): Record<string, unknown> {
  const out: Record<string, unknown> = { task: res.task, model: res.model };
  if (res.device) out.device = res.device;
  if (res.detections.length) {
    out.detections = res.detections.map((d) => ({ bbox: d.bbox, class: d.cls, conf: d.conf }));
  }
  if (res.masks.length) {
    out.masks = res.masks.map((m) => ({ rle: m.rle, bbox: m.bbox, conf: m.conf }));
  }
  if (res.grasps.length) {
    out.grasps = res.grasps.map((g) => {
      const item: Record<string, unknown> = {
        x: g.x,
        y: g.y,
        theta: g.theta,
        width: g.width,
        quality: g.quality,
      };
      if (g.cls) item.class = g.cls;
      if (g.conf) item.conf = g.conf;
      return item;
    });
  }
  if (res.classifications.length) {
    out.classifications = res.classifications.map((c) => ({ class: c.cls, conf: c.conf }));
  }
  if (res.embeddings.length) out.embeddings = res.embeddings;
  if (res.depthMap.length) {
    out.depth_map = res.depthMap;
    out.depth_width = res.depthWidth;
    out.depth_height = res.depthHeight;
  }
  out.duration_ms = res.durationMs;
  return out;
}

/** Build a self-describing output name: `<stem>.js.<model>.<task>.svg`. */
function autoName(imagePath: string, model: string, task: string, ext: string): string {
  const stem = path.basename(imagePath, path.extname(imagePath)) || "image";
  return `${stem}.${CLIENT_TYPE}.${model}.${task || "result"}.${ext}`;
}

// --------------------------------------------------------------------------- //
// Image-dimension sniffer (PNG + JPEG) — needed so the SVG overlay aligns with
// the image. Pure byte parsing, no native image library.
// --------------------------------------------------------------------------- //
function imageSize(buf: Uint8Array): { width: number; height: number; mime: string } | null {
  // PNG: 8-byte signature, then IHDR with width/height as big-endian uint32.
  if (
    buf.length >= 24 &&
    buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47
  ) {
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    return { width: dv.getUint32(16), height: dv.getUint32(20), mime: "image/png" };
  }
  // JPEG: starts with FFD8; scan segments for a Start-Of-Frame (SOFn) marker.
  if (buf.length >= 4 && buf[0] === 0xff && buf[1] === 0xd8) {
    let off = 2;
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    while (off + 9 < buf.length) {
      if (buf[off] !== 0xff) {
        off++;
        continue;
      }
      const marker = buf[off + 1] ?? 0;
      // SOF0..SOF15, excluding DHT(C4), JPG(C8), DAC(CC) — these carry frame dims.
      if (
        marker >= 0xc0 && marker <= 0xcf &&
        marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc
      ) {
        const height = dv.getUint16(off + 5);
        const width = dv.getUint16(off + 7);
        return { width, height, mime: "image/jpeg" };
      }
      // Skip this segment using its big-endian length.
      const segLen = dv.getUint16(off + 2);
      if (segLen < 2) break;
      off += 2 + segLen;
    }
  }
  return null;
}

/** Wrap the overlay SVG so the source image is embedded as a standalone background. */
function standaloneSVG(overlay: string, buf: Uint8Array, mime: string): string {
  const b64 = Buffer.from(buf).toString("base64");
  // Inject an <image> right after the opening <svg ...> tag.
  const gt = overlay.indexOf(">");
  if (gt < 0) return overlay;
  const head = overlay.slice(0, gt + 1);
  const body = overlay.slice(gt + 1);
  const image = `<image href="data:${mime};base64,${b64}" x="0" y="0" width="100%" height="100%"/>`;
  return head + image + body;
}

// --------------------------------------------------------------------------- //
// Prompt parsing
// --------------------------------------------------------------------------- //
function parseBoxes(spec: string | undefined): number[][] | undefined {
  if (!spec) return undefined;
  const boxes = spec
    .split(";")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((chunk) => {
      const vals = chunk.split(",").map((v) => Number(v));
      if (vals.length !== 4 || vals.some((v) => Number.isNaN(v))) {
        throw new Error(`box "${chunk}" must have 4 numbers x,y,w,h`);
      }
      return vals;
    });
  return boxes.length ? boxes : undefined;
}

function parsePoints(spec: string | undefined): number[][] | undefined {
  if (!spec) return undefined;
  const points = spec
    .split(";")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((chunk) => {
      const vals = chunk.split(",").map((v) => Number(v));
      if ((vals.length !== 2 && vals.length !== 3) || vals.some((v) => Number.isNaN(v))) {
        throw new Error(`point "${chunk}" must have 2 or 3 numbers x,y[,label]`);
      }
      return vals;
    });
  return points.length ? points : undefined;
}

// --------------------------------------------------------------------------- //
// Commands
// --------------------------------------------------------------------------- //
type Values = Record<string, string | boolean | undefined>;

async function cmdPredict(client: Client, positionals: string[], values: Values): Promise<number> {
  const model = positionals[0];
  const image = positionals[1];
  if (!model || !image) {
    process.stderr.write("usage: visionserve predict <model> <image>\n");
    return 2;
  }

  const buf = await readFile(image);
  const bytes = new Uint8Array(buf);

  // --- Inference: time ONLY the predict() round-trip (excludes SVG + save). ---
  const t0 = performance.now();
  let res = await client.predict(model, bytes, {
    prompt: values.prompt as string | undefined,
    box: parseBoxes(values.box as string | undefined),
    point: parsePoints(values.point as string | undefined),
  });
  const clientMs = performance.now() - t0;
  const serverMs = res.durationMs;

  // Optional client-side size filter (needs image dims).
  const dims = imageSize(bytes);
  const minSize = values["min-size"] != null ? Number(values["min-size"]) : undefined;
  const maxSize = values["max-size"] != null ? Number(values["max-size"]) : undefined;
  if ((minSize != null || maxSize != null) && dims) {
    res = client.filterBySize(res, {
      // CLI takes percent (e.g. 0.1 = 0.1%); filterBySize fractions are 0..1.
      minSize: minSize != null ? minSize / 100 : undefined,
      maxSize: maxSize != null ? maxSize / 100 : undefined,
      imageWidth: dims.width,
      imageHeight: dims.height,
    });
  }

  // stdout: result JSON (wire-faithful, pipe-friendly).
  const wire = resultToWire(res);
  process.stdout.write(
    (values.compact ? JSON.stringify(wire) : JSON.stringify(wire, null, 2)) + "\n",
  );

  // Optional annotated SVG (cost NOT counted in the durations above).
  let savedPath: string | null = null;
  if (values.save || values["save-as"]) {
    const outPath = (values["save-as"] as string) || autoName(image, model, res.task, "svg");
    if (!dims) {
      process.stderr.write(
        "warning: could not determine image size (only PNG/JPEG supported) — skipping --save\n",
      );
    } else {
      const overlay = toSVG(res, dims.width, dims.height);
      const svg = standaloneSVG(overlay, bytes, dims.mime);
      await writeFile(outPath, svg, "utf8");
      savedPath = outPath;
    }
  }

  if (!values.quiet) {
    const device = res.device || "?";
    process.stderr.write(
      `predict: model=${res.model || model} task=${res.task || "?"} device=${device}  ` +
        `client=${clientMs.toFixed(1)}ms server=${serverMs.toFixed(1)}ms  ${summaryCounts(res)}\n`,
    );
    if (savedPath) process.stderr.write(`saved: ${savedPath}\n`);
    if (res.grasps.length) {
      process.stderr.write("note: SVG overlay does not draw grasps; see the JSON for grasp data\n");
    }
  }
  return 0;
}

function summaryCounts(res: Result): string {
  const parts: string[] = [];
  if (res.detections.length) parts.push(`${res.detections.length} detections`);
  if (res.masks.length) parts.push(`${res.masks.length} masks`);
  if (res.grasps.length) parts.push(`${res.grasps.length} grasps`);
  if (res.classifications.length) parts.push(`${res.classifications.length} classes`);
  if (res.embeddings.length) parts.push(`${res.embeddings.length} embeddings`);
  if (res.depthMap.length) parts.push(`depth ${res.depthWidth}x${res.depthHeight}`);
  return parts.length ? `(${parts.join(", ")})` : "(no objects)";
}

async function cmdList(client: Client, values: Values): Promise<number> {
  const models = await client.listModels();
  if (values.json) {
    process.stdout.write(JSON.stringify(models.map(modelToObj), null, 2) + "\n");
  } else {
    printModelsTable(models);
  }
  return 0;
}

async function cmdPs(client: Client, values: Values): Promise<number> {
  const models = await client.ps();
  if (values.json) {
    process.stdout.write(JSON.stringify(models.map(modelToObj), null, 2) + "\n");
  } else {
    if (!models.length) process.stderr.write("no models loaded\n");
    printModelsTable(models);
  }
  return 0;
}

async function cmdLoad(client: Client, positionals: string[]): Promise<number> {
  const model = positionals[0];
  if (!model) {
    process.stderr.write("usage: visionserve load <model>\n");
    return 2;
  }
  process.stdout.write(JSON.stringify(await client.load(model)) + "\n");
  return 0;
}

async function cmdUnload(client: Client, positionals: string[]): Promise<number> {
  const model = positionals[0];
  if (!model) {
    process.stderr.write("usage: visionserve unload <model>\n");
    return 2;
  }
  process.stdout.write(JSON.stringify(await client.unload(model)) + "\n");
  return 0;
}

async function cmdHealth(client: Client): Promise<number> {
  process.stdout.write(JSON.stringify(await client.health()) + "\n");
  return 0;
}

function modelToObj(m: ModelInfo): Record<string, string> {
  return { name: m.name, task: m.task, license: m.license, state: m.state };
}

function printModelsTable(models: ModelInfo[]): void {
  if (!models.length) return;
  const w = (header: string, vals: string[]) => Math.max(header.length, ...vals.map((v) => v.length));
  const nameW = w("NAME", models.map((m) => m.name));
  const taskW = w("TASK", models.map((m) => m.task));
  const licW = w("LICENSE", models.map((m) => m.license));
  const pad = (s: string, n: number) => s.padEnd(n);
  process.stdout.write(
    `${pad("NAME", nameW)}  ${pad("TASK", taskW)}  ${pad("LICENSE", licW)}  STATE\n`,
  );
  for (const m of models) {
    process.stdout.write(
      `${pad(m.name, nameW)}  ${pad(m.task, taskW)}  ${pad(m.license, licW)}  ${m.state}\n`,
    );
  }
}

// --------------------------------------------------------------------------- //
// Entrypoint
// --------------------------------------------------------------------------- //
async function main(argv: string[]): Promise<number> {
  let parsed;
  try {
    parsed = parseArgs({
      args: argv,
      allowPositionals: true,
      strict: true,
      options: {
        host: { type: "string" },
        timeout: { type: "string" },
        prompt: { type: "string" },
        box: { type: "string" },
        point: { type: "string" },
        "min-size": { type: "string" },
        "max-size": { type: "string" },
        save: { type: "boolean" },
        "save-as": { type: "string" },
        compact: { type: "boolean" },
        quiet: { type: "boolean" },
        json: { type: "boolean" },
        help: { type: "boolean", short: "h" },
        version: { type: "boolean" },
      },
    });
  } catch (e) {
    process.stderr.write(`error: ${e instanceof Error ? e.message : String(e)}\n\n`);
    process.stderr.write(USAGE);
    return 2;
  }

  const { values, positionals } = parsed as { values: Values; positionals: string[] };

  if (values.version) {
    process.stdout.write("visionserve-client 0.1.2\n");
    return 0;
  }
  const command = positionals[0];
  if (values.help || !command) {
    process.stdout.write(USAGE);
    return command ? 0 : values.help ? 0 : 1;
  }

  const host = (values.host as string) || DEFAULT_HOST;
  const timeoutSec = values.timeout != null ? Number(values.timeout) : 120;
  const client = new Client(host, { timeoutMs: timeoutSec * 1000 });
  const rest = positionals.slice(1);

  try {
    switch (command) {
      case "predict":
      case "run":
        return await cmdPredict(client, rest, values);
      case "list":
      case "models":
      case "ls":
        return await cmdList(client, values);
      case "ps":
        return await cmdPs(client, values);
      case "load":
        return await cmdLoad(client, rest);
      case "unload":
      case "rm":
        return await cmdUnload(client, rest);
      case "health":
        return await cmdHealth(client);
      default:
        process.stderr.write(`unknown command: ${command}\n\n`);
        process.stdout.write(USAGE);
        return 2;
    }
  } catch (e) {
    if (e instanceof VisionServeError) {
      process.stderr.write(`error: ${e.message}\n`);
      return 1;
    }
    process.stderr.write(`error: ${e instanceof Error ? e.message : String(e)}\n`);
    return 1;
  }
}

main(process.argv.slice(2)).then(
  (code) => {
    process.exitCode = code;
  },
  (e) => {
    process.stderr.write(`fatal: ${e instanceof Error ? e.stack || e.message : String(e)}\n`);
    process.exitCode = 1;
  },
);
