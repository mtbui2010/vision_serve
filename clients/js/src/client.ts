/**
 * HTTP client for the VisionServe server.
 *
 * Transport uses only built-in globals (`fetch`, `FormData`, `Blob`), available on
 * Node >= 18 and in modern browsers — so the client has zero runtime dependencies.
 * Reading an image from a file path uses `node:fs/promises`, imported lazily so the
 * client still works in the browser when you pass bytes / a Blob instead.
 */

import { ModelInfo, Result } from "./types.js";
import { filterBySize as _filterBySize, type SizeFilterOptions } from "./filter.js";

/** Accepted image inputs for {@link Client.predict}. */
export type ImageInput = string | Uint8Array | ArrayBuffer | Blob;

/** A box is `[x, y, w, h]`. Pass one box or a list of boxes. */
export type BoxInput = number[] | number[][];
/** A point is `[x, y]` or `[x, y, label]` (label 1=fg, 0=bg). One or a list. */
export type PointInput = number[] | number[][];

/**
 * A depth map for {@link PredictOptions.depth}: a `Uint16Array` (sent as `uint16`, e.g.
 * millimetres from an RGB-D camera; 0 = no reading) or a `Float32Array` (sent as `float32`;
 * values <= 0 or NaN = no reading), row-major. Raw little-endian bytes (`Uint8Array`,
 * `ArrayBuffer`, `Blob`) are sent as they are and need `depthDtype`.
 */
export type DepthInput = Uint16Array | Float32Array | Uint8Array | ArrayBuffer | Blob;

/**
 * Options for {@link Client.predict}. Each one is sent as the server form field of the same
 * name in snake_case (`boxThreshold` -> `box_threshold`), the fields of
 * `api.PredictJSONRequest` in `pkg/api/types.go`. Leave an option out (or `undefined` /
 * `null`) for the server's default.
 */
export interface PredictOptions {
  /**
   * Free-text open-vocab prompt, e.g. `"cat. remote."`. Normalised like the Python SDK (see
   * {@link normalizePrompt}): `,` and `|` become `.`, a single phrase gets a trailing `.`, and
   * GroundingDINO-family models get `"object."` when no prompt is given.
   */
  prompt?: string | null;
  /** SAM box prompt(s): `[x,y,w,h]` or a list of them, in ORIGINAL image coords. */
  box?: BoxInput | null;
  /** SAM point prompt(s): `[x,y]`/`[x,y,label]` or a list, in ORIGINAL image coords. */
  point?: PointInput | null;
  /** `box_threshold`: GroundingDINO query-score threshold. */
  boxThreshold?: number | null;
  /** `text_threshold`: GroundingDINO second score floor (does not change labels). */
  textThreshold?: number | null;
  /** `bg_max_area`: `background` (sam/automask): a mask >= this % of the image is background. */
  bgMaxArea?: number | null;
  /** `fg_min_area`: `background` (sam/automask): a mask < this % of the image is noise. */
  fgMinArea?: number | null;
  /** `grid_size`: MobileSAM automask grid N (N x N decoder calls; the server caps it at 64). Integer. */
  gridSize?: number | null;
  /** `method`: algorithm for models that offer several (`background`: auto|depth|sam|cv|automask; textalign: exact|dual). */
  method?: string | null;
  /** `roi`: `[x, y, w, h]` region of interest, in pixels or as 0..1 fractions (when w, h <= 1). */
  roi?: number[] | null;
  /** `dilate`: grow (> 0) or shrink (< 0) every output mask by |n| pixels. Integer. */
  dilate?: number | null;
  /** `depth`: an aligned depth map (see {@link DepthInput}); read by `background` (depth / auto). */
  depth?: DepthInput | null;
  /** `depth_dtype`: inferred for `Uint16Array` / `Float32Array`; required for raw bytes. */
  depthDtype?: "uint16" | "float32" | null;
  /** `depth_width`: the depth map's width (default: the image's; give both or neither). */
  depthWidth?: number | null;
  /** `depth_height`: the depth map's height (default: the image's). */
  depthHeight?: number | null;
  /** `min_size`: drop objects whose bbox area is below this % of the image (server side). */
  minSize?: number | null;
  /** `max_size`: drop objects whose bbox area is above this % of the image (server side). */
  maxSize?: number | null;
  /** `gripper_min`: grasp models: smallest jaw opening, ORIGINAL-image pixels. */
  gripperMin?: number | null;
  /** `gripper_max`: grasp models: largest jaw opening, ORIGINAL-image pixels. */
  gripperMax?: number | null;
  /** `claim_threshold`: `rfdetr-textalign*` with `method: "dual"`; >= 1 never claims. */
  claimThreshold?: number | null;
  /** `crop_temp`: softmax temperature of the crop namer (lower = more decisive). */
  cropTemp?: number | null;
  /** `template_name`: `instance_detection` models: a set registered via `POST /api/templates`. */
  templateName?: string | null;
}

/** Number options and the server field each one is sent as. */
const NUMBER_FIELDS = {
  boxThreshold: "box_threshold",
  textThreshold: "text_threshold",
  bgMaxArea: "bg_max_area",
  fgMinArea: "fg_min_area",
  minSize: "min_size",
  maxSize: "max_size",
  gripperMin: "gripper_min",
  gripperMax: "gripper_max",
  claimThreshold: "claim_threshold",
  cropTemp: "crop_temp",
} as const;
/** Options the server reads with Atoi: `"2.5"` would silently become 0 (= default / off). */
const INT_FIELDS = { gridSize: "grid_size", dilate: "dilate" } as const;
const STRING_FIELDS = { method: "method", templateName: "template_name" } as const;
const OTHER_OPTIONS = ["prompt", "box", "point", "roi", "depth", "depthDtype", "depthWidth", "depthHeight"];
const KNOWN_OPTIONS = new Set<string>([
  ...Object.keys(NUMBER_FIELDS),
  ...Object.keys(INT_FIELDS),
  ...Object.keys(STRING_FIELDS),
  ...OTHER_OPTIONS,
]);

/** Options for constructing a {@link Client}. */
export interface ClientOptions {
  /** Per-request timeout in milliseconds (default 120000). */
  timeoutMs?: number;
  /**
   * Ask the server for depth maps and embeddings as base64 float32 (`encoding=base64`) instead
   * of JSON number arrays: about half the bytes and much cheaper to parse for a large depth map.
   * The SDK decodes them, so `Result.depthMap` / `embeddings` stay plain `number[]` holding the
   * exact float32 values. Default `false`. A server that predates the option ignores it.
   */
  base64Arrays?: boolean;
}

/** Raised when the server returns a non-2xx response or transport fails. */
export class VisionServeError extends Error {
  /** HTTP status, or `undefined` when no answer came back (unreachable, timeout, dropped). */
  readonly status?: number;
  /**
   * The server's `Retry-After` in seconds (sent with a 503 when the model's queue is full),
   * else `undefined`.
   */
  readonly retryAfter?: number;
  constructor(message: string, status?: number, retryAfter?: number) {
    super(message);
    this.name = "VisionServeError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

export class Client {
  readonly host: string;
  readonly timeoutMs: number;
  readonly base64Arrays: boolean;

  /**
   * @param host base URL of the server, e.g. `http://127.0.0.1:11435`.
   */
  constructor(host = "http://127.0.0.1:11435", opts: ClientOptions = {}) {
    this.host = host.replace(/\/+$/, "");
    this.timeoutMs = opts.timeoutMs ?? 120_000;
    this.base64Arrays = Boolean(opts.base64Arrays);
  }

  // ------------------------------------------------------------------ //
  // Public API
  // ------------------------------------------------------------------ //

  /** `GET /api/health` -> `{ status: "ok" }`. */
  async health(): Promise<{ status: string }> {
    return (await this.getJSON("/api/health")) as { status: string };
  }

  /** `GET /api/models` -> list of {@link ModelInfo}. */
  async listModels(): Promise<ModelInfo[]> {
    const data = (await this.getJSON("/api/models")) as unknown[] | null;
    return (data ?? []).map((x) => ModelInfo.fromJSON(x as Record<string, unknown>));
  }

  /** `POST /api/load` -> `{ model, state }`. */
  async load(model: string): Promise<Record<string, string>> {
    return (await this.postJSON("/api/load", { model })) as Record<string, string>;
  }

  /** `POST /api/unload` -> `{ model, state }`. */
  async unload(model: string): Promise<Record<string, string>> {
    return (await this.postJSON("/api/unload", { model })) as Record<string, string>;
  }

  /** Return only the currently loaded models (filtered from `/api/models`). */
  async ps(): Promise<ModelInfo[]> {
    return (await this.listModels()).filter((m) => m.isLoaded);
  }

  /**
   * `POST /api/predict` (multipart) -> {@link Result}.
   *
   * @param model model name (must be loaded, or the server may auto-load it).
   * @param image one of: a file path (`string`), raw encoded bytes
   *   (`Uint8Array`/`ArrayBuffer`), or a `Blob`.
   * @param opts  see {@link PredictOptions}; coordinates are in ORIGINAL image pixels.
   * @throws TypeError / Error for an unknown option or a malformed value, before anything is sent.
   */
  async predict(model: string, image: ImageInput, opts: PredictOptions = {}): Promise<Result> {
    const form = buildPredictForm(model, opts, this.base64Arrays);
    const { blob, filename } = await toBlob(image);
    form.append("image", blob, filename);
    const depth = depthBlob(opts);
    if (depth) form.append("depth", depth, "depth.bin");

    const data = await this.request("POST", "/api/predict", form);
    return Result.fromJSON((data ?? {}) as Record<string, unknown>);
  }

  /** Filter detections/masks by bounding-box size. */
  filterBySize(result: Result, opts: SizeFilterOptions): Result {
    return _filterBySize(result, opts);
  }

  // ------------------------------------------------------------------ //
  // Transport
  // ------------------------------------------------------------------ //
  private getJSON(path: string): Promise<unknown> {
    return this.request("GET", path);
  }

  private postJSON(path: string, payload: unknown): Promise<unknown> {
    return this.request("POST", path, JSON.stringify(payload), "application/json");
  }

  private async request(
    method: string,
    path: string,
    body?: BodyInit,
    contentType?: string,
  ): Promise<unknown> {
    const url = this.host + path;
    const headers: Record<string, string> = { Accept: "application/json" };
    // For FormData, let fetch set the multipart boundary itself — don't set Content-Type.
    if (contentType) headers["Content-Type"] = contentType;

    // The timeout covers the whole exchange: connecting, the upload, the server's work and
    // reading the answer's body.
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeoutMs);
    const transportError = (e: unknown, what: string): VisionServeError => {
      if (timedOut) {
        return new VisionServeError(
          `${method} ${path}: timed out after ${this.timeoutMs} ms waiting for VisionServe at ${this.host}`,
        );
      }
      const reason = e instanceof Error ? describeCause(e) : String(e);
      return new VisionServeError(`${what}: ${reason}`);
    };

    let resp: Response;
    let raw: string;
    try {
      try {
        resp = await fetch(url, { method, headers, body, signal: controller.signal });
      } catch (e) {
        throw transportError(e, `failed to reach VisionServe at ${url}`);
      }
      try {
        raw = await resp.text();
      } catch (e) {
        throw transportError(e, `${method} ${path}: connection to VisionServe at ${this.host} failed`);
      }
    } finally {
      clearTimeout(timer);
    }

    if (!resp.ok) {
      const message = extractError(raw) || resp.statusText || "HTTP error";
      throw new VisionServeError(
        `${method} ${path} -> ${resp.status}: ${message}`,
        resp.status,
        parseRetryAfter(resp.headers.get("Retry-After")),
      );
    }
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (e) {
      throw new VisionServeError(`invalid JSON response from ${url}: ${e}`);
    }
  }
}

// ---------------------------------------------------------------------- //
// Request building
// ---------------------------------------------------------------------- //

/**
 * The multipart text fields {@link Client.predict} sends (everything but the image and depth
 * parts). Validates every option first, so a mistake throws before anything is read or sent.
 */
function buildPredictForm(model: string, opts: PredictOptions = {}, base64Arrays = false): FormData {
  for (const key of Object.keys(opts)) {
    if (!KNOWN_OPTIONS.has(key)) {
      const camel = key.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
      const hint = KNOWN_OPTIONS.has(camel) ? ` (did you mean "${camel}"?)` : "";
      throw new TypeError(`predict(): unknown option "${key}"${hint}`);
    }
  }
  const form = new FormData();
  form.append("model", model);
  const prompt = normalizePrompt(model, opts.prompt);
  if (prompt != null) form.append("prompt", prompt);
  const boxStr = serializeBoxes(opts.box ?? undefined);
  if (boxStr) form.append("box", boxStr);
  if (opts.roi != null) {
    if (!isScalarSeq(opts.roi) || opts.roi.length !== 4) {
      throw new Error(`roi must be one box [x,y,w,h], got ${JSON.stringify(opts.roi)}`);
    }
    form.append("roi", opts.roi.map((v) => fmtNum("roi", v)).join(","));
  }
  const pointStr = serializePoints(opts.point ?? undefined);
  if (pointStr) form.append("point", pointStr);
  for (const [key, field] of Object.entries(NUMBER_FIELDS)) {
    const v = opts[key as keyof typeof NUMBER_FIELDS];
    if (v != null) form.append(field, fmtNum(key, v));
  }
  for (const [key, field] of Object.entries(INT_FIELDS)) {
    const v = opts[key as keyof typeof INT_FIELDS];
    if (v != null) form.append(field, String(asInt(key, v)));
  }
  for (const [key, field] of Object.entries(STRING_FIELDS)) {
    const v = opts[key as keyof typeof STRING_FIELDS];
    if (v != null) form.append(field, String(v));
  }
  if (opts.depth != null) {
    form.append("depth_dtype", depthDtype(opts));
    const { depthWidth: w, depthHeight: h } = opts;
    if ((w == null) !== (h == null)) {
      throw new Error("depthWidth and depthHeight go together: give both, or neither (= the image's size)");
    }
    if (w != null && h != null) {
      const dw = asInt("depthWidth", w);
      const dh = asInt("depthHeight", h);
      if (dw <= 0 || dh <= 0) throw new Error(`depthWidth/depthHeight must be > 0, got ${dw}x${dh}`);
      const d = opts.depth;
      if ((d instanceof Uint16Array || d instanceof Float32Array) && d.length !== dw * dh) {
        throw new Error(`depth has ${d.length} values but depthWidth*depthHeight = ${dw * dh}`);
      }
      form.append("depth_width", String(dw));
      form.append("depth_height", String(dh));
    }
  } else if (opts.depthDtype != null || opts.depthWidth != null || opts.depthHeight != null) {
    throw new Error("depthDtype / depthWidth / depthHeight need a depth map (opts.depth)");
  }
  if (base64Arrays) form.append("encoding", "base64");
  return form;
}

function depthDtype(opts: PredictOptions): "uint16" | "float32" {
  const d = opts.depth;
  const inferred = d instanceof Uint16Array ? "uint16" : d instanceof Float32Array ? "float32" : null;
  const given = opts.depthDtype ?? null;
  if (given != null && given !== "uint16" && given !== "float32") {
    throw new Error(`depthDtype must be "uint16" or "float32", got ${JSON.stringify(given)}`);
  }
  if (inferred && given && inferred !== given) {
    throw new Error(`depthDtype "${given}" contradicts the depth array (${inferred})`);
  }
  const dtype = inferred ?? given;
  if (!dtype) throw new Error('raw depth bytes need depthDtype: "uint16" or "float32"');
  return dtype;
}

/** The depth part to upload, or null. Typed arrays are sent little-endian. */
function depthBlob(opts: PredictOptions): Blob | null {
  const d = opts.depth;
  if (d == null) return null;
  if (d instanceof Blob) return d;
  let bytes: Uint8Array;
  if (d instanceof Uint16Array || d instanceof Float32Array) {
    bytes = new Uint8Array(d.length * d.BYTES_PER_ELEMENT);
    const view = new DataView(bytes.buffer);
    if (d instanceof Uint16Array) d.forEach((v, i) => view.setUint16(i * 2, v, true));
    else d.forEach((v, i) => view.setFloat32(i * 4, v, true));
  } else if (d instanceof Uint8Array || d instanceof ArrayBuffer) {
    bytes = d instanceof ArrayBuffer ? new Uint8Array(d) : d;
  } else {
    throw new TypeError("depth must be a Uint16Array, Float32Array, Uint8Array, ArrayBuffer or Blob");
  }
  return new Blob([bytes as unknown as BlobPart], { type: "application/octet-stream" });
}

// ---------------------------------------------------------------------- //
// Image encoding
// ---------------------------------------------------------------------- //
async function toBlob(image: ImageInput): Promise<{ blob: Blob; filename: string }> {
  // BlobPart's lib typings pin Uint8Array to a plain ArrayBuffer backing store; our
  // bytes may be backed by ArrayBufferLike (Node Buffer, SharedArrayBuffer), so we
  // funnel everything through this cast in one place.
  const bytesBlob = (b: Uint8Array | ArrayBuffer) =>
    new Blob([b as unknown as BlobPart], { type: "application/octet-stream" });

  if (typeof image === "string") {
    // File path — read via node:fs (lazy import so browsers can still use bytes/Blob).
    const { readFile } = await import("node:fs/promises");
    const path = await import("node:path");
    const buf = await readFile(image);
    return { blob: bytesBlob(buf), filename: path.basename(image) };
  }
  if (image instanceof Blob) {
    return { blob: image, filename: "image.png" };
  }
  if (image instanceof Uint8Array || image instanceof ArrayBuffer) {
    return { blob: bytesBlob(image), filename: "image.png" };
  }
  throw new TypeError("unsupported image type; expected path string, Uint8Array, ArrayBuffer, or Blob");
}

// ---------------------------------------------------------------------- //
// Prompt normalisation (same rule as the Python SDK's normalize_prompt)
// ---------------------------------------------------------------------- //

// Models that REQUIRE a text prompt (the server rejects an empty one): GroundingDINO and the
// pipelines built on it. "gdino-siglip" (and -sam) runs GroundingDINO first; rfdetr-gdino-* does
// NOT need a prompt (no prompt = everything RF-DETR knows), so it is excluded.
const OPEN_VOCAB_MODELS = ["grounding-dino", "grounded-sam", "grasp-gd"];
const OPEN_VOCAB_RE = /(?<!rfdetr-)gdino-siglip/;
// CLIP / SigLIP text and image towers: each "."-separated phrase is one label, and a comma is
// part of the label ("a photo of a cat, sitting"), so commas must not become phrase separators.
const EMBED_RE = /clip|siglip/;
const DETECTOR_RE = /gdino|grounding|grounded|textalign|grasp/;

/**
 * The prompt string {@link Client.predict} sends for `model`, or `null` when no prompt field is
 * sent. The same rule as the Python SDK's `normalize_prompt` (shared fixtures in
 * `clients/testdata/normalize_prompt.json`):
 *
 * - empty / missing: `"object."` for models that require a prompt (GroundingDINO family), else `null`;
 * - CLIP / SigLIP towers: sent verbatim (the server splits phrases on `"."` only);
 * - everything else: `","` and `"|"` become the `"."` phrase separator and a single phrase gets a
 *   trailing `"."` (`"cat, remote"` -> `"cat. remote"`, `"cat"` -> `"cat."`).
 */
export function normalizePrompt(model: string, prompt: string | null | undefined): string | null {
  const m = String(model).toLowerCase();
  if (prompt == null || !String(prompt).trim()) {
    const openVocab = OPEN_VOCAB_MODELS.some((k) => m.includes(k)) || OPEN_VOCAB_RE.test(m);
    return openVocab ? "object." : null;
  }
  let text = String(prompt);
  if (EMBED_RE.test(m) && !DETECTOR_RE.test(m)) return text;
  text = text.replace(/[,|]/g, ".");
  if (!text.includes(".")) text += ".";
  return text;
}

// ---------------------------------------------------------------------- //
// Box / point / number serialization (server string formats)
// ---------------------------------------------------------------------- //
function isScalarSeq(seq: unknown): seq is number[] {
  return Array.isArray(seq) && seq.length > 0 && seq.every((v) => typeof v === "number");
}

function normalizeList(values: number[] | number[][] | undefined): number[][] {
  if (values == null) return [];
  if (isScalarSeq(values)) return [values]; // a single box/point
  return values as number[][]; // already a list of boxes/points
}

function serializeBoxes(box: BoxInput | undefined): string {
  return normalizeList(box)
    .map((b) => {
      if (!Array.isArray(b) || b.length !== 4) {
        throw new Error(`box must have 4 values [x,y,w,h], got ${JSON.stringify(b)}`);
      }
      return b.map((v) => fmtNum("box", v)).join(",");
    })
    .join(";");
}

function serializePoints(point: PointInput | undefined): string {
  return normalizeList(point)
    .map((p) => {
      if (!Array.isArray(p) || (p.length !== 2 && p.length !== 3)) {
        throw new Error(`point must have 2 or 3 values [x,y[,label]], got ${JSON.stringify(p)}`);
      }
      return p.map((v) => fmtNum("point", v)).join(",");
    })
    .join(";");
}

/**
 * Format a number for the server. JS `String()` already drops a trailing `.0`
 * (`String(1.0) === "1"`, `String(2.5) === "2.5"`), matching the Python client. NaN and
 * infinities are refused (the server rejects them).
 */
function fmtNum(name: string, v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new TypeError(`${name} must be a finite number, got ${JSON.stringify(v) ?? String(v)}`);
  }
  return String(v);
}

/** An integer form field (the server uses Atoi): `2.5` is refused rather than read as 0. */
function asInt(name: string, v: unknown): number {
  if (typeof v !== "number" || !Number.isInteger(v)) {
    throw new TypeError(`${name} must be an integer, got ${JSON.stringify(v) ?? String(v)}`);
  }
  return v;
}

/** `Retry-After` as seconds (the server sends an integer), or undefined. */
function parseRetryAfter(value: string | null): number | undefined {
  if (value == null || !value.trim()) return undefined;
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : undefined;
}

/** An Error's message plus its `cause` (Node's fetch puts ECONNREFUSED etc. there). */
function describeCause(e: Error): string {
  const cause = (e as { cause?: unknown }).cause;
  if (cause instanceof Error && cause.message && !e.message.includes(cause.message)) {
    return `${e.message} (${cause.message})`;
  }
  return e.message;
}

function extractError(raw: string): string | null {
  if (!raw) return null;
  try {
    const d = JSON.parse(raw);
    if (d && typeof d === "object" && "error" in d) return String((d as Record<string, unknown>).error);
    return null;
  } catch {
    return raw;
  }
}
