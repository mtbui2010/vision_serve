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

/** Optional prompt inputs for {@link Client.predict}. */
export interface PredictOptions {
  /** Free-text open-vocab prompt, e.g. `"cat. remote."`. */
  prompt?: string;
  /** SAM box prompt(s): `[x,y,w,h]` or a list of them, in ORIGINAL image coords. */
  box?: BoxInput;
  /** SAM point prompt(s): `[x,y]`/`[x,y,label]` or a list, in ORIGINAL image coords. */
  point?: PointInput;
}

/** Options for constructing a {@link Client}. */
export interface ClientOptions {
  /** Per-request timeout in milliseconds (default 120000). */
  timeoutMs?: number;
}

/** Raised when the server returns a non-2xx response or transport fails. */
export class VisionServeError extends Error {
  readonly status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "VisionServeError";
    this.status = status;
  }
}

export class Client {
  readonly host: string;
  readonly timeoutMs: number;

  /**
   * @param host base URL of the server, e.g. `http://localhost:11435`.
   */
  constructor(host = "http://localhost:11435", opts: ClientOptions = {}) {
    this.host = host.replace(/\/+$/, "");
    this.timeoutMs = opts.timeoutMs ?? 120_000;
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
   * @param opts  optional `prompt` / `box` / `point` (in ORIGINAL image coords).
   */
  async predict(model: string, image: ImageInput, opts: PredictOptions = {}): Promise<Result> {
    const { blob, filename } = await toBlob(image);

    const form = new FormData();
    form.append("model", model);
    if (opts.prompt != null && String(opts.prompt).trim()) {
      form.append("prompt", String(opts.prompt));
    }
    const boxStr = serializeBoxes(opts.box);
    if (boxStr) form.append("box", boxStr);
    const pointStr = serializePoints(opts.point);
    if (pointStr) form.append("point", pointStr);
    form.append("image", blob, filename);

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

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    let resp: Response;
    try {
      resp = await fetch(url, { method, headers, body, signal: controller.signal });
    } catch (e) {
      const reason = e instanceof Error ? e.message : String(e);
      throw new VisionServeError(`failed to reach VisionServe at ${url}: ${reason}`);
    } finally {
      clearTimeout(timer);
    }

    const raw = await resp.text();
    if (!resp.ok) {
      const message = extractError(raw) || resp.statusText || "HTTP error";
      throw new VisionServeError(`${method} ${path} -> ${resp.status}: ${message}`, resp.status);
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
// Prompt / box / point serialization (server string formats)
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
      if (b.length !== 4) throw new Error(`box must have 4 values [x,y,w,h], got ${JSON.stringify(b)}`);
      return b.map(fmtNum).join(",");
    })
    .join(";");
}

function serializePoints(point: PointInput | undefined): string {
  return normalizeList(point)
    .map((p) => {
      if (p.length !== 2 && p.length !== 3) {
        throw new Error(`point must have 2 or 3 values [x,y[,label]], got ${JSON.stringify(p)}`);
      }
      return p.map(fmtNum).join(",");
    })
    .join(";");
}

/**
 * Format a number for the server. JS `String()` already drops a trailing `.0`
 * (`String(1.0) === "1"`, `String(2.5) === "2.5"`), matching the Python client.
 */
function fmtNum(v: number): string {
  return String(v);
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
