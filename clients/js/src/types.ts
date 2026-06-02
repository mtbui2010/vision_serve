/**
 * Result types for the VisionServe client.
 *
 * These mirror the server's unified wire schema (see `pkg/api/types.go`). The schema
 * is the SAME across every task — detection, segmentation, open-vocab — so there is a
 * single {@link Result} type rather than one per model.
 */

/** Task kind reported by the server. Open-ended on purpose (new tasks may appear). */
export type Task = "detection" | "segmentation" | "open_vocab" | (string & {});

/** Lifecycle state of a model in the registry. */
export type ModelState = "not_downloaded" | "available" | "loaded" | (string & {});

/** A single detected object. `bbox` is `[x, y, w, h]` in ORIGINAL image pixels. */
export class Detection {
  /** `[x, y, w, h]` (top-left corner + width/height) in ORIGINAL image pixels. */
  readonly bbox: number[];
  /** Class label string. */
  readonly cls: string;
  /** Confidence in `[0, 1]`. */
  readonly conf: number;

  constructor(bbox: number[], cls: string, conf: number) {
    this.bbox = bbox;
    this.cls = cls;
    this.conf = conf;
  }

  static fromJSON(d: Record<string, unknown>): Detection {
    const bbox = Array.isArray(d.bbox) ? (d.bbox as unknown[]).map(Number) : [0, 0, 0, 0];
    // Wire field is `class` (a reserved word in JS), exposed here as `cls`.
    return new Detection(bbox, String(d["class"] ?? ""), Number(d.conf ?? 0));
  }
}

/** A segmentation mask, encoded as column-major (COCO-style) uncompressed RLE. */
export class Mask {
  /**
   * COCO-style COLUMN-MAJOR uncompressed RLE — space-separated integer run counts,
   * starting with a background (0) run, read column-major (column outer, row inner)
   * over the ORIGINAL image `H x W`.
   */
  readonly rle: string;
  /** `[x, y, w, h]` bounding box of the mask in ORIGINAL image pixels. */
  readonly bbox: number[];
  /** Confidence (e.g. predicted IoU) in `[0, 1]`. */
  readonly conf: number;

  constructor(rle: string, bbox: number[], conf: number) {
    this.rle = rle;
    this.bbox = bbox;
    this.conf = conf;
  }

  static fromJSON(d: Record<string, unknown>): Mask {
    const bbox = Array.isArray(d.bbox) ? (d.bbox as unknown[]).map(Number) : [0, 0, 0, 0];
    return new Mask(String(d.rle ?? ""), bbox, Number(d.conf ?? 0));
  }

  /**
   * Decode the column-major RLE into a row-major `Uint8Array` of length
   * `width * height`, where index `y * width + x` is `1` inside the mask, `0` outside.
   *
   * This is the exact inverse of the Go encoder `encodeRLEColumnMajor` (and matches
   * the Python client's `Mask.to_ndarray`): runs alternate starting from background
   * (0) and are laid out in column-major order — the i-th pixel in run order maps to
   * `x = floor(i / height)`, `y = i % height`.
   *
   * @param width  ORIGINAL image width (W) the mask was encoded against.
   * @param height ORIGINAL image height (H) the mask was encoded against.
   * @throws if the run counts do not sum to `width * height`.
   */
  toMask(width: number, height: number): Uint8Array {
    const total = width * height;
    const counts = this.rle.trim() ? this.rle.trim().split(/\s+/).map((c) => parseInt(c, 10)) : [];
    const sum = counts.reduce((a, b) => a + b, 0);
    if (sum !== total) {
      throw new Error(`RLE run counts sum to ${sum} but width*height = ${total}`);
    }

    const out = new Uint8Array(total);
    let idx = 0;
    let value = false; // runs start with background
    for (const c of counts) {
      if (value && c > 0) {
        // The run [idx, idx+c) is in COLUMN-MAJOR order: linear k -> (x = k / H, y = k % H).
        for (let k = idx; k < idx + c; k++) {
          const x = Math.floor(k / height);
          const y = k % height;
          out[y * width + x] = 1;
        }
      }
      idx += c;
      value = !value;
    }
    return out;
  }

  /** Convenience: decode into a `boolean[height][width]` 2D array (row-major). */
  toMask2D(width: number, height: number): boolean[][] {
    const flat = this.toMask(width, height);
    const rows: boolean[][] = [];
    for (let y = 0; y < height; y++) {
      const row: boolean[] = new Array(width);
      for (let x = 0; x < width; x++) {
        row[x] = flat[y * width + x] === 1;
      }
      rows.push(row);
    }
    return rows;
  }
}

/** Unified prediction result returned by `POST /api/predict`. */
export class Result {
  readonly task: Task;
  readonly model: string;
  readonly detections: Detection[];
  readonly masks: Mask[];
  readonly durationMs: number;

  constructor(task: Task, model: string, detections: Detection[], masks: Mask[], durationMs: number) {
    this.task = task;
    this.model = model;
    this.detections = detections;
    this.masks = masks;
    this.durationMs = durationMs;
  }

  static fromJSON(d: Record<string, unknown>): Result {
    const dets = Array.isArray(d.detections) ? d.detections : [];
    const masks = Array.isArray(d.masks) ? d.masks : [];
    return new Result(
      String(d.task ?? ""),
      String(d.model ?? ""),
      dets.map((x) => Detection.fromJSON(x as Record<string, unknown>)),
      masks.map((x) => Mask.fromJSON(x as Record<string, unknown>)),
      Number(d.duration_ms ?? 0),
    );
  }
}

/** An entry from `GET /api/models`. */
export class ModelInfo {
  readonly name: string;
  readonly task: Task;
  readonly license: string;
  readonly state: ModelState;

  constructor(name: string, task: Task, license: string, state: ModelState) {
    this.name = name;
    this.task = task;
    this.license = license;
    this.state = state;
  }

  static fromJSON(d: Record<string, unknown>): ModelInfo {
    return new ModelInfo(
      String(d.name ?? ""),
      String(d.task ?? ""),
      String(d.license ?? ""),
      String(d.state ?? ""),
    );
  }

  get isLoaded(): boolean {
    return this.state === "loaded";
  }
}
