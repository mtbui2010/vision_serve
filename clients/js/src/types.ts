/**
 * Result types for the VisionServe client.
 *
 * These mirror the server's unified wire schema (see `pkg/api/types.go`). The schema
 * is the SAME across every task — detection, segmentation, open-vocab — so there is a
 * single {@link Result} type rather than one per model.
 */

/** Task kind reported by the server. Open-ended on purpose (new tasks may appear). */
export type Task = "detection" | "segmentation" | "open_vocab" | "classification" | "depth" | "embed" | (string & {});

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

/** A single image classification prediction. */
export class Classification {
  /** Class label string. */
  readonly cls: string;
  /** Confidence in `[0, 1]`. */
  readonly conf: number;

  constructor(cls: string, conf: number) {
    this.cls = cls;
    this.conf = conf;
  }

  static fromJSON(d: Record<string, unknown>): Classification {
    // wire field is "class"
    return new Classification(String(d["class"] ?? ""), Number(d.conf ?? 0));
  }
}

/** Unified prediction result returned by `POST /api/predict`. */
export class Result {
  readonly task: Task;
  readonly model: string;
  readonly detections: Detection[];
  readonly masks: Mask[];
  readonly classifications: Classification[];
  /** Flat row-major float array for depth maps. */
  readonly depthMap: number[];
  readonly depthWidth: number;
  readonly depthHeight: number;
  /** One embedding vector per image. */
  readonly embeddings: number[][];
  readonly durationMs: number;

  constructor(
    task: Task,
    model: string,
    detections: Detection[],
    masks: Mask[],
    classifications: Classification[],
    depthMap: number[],
    depthWidth: number,
    depthHeight: number,
    embeddings: number[][],
    durationMs: number,
  ) {
    this.task = task;
    this.model = model;
    this.detections = detections;
    this.masks = masks;
    this.classifications = classifications;
    this.depthMap = depthMap;
    this.depthWidth = depthWidth;
    this.depthHeight = depthHeight;
    this.embeddings = embeddings;
    this.durationMs = durationMs;
  }

  static fromJSON(d: Record<string, unknown>): Result {
    const dets = Array.isArray(d.detections) ? d.detections : [];
    const masks = Array.isArray(d.masks) ? d.masks : [];
    const clsArr = Array.isArray(d.classifications) ? d.classifications : [];
    const depthMap = Array.isArray(d.depth_map) ? (d.depth_map as unknown[]).map(Number) : [];
    const embeddings = Array.isArray(d.embeddings)
      ? (d.embeddings as unknown[]).map((row) =>
          Array.isArray(row) ? (row as unknown[]).map(Number) : [],
        )
      : [];
    return new Result(
      String(d.task ?? ""),
      String(d.model ?? ""),
      dets.map((x) => Detection.fromJSON(x as Record<string, unknown>)),
      masks.map((x) => Mask.fromJSON(x as Record<string, unknown>)),
      clsArr.map((x) => Classification.fromJSON(x as Record<string, unknown>)),
      depthMap,
      Number(d.depth_width ?? 0),
      Number(d.depth_height ?? 0),
      embeddings,
      Number(d.duration_ms ?? 0),
    );
  }

  filterByConf(minConf = 0, maxConf = 1): Result {
    const inRange = (conf: number) => conf >= minConf && conf <= maxConf;
    return new Result(
      this.task,
      this.model,
      this.detections.filter((d) => inRange(d.conf)),
      this.masks.filter((m) => inRange(m.conf)),
      this.classifications.filter((c) => inRange(c.conf)),
      this.depthMap,
      this.depthWidth,
      this.depthHeight,
      this.embeddings,
      this.durationMs,
    );
  }

  sortByConf(descending = true): Result {
    const cmp = descending
      ? (a: { conf: number }, b: { conf: number }) => b.conf - a.conf
      : (a: { conf: number }, b: { conf: number }) => a.conf - b.conf;
    return new Result(
      this.task,
      this.model,
      this.detections.slice().sort(cmp),
      this.masks.slice().sort(cmp),
      this.classifications.slice().sort(cmp),
      this.depthMap,
      this.depthWidth,
      this.depthHeight,
      this.embeddings,
      this.durationMs,
    );
  }

  topK(k: number): Result {
    const sorted = this.sortByConf(true);
    return new Result(
      this.task,
      this.model,
      sorted.detections.slice(0, k),
      sorted.masks.slice(0, k),
      sorted.classifications.slice(0, k),
      this.depthMap,
      this.depthWidth,
      this.depthHeight,
      this.embeddings,
      this.durationMs,
    );
  }

  nms(iouThreshold = 0.5): Result {
    const toX1Y1X2Y2 = (bbox: number[]): [number, number, number, number] => [
      bbox[0],
      bbox[1],
      bbox[0] + bbox[2],
      bbox[1] + bbox[3],
    ];

    const iou = (a: number[], b: number[]): number => {
      const [ax1, ay1, ax2, ay2] = toX1Y1X2Y2(a);
      const [bx1, by1, bx2, by2] = toX1Y1X2Y2(b);
      const ix1 = Math.max(ax1, bx1);
      const iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2);
      const iy2 = Math.min(ay2, by2);
      const interW = Math.max(0, ix2 - ix1);
      const interH = Math.max(0, iy2 - iy1);
      const inter = interW * interH;
      if (inter === 0) return 0;
      const areaA = (ax2 - ax1) * (ay2 - ay1);
      const areaB = (bx2 - bx1) * (by2 - by1);
      return inter / (areaA + areaB - inter);
    };

    const sorted = this.detections.slice().sort((a, b) => b.conf - a.conf);
    const kept: Detection[] = [];
    for (const det of sorted) {
      if (kept.every((k) => iou(det.bbox, k.bbox) < iouThreshold)) {
        kept.push(det);
      }
    }

    return new Result(
      this.task,
      this.model,
      kept,
      this.masks,
      this.classifications,
      this.depthMap,
      this.depthWidth,
      this.depthHeight,
      this.embeddings,
      this.durationMs,
    );
  }

  groupByClass(): Record<string, Result> {
    const groups: Record<string, { detections: Detection[]; masks: Mask[] }> = {};

    for (const det of this.detections) {
      if (!groups[det.cls]) groups[det.cls] = { detections: [], masks: [] };
      groups[det.cls].detections.push(det);
    }
    for (const mask of this.masks) {
      const key = mask.bbox.join(",");
      const matchingDet = this.detections.find((d) => d.bbox.join(",") === key);
      const cls = matchingDet?.cls ?? "";
      if (!groups[cls]) groups[cls] = { detections: [], masks: [] };
      groups[cls].masks.push(mask);
    }

    const result: Record<string, Result> = {};
    for (const [cls, group] of Object.entries(groups)) {
      result[cls] = new Result(
        this.task,
        this.model,
        group.detections,
        group.masks,
        this.classifications.filter((c) => c.cls === cls),
        this.depthMap,
        this.depthWidth,
        this.depthHeight,
        this.embeddings,
        this.durationMs,
      );
    }
    return result;
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
