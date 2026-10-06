/**
 * Result types for the VisionServe client.
 *
 * These mirror the server's unified wire schema (see `pkg/api/types.go`). The schema
 * is the SAME across every task — detection, segmentation, open-vocab — so there is a
 * single {@link Result} type rather than one per model.
 */

import { ClientResize } from "./resize.js";

/** Task kind reported by the server. Open-ended on purpose (new tasks may appear). */
export type Task =
  | "detection"
  | "segmentation"
  | "open_vocab"
  | "classification"
  | "depth"
  | "embed"
  | "grasp"
  | "instance_detection"
  | (string & {});

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

  /** The wire object (`class`, not `cls`); inverse of {@link Detection.fromJSON}. `JSON.stringify` calls it. */
  toJSON(): Record<string, unknown> {
    return { bbox: this.bbox.slice(), class: this.cls, conf: this.conf };
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
  /**
   * `[width, height]` the RLE was encoded at when the client shrank the image before uploading
   * it (`Result.clientResize`); `null` = the original size. Not part of the wire format.
   */
  readonly rleSize: [number, number] | null;

  constructor(rle: string, bbox: number[], conf: number, rleSize: [number, number] | null = null) {
    this.rle = rle;
    this.bbox = bbox;
    this.conf = conf;
    this.rleSize = rleSize;
  }

  static fromJSON(d: Record<string, unknown>): Mask {
    const bbox = Array.isArray(d.bbox) ? (d.bbox as unknown[]).map(Number) : [0, 0, 0, 0];
    return new Mask(String(d.rle ?? ""), bbox, Number(d.conf ?? 0));
  }

  /**
   * The wire object (`rle` left out when empty, as the server does); inverse of
   * {@link Mask.fromJSON}. `rleSize` is client side and not written: {@link Result.toJSON} keeps
   * it through `client_resize` instead.
   */
  toJSON(): Record<string, unknown> {
    const out: Record<string, unknown> = this.rle ? { rle: this.rle } : {};
    out.bbox = this.bbox.slice();
    out.conf = this.conf;
    return out;
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
   *
   * When the client shrank the image before uploading it, the RLE covers the SENT image
   * (`rleSize`); pass the ORIGINAL size anyway: the mask is decoded at the sent size and scaled
   * to `width` x `height` by nearest neighbour. Passing the sent size returns it as received.
   * @throws if the run counts do not sum to `width * height`.
   */
  toMask(width: number, height: number): Uint8Array {
    if (this.rleSize && (this.rleSize[0] !== width || this.rleSize[1] !== height)) {
      const [sw, sh] = this.rleSize;
      const small = new Mask(this.rle, this.bbox, this.conf).toMask(sw, sh);
      const out = new Uint8Array(width * height);
      // Nearest neighbour on pixel centres: original pixel i samples sent pixel floor((i + 0.5) * sent / original).
      const xs = Array.from({ length: width }, (_, x) => Math.min(sw - 1, Math.floor(((x + 0.5) * sw) / width)));
      for (let y = 0; y < height; y++) {
        const row = Math.min(sh - 1, Math.floor(((y + 0.5) * sh) / height)) * sw;
        for (let x = 0; x < width; x++) out[y * width + x] = small[row + xs[x]!]!;
      }
      return out;
    }
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

  /** The wire object; inverse of {@link Classification.fromJSON}. */
  toJSON(): Record<string, unknown> {
    return { class: this.cls, conf: this.conf };
  }
}

/** A planar parallel-jaw grasp in ORIGINAL image coordinates (from a `grasp` model). */
export class Grasp {
  /** Grasp centre X in ORIGINAL image pixels. */
  readonly x: number;
  /** Grasp centre Y in ORIGINAL image pixels. */
  readonly y: number;
  /** In-plane gripper-closing angle in radians (jaws close along `(cos θ, sin θ)`). */
  readonly theta: number;
  /** Jaw opening in ORIGINAL image pixels. */
  readonly width: number;
  /** Analytic grasp score in `[0, 1]`. */
  readonly quality: number;
  /** Source object label (box mode); `""` for class-agnostic grasps. */
  readonly cls: string;
  /** Source detector confidence (box mode); `0` if class-agnostic. */
  readonly conf: number;

  constructor(x: number, y: number, theta: number, width: number, quality: number, cls = "", conf = 0) {
    this.x = x;
    this.y = y;
    this.theta = theta;
    this.width = width;
    this.quality = quality;
    this.cls = cls;
    this.conf = conf;
  }

  static fromJSON(d: Record<string, unknown>): Grasp {
    return new Grasp(
      Number(d.x ?? 0),
      Number(d.y ?? 0),
      Number(d.theta ?? 0),
      Number(d.width ?? 0),
      Number(d.quality ?? 0),
      String(d["class"] ?? ""),
      Number(d.conf ?? 0),
    );
  }

  /**
   * The wire object (`class` / `conf` left out for a class-agnostic grasp, as the server does);
   * inverse of {@link Grasp.fromJSON}.
   */
  toJSON(): Record<string, unknown> {
    const out: Record<string, unknown> = { x: this.x, y: this.y, theta: this.theta, width: this.width, quality: this.quality };
    if (this.cls) out.class = this.cls;
    if (this.conf) out.conf = this.conf;
    return out;
  }

  /** The grasp pose as `[x, y, width, theta]` for robot control (Python's `Grasp.pose`). */
  get pose(): [number, number, number, number] {
    return [this.x, this.y, this.width, this.theta];
  }

  /**
   * The two jaw-contact points `[[x0, y0], [x1, y1]]` in image pixels:
   * `centre ∓ (width / 2) * (cos θ, sin θ)` (Python's `Grasp.contacts()`).
   */
  contacts(): [[number, number], [number, number]] {
    const [x0, y0, x1, y1] = this.contactsFlat();
    return [
      [x0, y0],
      [x1, y1],
    ];
  }

  /** The jaw-contact points as a flat `[x0, y0, x1, y1]` (Python's `Grasp.contacts_flat()`). */
  contactsFlat(): [number, number, number, number] {
    const dx = (Math.cos(this.theta) * this.width) / 2;
    const dy = (Math.sin(this.theta) * this.width) / 2;
    return [this.x - dx, this.y - dy, this.x + dx, this.y + dy];
  }
}

/** Unified prediction result returned by `POST /api/predict`. */
export class Result {
  readonly task: Task;
  readonly model: string;
  readonly detections: Detection[];
  readonly masks: Mask[];
  readonly grasps: Grasp[];
  readonly classifications: Classification[];
  /**
   * Flat row-major float array for depth maps, at the MODEL's resolution
   * (`depthWidth` x `depthHeight`), not the image's.
   */
  readonly depthMap: number[];
  readonly depthWidth: number;
  readonly depthHeight: number;
  /** One embedding vector per image. */
  readonly embeddings: number[][];
  readonly durationMs: number;
  /** Execution device the server ran on, e.g. `"cpu"`, `"gpu:0"`, `"gpu:0+trt"`. */
  readonly device: string;
  /** The server's setup recommendation, if any (e.g. how to enable TensorRT); `""` otherwise. */
  readonly hint: string;
  /**
   * What the client did to the image before uploading it (original and sent size, JPEG quality),
   * or `null` when the input bytes were sent untouched. Coordinates above are ALWAYS in ORIGINAL
   * image pixels either way. Client side: the server never sends it; {@link Result.toJSON} writes
   * it as `client_resize` (only when set) so a saved result reads back the same.
   */
  readonly clientResize: ClientResize | null;

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
    grasps: Grasp[] = [],
    device = "",
    hint = "",
    clientResize: ClientResize | null = null,
  ) {
    this.task = task;
    this.model = model;
    this.detections = detections;
    this.masks = masks;
    this.grasps = grasps;
    this.classifications = classifications;
    this.depthMap = depthMap;
    this.depthWidth = depthWidth;
    this.depthHeight = depthHeight;
    this.embeddings = embeddings;
    this.durationMs = durationMs;
    this.device = device;
    this.hint = hint;
    this.clientResize = clientResize;
  }

  /**
   * Parse the server's JSON (an object). Both array encodings are accepted: JSON numbers, and
   * `encoding=base64` (`depth_map_base64` / `embeddings_base64` + `embeddings_shape`).
   */
  static fromJSON(d: Record<string, unknown>): Result {
    const dets = Array.isArray(d.detections) ? d.detections : [];
    const masks = Array.isArray(d.masks) ? d.masks : [];
    const grasps = Array.isArray(d.grasps) ? d.grasps : [];
    const clsArr = Array.isArray(d.classifications) ? d.classifications : [];
    // Both array encodings are accepted: JSON numbers, and `encoding=base64`
    // (`depth_map_base64` / `embeddings_base64` + `embeddings_shape`), decoded here.
    const depthMap =
      typeof d.depth_map_base64 === "string" && d.depth_map_base64
        ? decodeFloat32Base64(d.depth_map_base64)
        : Array.isArray(d.depth_map)
          ? (d.depth_map as unknown[]).map(Number)
          : [];
    let embeddings: number[][];
    if (typeof d.embeddings_base64 === "string" && d.embeddings_base64) {
      const flat = decodeFloat32Base64(d.embeddings_base64);
      const shape = Array.isArray(d.embeddings_shape) ? (d.embeddings_shape as unknown[]).map(Number) : [];
      const dim = shape.length === 2 && shape[1]! > 0 ? shape[1]! : flat.length;
      embeddings = [];
      for (let i = 0; i < flat.length; i += dim) embeddings.push(flat.slice(i, i + dim));
    } else {
      embeddings = Array.isArray(d.embeddings)
        ? (d.embeddings as unknown[]).map((row) =>
            Array.isArray(row) ? (row as unknown[]).map(Number) : [],
          )
        : [];
    }
    const res = new Result(
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
      grasps.map((x) => Grasp.fromJSON(x as Record<string, unknown>)),
      String(d.device ?? ""),
      String(d.hint ?? ""),
    );
    // `client_resize` is not from the server: it is what toJSON() wrote for a result whose photo
    // the client shrank. Restore it, and the masks' RLE size, exactly as mapResult() set them.
    const cr = clientResizeFromJSON(d.client_resize);
    if (!cr) return res;
    return res.with({
      clientResize: cr,
      masks: cr.resized
        ? res.masks.map((m) => (m.rle ? new Mask(m.rle, m.bbox, m.conf, [cr.sentWidth, cr.sentHeight]) : m))
        : res.masks,
    });
  }

  /**
   * The server's JSON wire shape for this result: the inverse of {@link Result.fromJSON}, so
   * `Result.fromJSON(JSON.parse(JSON.stringify(res)))` deep-equals `res`. `JSON.stringify` calls
   * it. Field names and order follow `pkg/api/types.go` (`class`, `duration_ms`, ...) and empty
   * fields are left out like the Go `omitempty` tags (same as Python's `Result.to_json()`).
   *
   * `{ encoding: "base64" }` writes `depthMap` / `embeddings` the way the server does for
   * `encoding=base64` (`depth_map_base64`, `embeddings_base64` + `embeddings_shape`: exact float32
   * bytes, compact); ragged embedding rows stay numbers. The default writes number arrays.
   *
   * One client-side field is added when set: `client_resize` (what the client uploaded:
   * `original_width`, `original_height`, `sent_width`, `sent_height`, `jpeg_quality`, `reason`).
   * The server never sends it and the Python SDK ignores it.
   */
  toJSON(opts?: { encoding?: "json" | "base64" } | string): Record<string, unknown> {
    // JSON.stringify passes the property key (a string) here: that means the default.
    const encoding = typeof opts === "object" && opts !== null ? (opts.encoding ?? "json") : "json";
    if (encoding !== "json" && encoding !== "base64") {
      throw new Error(`encoding must be "json" or "base64", got ${JSON.stringify(encoding)}`);
    }
    const out: Record<string, unknown> = { task: this.task, model: this.model };
    if (this.device) out.device = this.device;
    if (this.hint) out.hint = this.hint;
    if (this.detections.length) out.detections = this.detections.map((d) => d.toJSON());
    if (this.masks.length) out.masks = this.masks.map((m) => m.toJSON());
    if (this.grasps.length) out.grasps = this.grasps.map((g) => g.toJSON());
    if (this.classifications.length) out.classifications = this.classifications.map((c) => c.toJSON());
    const b64 = encoding === "base64";
    const rows = this.embeddings.length;
    const dim = rows ? this.embeddings[0]!.length : 0;
    const embB64 = b64 && rows > 0 && this.embeddings.every((r) => r.length === dim);
    if (rows && !embB64) out.embeddings = this.embeddings.map((r) => r.slice());
    if (this.depthMap.length && !b64) out.depth_map = this.depthMap.slice();
    if (this.depthWidth) out.depth_width = this.depthWidth;
    if (this.depthHeight) out.depth_height = this.depthHeight;
    out.duration_ms = this.durationMs;
    if (b64 && this.depthMap.length) out.depth_map_base64 = encodeFloat32Base64(this.depthMap);
    if (embB64) {
      out.embeddings_base64 = encodeFloat32Base64(this.embeddings.flat());
      out.embeddings_shape = [rows, dim];
    }
    const cr = this.clientResize;
    if (cr) {
      out.client_resize = {
        original_width: cr.originalWidth,
        original_height: cr.originalHeight,
        sent_width: cr.sentWidth,
        sent_height: cr.sentHeight,
        jpeg_quality: cr.jpegQuality,
        reason: cr.reason,
      };
    }
    return out;
  }

  /** A copy with some fields replaced (every helper below returns a new Result this way). */
  private with(o: Partial<ResultFields>): Result {
    const f = { ...this.fields(), ...o };
    return new Result(
      f.task,
      f.model,
      f.detections,
      f.masks,
      f.classifications,
      f.depthMap,
      f.depthWidth,
      f.depthHeight,
      f.embeddings,
      f.durationMs,
      f.grasps,
      f.device,
      f.hint,
      f.clientResize,
    );
  }

  private fields(): ResultFields {
    return {
      task: this.task,
      model: this.model,
      detections: this.detections,
      masks: this.masks,
      grasps: this.grasps,
      classifications: this.classifications,
      depthMap: this.depthMap,
      depthWidth: this.depthWidth,
      depthHeight: this.depthHeight,
      embeddings: this.embeddings,
      durationMs: this.durationMs,
      device: this.device,
      hint: this.hint,
      clientResize: this.clientResize,
    };
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
      this.grasps,
      this.device,
      this.hint,
      this.clientResize,
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
      this.grasps,
      this.device,
      this.hint,
      this.clientResize,
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
      this.grasps,
      this.device,
      this.hint,
      this.clientResize,
    );
  }

  nms(iouThreshold = 0.5): Result {
    const toX1Y1X2Y2 = (bbox: number[]): [number, number, number, number] => {
      const [x = 0, y = 0, w = 0, h = 0] = bbox;
      return [x, y, x + w, y + h];
    };

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
      this.grasps,
      this.device,
      this.hint,
      this.clientResize,
    );
  }

  /**
   * Keep the `maxPerObject` highest-quality grasps per object (Python's `filter_grasps`).
   *
   * A class-aware grasp (box mode: `cls` / `conf` copied from its detection) belongs to the
   * detection with the same class and conf, wherever its centre lies; a class-agnostic grasp
   * (automatic or box-prompted masks) to the SMALLEST detection box (or mask box, when there are
   * no detections) containing its centre, or to its label's bucket when no box does. Within a
   * group the order is by quality, best first; groups keep the order of their first grasp.
   * `undefined`, `null` or `<= 0` returns this result unchanged. Other fields are kept.
   */
  filterGrasps(maxPerObject?: number | null): Result {
    if (maxPerObject == null || maxPerObject <= 0 || !this.grasps.length) return this;
    return this.with({ grasps: topGraspsPerObject(this.grasps, this.detections, this.masks, maxPerObject) });
  }

  /**
   * `{ label: Result }`: one result per class label, holding only that class's items. Same
   * semantics as Python's `group_by_class()`:
   *
   * - `detections`: the detections with that label.
   * - `masks`: masks carry no class on the wire, so a mask whose box equals a detection's box
   *   (Grounded-SAM and the grasp pipelines copy the detection box onto its mask) takes that
   *   detection's label; any other mask (e.g. a box-prompted SAM mask) goes under `""`.
   * - `grasps`: a class-aware grasp goes with its `cls` (its source detection's class); a
   *   class-agnostic one with the class of the smallest detection box containing its centre,
   *   else under `""`.
   * - `classifications`, `depthMap` and `embeddings` are per image, not per object: every group
   *   has them empty (`depthWidth` / `depthHeight` 0) instead of a copy of the whole map.
   * - `task`, `model`, `durationMs`, `device`, `hint` and `clientResize` are kept.
   *
   * Groups appear in the order their label is first met (detections, then masks, then grasps).
   */
  groupByClass(): Record<string, Result> {
    const groups = new Map<string, { detections: Detection[]; masks: Mask[]; grasps: Grasp[] }>();
    const group = (label: string) => {
      let g = groups.get(label);
      if (!g) groups.set(label, (g = { detections: [], masks: [], grasps: [] }));
      return g;
    };
    const boxLabel = new Map<string, string>();
    for (const det of this.detections) {
      group(det.cls).detections.push(det);
      const key = det.bbox.map(Number).join(",");
      if (!boxLabel.has(key)) boxLabel.set(key, det.cls);
    }
    for (const mask of this.masks) {
      group(boxLabel.get(mask.bbox.map(Number).join(",")) ?? "").masks.push(mask);
    }
    const boxes = this.detections.map((d) => d.bbox);
    for (const g of this.grasps) {
      let label = g.cls;
      if (!g.cls && !g.conf) {
        const i = boxes.length ? graspObjectIndex(g, boxes) : -1;
        label = i >= 0 ? this.detections[i]!.cls : "";
      }
      group(label).grasps.push(g);
    }

    const out: Record<string, Result> = {};
    for (const [label, g] of groups) {
      out[label] = this.with({
        detections: g.detections,
        masks: g.masks,
        grasps: g.grasps,
        classifications: [],
        depthMap: [],
        depthWidth: 0,
        depthHeight: 0,
        embeddings: [],
      });
    }
    return out;
  }
}

/** Every field of a {@link Result}, for copies with some of them replaced. */
interface ResultFields {
  task: Task;
  model: string;
  detections: Detection[];
  masks: Mask[];
  grasps: Grasp[];
  classifications: Classification[];
  depthMap: number[];
  depthWidth: number;
  depthHeight: number;
  embeddings: number[][];
  durationMs: number;
  device: string;
  hint: string;
  clientResize: ClientResize | null;
}

/** Index of the SMALLEST `[x, y, w, h]` box whose interior contains the grasp centre, or -1. */
function graspObjectIndex(g: Grasp, objects: readonly (readonly number[])[]): number {
  let best = -1;
  let bestArea = Infinity;
  objects.forEach((b, i) => {
    const [x = 0, y = 0, w = 0, h = 0] = b;
    if (x <= g.x && g.x <= x + w && y <= g.y && g.y <= y + h) {
      const area = w * h;
      if (area < bestArea) {
        bestArea = area;
        best = i;
      }
    }
  });
  return best;
}

/**
 * A class-aware grasp carries its source detection's conf as sent; the tolerance only absorbs a
 * round trip through a lower-precision format (float32, a rounded JSON writer).
 */
const GRASP_CONF_TOL = 1e-6;

/**
 * Index of the detection a class-aware grasp was planned on: same class and conf (within
 * {@link GRASP_CONF_TOL}); among several, the smallest whose box contains the grasp centre, else
 * the first. -1 when none matches.
 */
function graspSource(g: Grasp, detections: readonly Detection[]): number {
  const cands: number[] = [];
  detections.forEach((d, i) => {
    if (d.cls === g.cls && Math.abs(d.conf - g.conf) <= GRASP_CONF_TOL) cands.push(i);
  });
  if (cands.length <= 1) return cands.length ? cands[0]! : -1;
  const inside = graspObjectIndex(g, cands.map((i) => detections[i]!.bbox));
  return cands[inside >= 0 ? inside : 0]!;
}

/**
 * The `k` best grasps per object: the one grouping rule behind {@link Result.filterGrasps} and
 * `toSVG` (Python's `_top_grasps_per_object`).
 *
 * A class-aware grasp (`cls` or `conf` set) belongs to its SOURCE detection, found by class and
 * conf, not by position: a bowl grasp whose centre lies inside the carrot's box is still the
 * bowl's. Without a matching detection it is grouped by its `(cls, conf)`. A class-agnostic grasp
 * belongs to the smallest detection (else mask) box containing its centre; one that no box
 * contains is bucketed by its label. Groups keep the order of their first grasp; within a group
 * the best quality comes first.
 */
export function topGraspsPerObject(
  grasps: readonly Grasp[],
  detections: readonly Detection[],
  masks: readonly Mask[],
  k: number,
): Grasp[] {
  const boxes = detections.length ? detections.map((d) => d.bbox) : masks.map((m) => m.bbox);
  const groups = new Map<string, Grasp[]>();
  for (const g of grasps) {
    let key: string;
    if (g.cls || g.conf) {
      const src = graspSource(g, detections);
      key = src >= 0 ? "obj:" + src : "src:" + JSON.stringify([g.cls, g.conf]);
    } else {
      const i = boxes.length ? graspObjectIndex(g, boxes) : -1;
      key = i >= 0 ? "obj:" + i : "cls:" + g.cls;
    }
    let gs = groups.get(key);
    if (!gs) groups.set(key, (gs = []));
    gs.push(g);
  }
  const kept: Grasp[] = [];
  for (const gs of groups.values()) {
    gs.sort((a, b) => b.quality - a.quality); // stable, like Python's sort
    kept.push(...gs.slice(0, k));
  }
  return kept;
}

/** A ClientResize from its `client_resize` JSON (see {@link Result.toJSON}), or null. */
function clientResizeFromJSON(v: unknown): ClientResize | null {
  if (v == null || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const n = (k: string) => Number(o[k] ?? 0);
  const q = o.jpeg_quality;
  return new ClientResize(
    n("original_width"),
    n("original_height"),
    n("sent_width"),
    n("sent_height"),
    typeof q === "number" ? q : null,
    String(o.reason ?? ""),
  );
}

/** An entry from `GET /api/models`. */
export class ModelInfo {
  readonly name: string;
  readonly task: Task;
  readonly license: string;
  readonly state: ModelState;
  /**
   * The server's client-resize hint: the longest LONGER side (`maxUsefulSide`) or SHORTER side
   * (`maxUsefulShortSide`) worth uploading for this model, which resizes to its own input
   * anyway. At most one is set; both `null` = send full resolution (masks, OCR, templates, ...,
   * or a server that predates the hint). {@link Client} applies it by default.
   */
  readonly maxUsefulSide: number | null;
  readonly maxUsefulShortSide: number | null;
  /** Whether the model reads an uploaded depth map (`accepts_depth`; false on an older server). */
  readonly acceptsDepth: boolean;

  constructor(
    name: string,
    task: Task,
    license: string,
    state: ModelState,
    maxUsefulSide: number | null = null,
    maxUsefulShortSide: number | null = null,
    acceptsDepth = false,
  ) {
    this.name = name;
    this.task = task;
    this.license = license;
    this.state = state;
    this.maxUsefulSide = maxUsefulSide;
    this.maxUsefulShortSide = maxUsefulShortSide;
    this.acceptsDepth = acceptsDepth;
  }

  static fromJSON(d: Record<string, unknown>): ModelInfo {
    return new ModelInfo(
      String(d.name ?? ""),
      String(d.task ?? ""),
      String(d.license ?? ""),
      String(d.state ?? ""),
      positiveInt(d.max_useful_side),
      positiveInt(d.max_useful_short_side),
      d.accepts_depth === true,
    );
  }

  get isLoaded(): boolean {
    return this.state === "loaded";
  }
}

/** A positive integer hint, or null (null, absent, 0, or not an integer). */
function positiveInt(v: unknown): number | null {
  return typeof v === "number" && Number.isInteger(v) && v > 0 ? v : null;
}

/**
 * Decode base64 of little-endian float32 bytes (the server's `encoding=base64` arrays) into
 * plain numbers: the exact float32 values the server computed.
 */
export function decodeFloat32Base64(b64: string): number[] {
  const bin = atob(b64);
  if (bin.length % 4 !== 0) {
    throw new Error(`base64 float32 array has ${bin.length} bytes, not a multiple of 4`);
  }
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const out = new Array<number>(bin.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/** Inverse of {@link decodeFloat32Base64}: base64 of the little-endian float32 bytes. */
export function encodeFloat32Base64(values: ArrayLike<number>): string {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) view.setFloat32(i * 4, values[i]!, true);
  return bytesToBase64(bytes);
}

/** Base64 of raw bytes with `btoa` (Node 16+ and browsers), in chunks to bound the arguments. */
export function bytesToBase64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}
