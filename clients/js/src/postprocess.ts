/**
 * Robot / grasp helpers: metric depth, camera distances and target selection. A port of the
 * Python SDK's `visionserve/postprocess.py` with the same maths and defaults; both SDKs run the
 * shared cases in `clients/testdata/postprocess_sync.json` (generated from the Python code).
 *
 * Distances in metres need a METRIC depth image from an RGB-D sensor ({@link DepthImage}), aligned
 * to the photo. A depth model's answer (`midas`, `depth-anything-v2`) is relative inverse depth
 * normalised per image, with no scale, so these helpers refuse it.
 */

import { Detection, Grasp, Mask, Result } from "./types.js";

/** Pinhole camera intrinsics in pixels: focal lengths `fx`, `fy`, principal point `cx`, `cy`. */
export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

/** {@link CameraIntrinsics} or the same four numbers as `[fx, fy, cx, cy]`. */
export type IntrinsicsInput = CameraIntrinsics | readonly number[];

/**
 * A metric depth image, row-major, at the photo's resolution (or pass the photo's size where a
 * helper takes `imageWidth` / `imageHeight`). An INTEGER typed array (`Uint16Array`, ...) is read
 * as millimetres (`depthScale` 0.001 by default), anything else (`Float32Array`, `number[]`) as
 * metres (`depthScale` 1). A value `<= 0` (or NaN) means "no reading".
 */
export interface DepthImage {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/** Back-project pixel `(u, v)` at depth `z` to `[X, Y, Z]` in the camera frame. */
export function backproject(u: number, v: number, z: number, K: IntrinsicsInput): [number, number, number] {
  const k = asIntrinsics(K);
  return [((u - k.cx) * z) / k.fx, ((v - k.cy) * z) / k.fy, z];
}

/**
 * Euclidean distance (same unit as `z`) from the camera centre to the 3D point that pixel
 * `(u, v)` at depth `z` back-projects to.
 */
export function cameraDistance(u: number, v: number, z: number, K: IntrinsicsInput): number {
  const [x, y, zz] = backproject(u, v, z, K);
  return Math.sqrt(x * x + y * y + zz * zz);
}

/** `CameraIntrinsics` from either form; throws on anything else. */
export function asIntrinsics(K: IntrinsicsInput): CameraIntrinsics {
  if (Array.isArray(K) || ArrayBuffer.isView(K)) {
    const vals = Array.from(K as ArrayLike<number>, Number);
    if (vals.length !== 4) {
      throw new Error(`intrinsics must have 4 values [fx, fy, cx, cy], got ${vals.length}`);
    }
    return { fx: vals[0]!, fy: vals[1]!, cx: vals[2]!, cy: vals[3]! };
  }
  const k = K as CameraIntrinsics;
  if (k == null || typeof k !== "object" || !["fx", "fy", "cx", "cy"].every((n) => typeof (k as any)[n] === "number")) {
    throw new TypeError("intrinsics must be { fx, fy, cx, cy } or [fx, fy, cx, cy]");
  }
  return { fx: k.fx, fy: k.fy, cx: k.cx, cy: k.cy };
}

const RELATIVE_DEPTH_MSG =
  "a depth Result from the server (midas / depth-anything-v2) is RELATIVE inverse depth, " +
  "min-max normalised to [0, 1] per image (larger = closer) at the model's resolution: it " +
  "has no metric scale, so it cannot give distances in metres. Pass a METRIC depth image " +
  "from an RGB-D sensor instead ({ data: Uint16Array of millimetres, width, height }, aligned " +
  "to the photo), or use getDepthAtDetection(depthResult, det, { imageWidth, imageHeight }) " +
  "for a relative near/far ordering.";

/** True for a depth `Result` (or anything with a `depthMap`), false for a {@link DepthImage}. */
export function isDepthResult(depth: unknown): depth is Result {
  return depth instanceof Result || (typeof depth === "object" && depth !== null && "depthMap" in depth);
}

const INTEGER_ARRAYS = [Uint8Array, Uint8ClampedArray, Int8Array, Uint16Array, Int16Array, Uint32Array, Int32Array];

/**
 * The depth image in METRES as float32 values (Python's `_as_depth_meters`): `raw * depthScale`,
 * `depthScale` defaulting to 0.001 for integer arrays (millimetres) and 1 otherwise.
 * @internal
 */
export function depthMeters(depth: DepthImage | Result, depthScale?: number | null): Float32Array {
  if (isDepthResult(depth)) throw new Error(RELATIVE_DEPTH_MSG);
  if (depth == null || typeof depth !== "object" || depth.data == null) {
    throw new TypeError("depth must be { data, width, height } of METRIC depth");
  }
  const { data, width, height } = depth;
  if (!(Number.isInteger(width) && width > 0 && Number.isInteger(height) && height > 0)) {
    throw new Error(`depth width and height must be positive integers, got ${width}x${height}`);
  }
  if (data.length !== width * height) {
    throw new Error(`depth has ${data.length} values, width * height = ${width * height}`);
  }
  const scale = depthScale ?? (INTEGER_ARRAYS.some((T) => data instanceof T) ? 0.001 : 1.0);
  const s = Math.fround(scale);
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Math.fround(data[i]!) * s; // float32 * float32, as numpy
  return out;
}

/** Median of a non-empty list (the mean of the two middle values for an even count). */
export function median(vals: number[]): number {
  const s = vals.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1]! + s[mid]!) / 2 : s[mid]!;
}

/** Python's `round()`: half to even. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Median of the valid (> 0) depth pixels in a `(2 * window + 1)` square around `(x, y)`, or null. */
function depthAtPoint(arr: Float32Array, w: number, h: number, x: number, y: number, window: number): number | null {
  const xi = roundHalfEven(x);
  const yi = roundHalfEven(y);
  const x1 = Math.max(0, xi - window);
  const x2 = Math.min(w, xi + window + 1);
  const y1 = Math.max(0, yi - window);
  const y2 = Math.min(h, yi + window + 1);
  if (x2 <= x1 || y2 <= y1) return null;
  const vals: number[] = [];
  for (let yy = y1; yy < y2; yy++) {
    for (let xx = x1; xx < x2; xx++) {
      const v = arr[yy * w + xx]!;
      if (v > 0) vals.push(v);
    }
  }
  return vals.length ? Math.fround(median(vals)) : null;
}

/** Options for {@link objectDistances}. */
export interface ObjectDistanceOptions {
  /** How the depth pixels under a box become one value: `"median"` (default), `"mean"`, `"min"`, `"max"`. */
  mode?: "median" | "mean" | "min" | "max";
  /** Metres per depth unit; default 0.001 for integer arrays (millimetres), 1 otherwise. */
  depthScale?: number | null;
}

/**
 * True camera→object Euclidean distance in METRES for each detection (or each mask when there are
 * none): the object's depth (over its box, aggregated by `mode`), back-projected at the box CENTRE
 * through the intrinsics. One entry per object, `null` where the box has no depth reading.
 * `depth` must be a metric {@link DepthImage} at the photo's resolution; a depth `Result` throws.
 */
export function objectDistances(
  depth: DepthImage,
  detResult: Result,
  intrinsics: IntrinsicsInput,
  opts: ObjectDistanceOptions = {},
): Array<number | null> {
  const K = asIntrinsics(intrinsics);
  if (isDepthResult(depth)) throw new Error(RELATIVE_DEPTH_MSG);
  const depths = depthUnderBoxes(depth, detResult, opts.mode ?? "median", opts.depthScale);
  const items: Array<Detection | Mask> = detResult.detections.length ? detResult.detections : detResult.masks;
  return items.map((item, i) => {
    const z = depths[i];
    if (z == null || z <= 0) return null;
    const [x = 0, y = 0, w = 0, h = 0] = item.bbox;
    return cameraDistance(x + w / 2, y + h / 2, z, K);
  });
}

/** Options for {@link graspDistances}. */
export interface GraspDistanceOptions {
  /** Half-size of the square of depth pixels around the grasp centre (median of the valid ones). Default 2 (5 x 5). */
  window?: number;
  /** Metres per depth unit; default 0.001 for integer arrays (millimetres), 1 otherwise. */
  depthScale?: number | null;
}

/**
 * True camera→grasp Euclidean distance in METRES for each grasp, from the depth at the grasp
 * centre. One entry per grasp, `null` where there is no depth reading. `depth` must be a metric
 * {@link DepthImage} at the grasps' pixel resolution; a depth `Result` throws.
 */
export function graspDistances(
  depth: DepthImage,
  grasps: readonly Grasp[],
  intrinsics: IntrinsicsInput,
  opts: GraspDistanceOptions = {},
): Array<number | null> {
  const K = asIntrinsics(intrinsics);
  const arr = depthMeters(depth, opts.depthScale);
  const window = opts.window ?? 2;
  return grasps.map((g) => {
    const z = depthAtPoint(arr, depth.width, depth.height, g.x, g.y, window);
    return z == null || z <= 0 ? null : cameraDistance(g.x, g.y, z, K);
  });
}

/**
 * The metric depth under each box of `detResult` (each detection, else each mask), in metres,
 * aggregated by `mode` over the valid (finite, > 0) pixels; `null` where there are none. The
 * boxes are scaled onto the depth image when `imageSize` (the photo's `[W, H]`) is given.
 * @internal shared by `getDepthAtDetection` (with a DepthImage) and {@link objectDistances}.
 */
export function depthUnderBoxes(
  depth: DepthImage,
  detResult: Result,
  mode: string,
  depthScale?: number | null,
  imageSize?: [number, number] | null,
): Array<number | null> {
  const agg = Object.hasOwn(AGGREGATES, mode) ? AGGREGATES[mode] : undefined;
  if (!agg) throw new Error(`unknown mode ${JSON.stringify(mode)}; use "median", "mean", "min" or "max"`);
  const arr = depthMeters(depth, depthScale);
  const { width: W, height: H } = depth;
  let sx = 1;
  let sy = 1;
  if (imageSize) {
    const [iw, ih] = imageSize;
    if (!(iw > 0) || !(ih > 0)) throw new Error(`imageWidth and imageHeight must be > 0, got ${iw}x${ih}`);
    sx = W / iw;
    sy = H / ih;
  }
  const boxes = detResult.detections.length ? detResult.detections.map((d) => d.bbox) : detResult.masks.map((m) => m.bbox);
  return boxes.map((bbox) => {
    const [bx = 0, by = 0, bw = 0, bh = 0] = bbox;
    const x0 = Math.max(0, Math.floor(bx * sx));
    const y0 = Math.max(0, Math.floor(by * sy));
    const x1 = Math.min(W, Math.ceil((bx + bw) * sx));
    const y1 = Math.min(H, Math.ceil((by + bh) * sy));
    if (x1 <= x0 || y1 <= y0) return null;
    const vals: number[] = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const v = arr[y * W + x]!;
        if (Number.isFinite(v) && v > 0) vals.push(v);
      }
    }
    return vals.length ? Math.fround(agg(vals)) : null; // numpy aggregates float32 to float32
  });
}

/** The four reductions of `mode`, shared with `getDepthAtDetection`. @internal */
export const AGGREGATES: Record<string, (vals: number[]) => number> = {
  median,
  mean: (v) => v.reduce((s, x) => s + x, 0) / v.length,
  min: (v) => v.reduce((a, b) => (b < a ? b : a)),
  max: (v) => v.reduce((a, b) => (b > a ? b : a)),
};

// --------------------------------------------------------------------------- //
// Target selection
// --------------------------------------------------------------------------- //

/** `exp(-0.5 * ((value - target) / sigma)^2)`: 1 at the target, towards 0 far from it. */
function gaussCloseness(value: number, target: number, sigma: number): number {
  if (sigma <= 0) sigma = 1e-6;
  const d = (value - target) / sigma;
  return Math.exp(-0.5 * d * d);
}

/**
 * The active criteria: explicit `weights` (those > 0) win; else the single most specific one
 * available: distance > near > the model's own score.
 */
function resolveWeights(
  weights: Record<string, number> | undefined | null,
  hasNear: boolean,
  hasDistance: boolean,
  scoreKey: string,
): Record<string, number> {
  if (weights && Object.keys(weights).length) {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(weights)) if (v && v > 0) out[k] = Number(v);
    return out;
  }
  if (hasDistance) return { distance: 1 };
  if (hasNear) return { near: 1 };
  return { [scoreKey]: 1 };
}

const labelSet = (cls: string | readonly string[] | undefined | null) =>
  cls == null ? null : new Set(typeof cls === "string" ? [cls] : cls);

/** Options for {@link selectTargetObject}. */
export interface SelectObjectOptions {
  /** Keep only objects with this label (or one of these labels). Masks have no label. */
  cls?: string | readonly string[] | null;
  /** Keep only objects with `conf >= minConf`. Default 0. */
  minConf?: number;
  /** Prefer the object whose box centre is nearest this pixel `[x, y]`, or `"center"` of the image. */
  nearPoint?: "center" | readonly [number, number] | null;
  /** The photo's width, needed by `nearPoint: "center"`. */
  imageWidth?: number;
  /** The photo's height, needed by `nearPoint: "center"`. */
  imageHeight?: number;
  /** Metric depth for the `"distance"` criterion (with `intrinsics` and `targetDistance`). */
  depth?: DepthImage | null;
  intrinsics?: IntrinsicsInput | null;
  /** Prefer objects whose camera distance (metres) is close to this. */
  targetDistance?: number | null;
  /** Width of the distance preference (metres). Default `0.5 * targetDistance`. */
  distanceSigma?: number | null;
  /** How the depth under a box is reduced. Default `"median"`. */
  mode?: "median" | "mean" | "min" | "max";
  /** Metres per depth unit (default 0.001 for integer arrays, 1 otherwise). */
  depthScale?: number | null;
  /** A weighted mix of criteria, e.g. `{ conf: 1, area: 1, near: 2 }`; each is in `[0, 1]`. */
  weights?: Record<string, number> | null;
}

/**
 * The best target object of a detection / segmentation result (Python's `select_target_object`,
 * same scoring). Candidates are filtered by `cls` and `minConf`, then scored on criteria in
 * `[0, 1]`:
 *
 * - `conf`: the detection's confidence;
 * - `area`: box area relative to the largest candidate;
 * - `near`: 2D closeness of the box centre to `nearPoint` (nearest → 1, farthest → 0);
 * - `distance`: Gaussian closeness of the TRUE camera→object distance to `targetDistance`
 *   (needs a metric `depth` and `intrinsics`; width `distanceSigma`, default half the target).
 *
 * Without `weights`, the single most specific available criterion is used: distance > near >
 * conf. Ties go to the earlier object. Returns the chosen `Detection` (or `Mask`, when the
 * result has no detections), or `null` when no candidate passes the filters; use
 * {@link selectTargetObjectIndex} for its index too.
 */
export function selectTargetObject(result: Result, opts: SelectObjectOptions = {}): Detection | Mask | null {
  return selectTargetObjectIndex(result, opts)[0];
}

/** {@link selectTargetObject} returning `[object, index]` (`[null, -1]` when none). */
export function selectTargetObjectIndex(result: Result, opts: SelectObjectOptions = {}): [Detection | Mask | null, number] {
  const items: Array<Detection | Mask> = result.detections.length ? result.detections : result.masks;
  const clsSet = labelSet(opts.cls);
  const minConf = opts.minConf ?? 0;
  const cand: number[] = [];
  items.forEach((it, i) => {
    if (clsSet && !clsSet.has((it as Partial<Detection>).cls ?? "")) return;
    if ((it.conf ?? 0) < minConf) return;
    cand.push(i);
  });
  if (!cand.length) return [null, -1];

  const center = (i: number): [number, number] => {
    const [x = 0, y = 0, w = 0, h = 0] = items[i]!.bbox;
    return [x + w / 2, y + h / 2];
  };
  const area = new Map(cand.map((i) => [i, Math.max(0, (items[i]!.bbox[2] ?? 0) * (items[i]!.bbox[3] ?? 0))]));
  const maxArea = Math.max(...area.values()) || 1;

  let pt: [number, number] | null = null;
  if (opts.nearPoint != null) {
    if (typeof opts.nearPoint === "string") {
      if (opts.nearPoint !== "center") throw new Error('nearPoint string must be "center"');
      if (opts.imageWidth == null || opts.imageHeight == null) {
        throw new Error('nearPoint "center" requires imageWidth and imageHeight');
      }
      pt = [opts.imageWidth / 2, opts.imageHeight / 2];
    } else {
      pt = [Number(opts.nearPoint[0]), Number(opts.nearPoint[1])];
    }
  }
  const nearD = new Map<number, number>();
  if (pt) for (const i of cand) nearD.set(i, Math.hypot(center(i)[0] - pt[0], center(i)[1] - pt[1]));
  const maxNear = nearD.size ? Math.max(...nearD.values()) : 0;

  const target = opts.targetDistance;
  let camD: Array<number | null> = [];
  let sigma = opts.distanceSigma;
  if (target != null) {
    if (opts.depth == null || opts.intrinsics == null) throw new Error("targetDistance requires depth and intrinsics");
    camD = objectDistances(opts.depth, result, opts.intrinsics, { mode: opts.mode, depthScale: opts.depthScale });
    if (sigma == null) sigma = Math.max(1e-6, 0.5 * Math.abs(target));
  }

  const w = resolveWeights(opts.weights, pt != null, target != null, "conf");
  let best = -1;
  let bestScore = -Infinity;
  for (const i of cand) {
    let s = 0;
    let wsum = 0;
    if ((w.conf ?? 0) > 0) {
      s += w.conf! * (items[i]!.conf ?? 0);
      wsum += w.conf!;
    }
    if ((w.area ?? 0) > 0) {
      s += w.area! * (area.get(i)! / maxArea);
      wsum += w.area!;
    }
    if ((w.near ?? 0) > 0 && pt) {
      s += w.near! * (maxNear > 0 ? 1 - nearD.get(i)! / maxNear : 1);
      wsum += w.near!;
    }
    if ((w.distance ?? 0) > 0 && target != null) {
      const cd = camD[i];
      s += w.distance! * (cd == null ? 0 : gaussCloseness(cd, target, sigma!));
      wsum += w.distance!;
    }
    const score = wsum > 0 ? s / wsum : 0;
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best >= 0 ? [items[best]!, best] : [null, -1];
}

/** Options for {@link selectTargetGrasp}. */
export interface SelectGraspOptions {
  /** Keep only grasps with this label (or one of these labels). */
  cls?: string | readonly string[] | null;
  /** Keep only grasps with `width >= gripperMin` (pixels). */
  gripperMin?: number | null;
  /** Keep only grasps with `width <= gripperMax` (pixels). */
  gripperMax?: number | null;
  /** Prefer the grasp whose centre is nearest this pixel `[x, y]`. */
  targetPoint?: readonly [number, number] | null;
  /** Metric depth for the `"distance"` criterion (with `intrinsics` and `targetDistance`). */
  depth?: DepthImage | null;
  intrinsics?: IntrinsicsInput | null;
  /** Prefer grasps whose camera distance (metres) is close to this. */
  targetDistance?: number | null;
  /** Width of the distance preference (metres). Default `0.15 * targetDistance`. */
  distanceSigma?: number | null;
  /** Half-size of the depth window at a grasp centre. Default 2. */
  window?: number;
  /** Metres per depth unit (default 0.001 for integer arrays, 1 otherwise). */
  depthScale?: number | null;
  /** A weighted mix of criteria, e.g. `{ quality: 1, near: 1, width: 0.5 }`; each is in `[0, 1]`. */
  weights?: Record<string, number> | null;
}

/**
 * The best grasp of a list (Python's `select_target_grasp`, same scoring). Candidates are
 * filtered by `cls` and gripper feasibility (`gripperMin <= width <= gripperMax`), then scored on
 * criteria in `[0, 1]`:
 *
 * - `quality`: the analytic grasp score;
 * - `near`: 2D closeness of the grasp centre to `targetPoint`;
 * - `distance`: Gaussian closeness of the TRUE camera→grasp distance to `targetDistance` (needs
 *   a metric `depth` and `intrinsics`; width `distanceSigma`, default 0.15 × the target);
 * - `width`: preference for an opening in the middle of `[gripperMin, gripperMax]` (both given).
 *
 * Without `weights`, the single most specific available criterion is used: distance > near >
 * quality. Ties go to the earlier grasp. Returns the chosen `Grasp` (the same object as in
 * `grasps`), or `null`; use {@link selectTargetGraspIndex} for its index too.
 */
export function selectTargetGrasp(grasps: readonly Grasp[], opts: SelectGraspOptions = {}): Grasp | null {
  return selectTargetGraspIndex(grasps, opts)[0];
}

/** {@link selectTargetGrasp} returning `[grasp, index]` (`[null, -1]` when none). */
export function selectTargetGraspIndex(grasps: readonly Grasp[], opts: SelectGraspOptions = {}): [Grasp | null, number] {
  const clsSet = labelSet(opts.cls);
  const { gripperMin, gripperMax } = opts;
  const cand: number[] = [];
  grasps.forEach((g, i) => {
    if (clsSet && !clsSet.has(g.cls)) return;
    if (gripperMin != null && g.width < gripperMin) return;
    if (gripperMax != null && g.width > gripperMax) return;
    cand.push(i);
  });
  if (!cand.length) return [null, -1];

  const tp = opts.targetPoint;
  const nearD = new Map<number, number>();
  if (tp != null) for (const i of cand) nearD.set(i, Math.hypot(grasps[i]!.x - tp[0], grasps[i]!.y - tp[1]));
  const maxNear = nearD.size ? Math.max(...nearD.values()) : 0;

  const target = opts.targetDistance;
  let camD: Array<number | null> = [];
  let sigma = opts.distanceSigma;
  if (target != null) {
    if (opts.depth == null || opts.intrinsics == null) throw new Error("targetDistance requires depth and intrinsics");
    camD = graspDistances(opts.depth, grasps, opts.intrinsics, { window: opts.window, depthScale: opts.depthScale });
    if (sigma == null) sigma = Math.max(1e-6, 0.15 * Math.abs(target));
  }

  let widthMid: number | null = null;
  let widthHalf: number | null = null;
  if (gripperMin != null && gripperMax != null && gripperMax > gripperMin) {
    widthMid = (gripperMin + gripperMax) / 2;
    widthHalf = (gripperMax - gripperMin) / 2;
  }

  const w = resolveWeights(opts.weights, nearD.size > 0, target != null, "quality");
  let best = -1;
  let bestScore = -Infinity;
  for (const i of cand) {
    const g = grasps[i]!;
    let s = 0;
    let wsum = 0;
    if ((w.quality ?? 0) > 0) {
      s += w.quality! * g.quality;
      wsum += w.quality!;
    }
    if ((w.near ?? 0) > 0 && nearD.size) {
      s += w.near! * (maxNear > 0 ? 1 - nearD.get(i)! / maxNear : 1);
      wsum += w.near!;
    }
    if ((w.distance ?? 0) > 0 && target != null) {
      const cd = camD[i];
      s += w.distance! * (cd == null ? 0 : gaussCloseness(cd, target, sigma!));
      wsum += w.distance!;
    }
    if ((w.width ?? 0) > 0 && widthMid != null) {
      s += w.width! * Math.max(0, 1 - Math.abs(g.width - widthMid) / widthHalf!);
      wsum += w.width!;
    }
    const score = wsum > 0 ? s / wsum : 0;
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best >= 0 ? [grasps[best]!, best] : [null, -1];
}
