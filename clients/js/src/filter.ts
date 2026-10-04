import { Result } from "./types.js";

export interface SizeFilterOptions {
  /**
   * Minimum bbox area. If `imageWidth` + `imageHeight` are given: fraction of image area
   * (0..1). Otherwise: absolute pixels². `0` = no lower limit.
   */
  minSize?: number;
  /**
   * Maximum bbox area. If `imageWidth` + `imageHeight` are given: fraction of image area
   * (0..1). Otherwise: absolute pixels². `0` = no upper limit.
   */
  maxSize?: number;
  /** Original image width — required for relative (fraction) mode. */
  imageWidth?: number;
  /** Original image height — required for relative (fraction) mode. */
  imageHeight?: number;
}

/**
 * Return a new {@link Result} keeping only detections and masks whose bounding-box
 * area is within the given range.
 *
 * Area is `bbox[2] * bbox[3]` (width × height). When `imageWidth` and `imageHeight`
 * are both provided the thresholds are treated as fractions of the full image area
 * (`imageWidth * imageHeight`); otherwise they are absolute pixel² values.
 */
export function filterBySize(result: Result, opts: SizeFilterOptions): Result {
  const { minSize = 0, maxSize = 0, imageWidth, imageHeight } = opts;

  const imageArea =
    imageWidth != null && imageHeight != null && imageWidth > 0 && imageHeight > 0
      ? imageWidth * imageHeight
      : null;

  const minAbs = imageArea != null ? minSize * imageArea : minSize;
  const maxAbs = imageArea != null ? maxSize * imageArea : maxSize;

  const inRange = (bbox: number[]): boolean => {
    const area = (bbox[2] ?? 0) * (bbox[3] ?? 0);
    if (minAbs > 0 && area < minAbs) return false;
    if (maxAbs > 0 && area > maxAbs) return false;
    return true;
  };

  const filteredDetections = result.detections.filter((d) => inRange(d.bbox));
  const filteredMasks = result.masks.filter((m) => inRange(m.bbox));

  return new Result(
    result.task,
    result.model,
    filteredDetections,
    filteredMasks,
    result.classifications,
    result.depthMap,
    result.depthWidth,
    result.depthHeight,
    result.embeddings,
    result.durationMs,
    result.grasps,
    result.device,
    result.hint,
  );
}

/** How {@link getDepthAtDetection} reduces the depth pixels under one box to one number. */
export type DepthMode = "median" | "mean" | "min" | "max";

/** Options for {@link getDepthAtDetection}. */
export interface DepthAtDetectionOptions {
  /** `"median"` (default), `"mean"`, `"min"` or `"max"`. */
  mode?: DepthMode;
  /** Width of the ORIGINAL image the boxes refer to (the photo sent to both models). */
  imageWidth: number;
  /** Height of the ORIGINAL image the boxes refer to. */
  imageHeight: number;
}

/**
 * The depth under each detection (or each mask, when there are no detections) of `detResult`.
 *
 * A depth model (`midas`, `depth-anything-v2`) answers at the MODEL's resolution
 * (`depthWidth` x `depthHeight`, e.g. 256 x 256) whatever the photo's size, while boxes are in
 * ORIGINAL image pixels. The boxes are therefore scaled onto the map by
 * `depthWidth / imageWidth` and `depthHeight / imageHeight`, so `imageWidth` and `imageHeight`
 * (the photo's size) are required. A map that already has the photo's size is read 1:1.
 *
 * The values are the server's map: RELATIVE inverse depth min-max normalised to `[0, 1]` per
 * image (larger = closer), not metres. Every finite value counts, including `0` (the farthest
 * point of the map). Same rule as the Python `get_depth_at_detection(..., image_size=(W, H))`.
 *
 * @returns one value per box, `null` where the box misses the map.
 * @throws if the depth result has no depth map, the image size is missing or not positive, or
 *   `mode` is unknown.
 */
export function getDepthAtDetection(
  depthResult: Result,
  detResult: Result,
  opts: DepthAtDetectionOptions,
): Array<number | null> {
  const { depthMap, depthWidth, depthHeight } = depthResult;
  const options = (opts ?? {}) as Partial<DepthAtDetectionOptions>;
  const mode = options.mode ?? "median";
  if (!["median", "mean", "min", "max"].includes(mode)) {
    throw new Error(`unknown mode ${JSON.stringify(mode)}; use "median", "mean", "min" or "max"`);
  }
  if (!depthMap.length || depthWidth <= 0 || depthHeight <= 0) {
    throw new Error("depthResult has no depth map");
  }
  const { imageWidth, imageHeight } = options;
  if (imageWidth == null || imageHeight == null) {
    throw new Error(
      `a depth result is at the model's resolution (${depthWidth}x${depthHeight}), not the image's: ` +
        "pass { imageWidth, imageHeight } of the original image so the boxes can be mapped onto it",
    );
  }
  if (!(imageWidth > 0) || !(imageHeight > 0)) {
    throw new Error(`imageWidth and imageHeight must be > 0, got ${imageWidth}x${imageHeight}`);
  }
  const sx = depthWidth / imageWidth;
  const sy = depthHeight / imageHeight;

  const aggregate = (vals: number[]): number | null => {
    if (vals.length === 0) return null;
    if (mode === "min") return vals.reduce((a, b) => (b < a ? b : a));
    if (mode === "max") return vals.reduce((a, b) => (b > a ? b : a));
    if (mode === "mean") return vals.reduce((s, v) => s + v, 0) / vals.length;
    const sorted = vals.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
  };

  const bboxes =
    detResult.detections.length > 0
      ? detResult.detections.map((d) => d.bbox)
      : detResult.masks.map((m) => m.bbox);

  return bboxes.map((bbox) => {
    const [bx = 0, by = 0, bw = 0, bh = 0] = bbox;
    const x0 = Math.max(0, Math.floor(bx * sx));
    const y0 = Math.max(0, Math.floor(by * sy));
    const x1 = Math.min(depthWidth, Math.ceil((bx + bw) * sx));
    const y1 = Math.min(depthHeight, Math.ceil((by + bh) * sy));
    if (x1 <= x0 || y1 <= y0) return null;
    const vals: number[] = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const v = depthMap[y * depthWidth + x];
        if (v != null && Number.isFinite(v)) vals.push(v);
      }
    }
    return aggregate(vals);
  });
}
