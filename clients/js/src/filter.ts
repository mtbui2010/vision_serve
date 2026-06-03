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
  );
}

export function getDepthAtDetection(
  depthResult: Result,
  detResult: Result,
  mode: "median" | "mean" | "min" | "max" = "median",
): Array<number | null> {
  const { depthMap, depthWidth, depthHeight } = depthResult;

  const aggregate = (pixels: number[]): number | null => {
    const valid = pixels.filter((v) => v !== 0);
    if (valid.length === 0) return null;
    if (mode === "min") return Math.min(...valid);
    if (mode === "max") return Math.max(...valid);
    if (mode === "mean") return valid.reduce((s, v) => s + v, 0) / valid.length;
    const sorted = valid.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  };

  const bboxes =
    detResult.detections.length > 0
      ? detResult.detections.map((d) => d.bbox)
      : detResult.masks.map((m) => m.bbox);

  return bboxes.map((bbox) => {
    const x0 = Math.max(0, Math.floor(bbox[0]));
    const y0 = Math.max(0, Math.floor(bbox[1]));
    const x1 = Math.min(depthWidth, Math.ceil(bbox[0] + bbox[2]));
    const y1 = Math.min(depthHeight, Math.ceil(bbox[1] + bbox[3]));
    const pixels: number[] = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        pixels.push(depthMap[y * depthWidth + x]);
      }
    }
    return aggregate(pixels);
  });
}
