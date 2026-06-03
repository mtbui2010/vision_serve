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
