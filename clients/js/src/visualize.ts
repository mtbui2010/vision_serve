import { Result } from "./types.js";

/** Color palette matching the Go server (and Python client). */
const PALETTE = [
  "#FF3B3B",
  "#FFA500",
  "#32CD32",
  "#00BFFF",
  "#EE82EE",
  "#FFD700",
  "#00FF7F",
  "#FF6347",
];

/** Escape a string so it is safe to embed inside an SVG text element. */
function escapeXML(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/**
 * Return an SVG string that overlays detection boxes, mask bboxes, or
 * classification labels over an image.
 *
 * Usage in HTML:
 * ```html
 * <div style="position:relative">
 *   <img src="..." width={w} height={h} />
 *   <svg style="position:absolute;top:0;left:0" innerHTML={toSVG(result, w, h)} />
 * </div>
 * ```
 *
 * - **detection / open_vocab**: colored `<rect>` + `<text>` label per detection.
 * - **segmentation**: colored `<rect>` outline + `<text>` "label conf%" per mask bbox.
 * - **classification**: `<text>` lines in top-left corner listing top-K labels.
 * - **depth / embed**: returns an empty `<svg>` (no meaningful annotation).
 *
 * @param result  prediction result from the server.
 * @param width   display width of the image in CSS pixels.
 * @param height  display height of the image in CSS pixels.
 */
export function toSVG(result: Result, width: number, height: number): string {
  const parts: string[] = [];

  const task = result.task;

  if (task === "detection" || task === "open_vocab") {
    result.detections.forEach((det, i) => {
      const color = PALETTE[i % PALETTE.length] ?? PALETTE[0]!;
      const [x, y, w, h] = [det.bbox[0] ?? 0, det.bbox[1] ?? 0, det.bbox[2] ?? 0, det.bbox[3] ?? 0];
      const label = escapeXML(`${det.cls} ${(det.conf * 100).toFixed(1)}%`);
      // Box
      parts.push(
        `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="3"/>`,
      );
      // Label above the box — clamp to at least y=14 so text is not clipped at the top
      const textY = Math.max(y - 4, 14);
      parts.push(
        `<text x="${x + 2}" y="${textY}" font-family="sans-serif" font-size="13" fill="${color}" ` +
          `stroke="black" stroke-width="0.5" paint-order="stroke">${label}</text>`,
      );
    });
  } else if (task === "segmentation") {
    result.masks.forEach((mask, i) => {
      const color = PALETTE[i % PALETTE.length] ?? PALETTE[0]!;
      const [x, y, w, h] = [
        mask.bbox[0] ?? 0,
        mask.bbox[1] ?? 0,
        mask.bbox[2] ?? 0,
        mask.bbox[3] ?? 0,
      ];
      const label = escapeXML(`mask ${(mask.conf * 100).toFixed(1)}%`);
      // Outline rect
      parts.push(
        `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="3"/>`,
      );
      const textY = Math.max(y - 4, 14);
      parts.push(
        `<text x="${x + 2}" y="${textY}" font-family="sans-serif" font-size="13" fill="${color}" ` +
          `stroke="black" stroke-width="0.5" paint-order="stroke">${label}</text>`,
      );
    });
  } else if (task === "classification") {
    result.classifications.forEach((cls, i) => {
      const color = PALETTE[i % PALETTE.length] ?? PALETTE[0]!;
      const label = escapeXML(`${cls.cls} ${(cls.conf * 100).toFixed(1)}%`);
      const textY = 30 + i * 30;
      parts.push(
        `<text x="16" y="${textY}" font-family="sans-serif" font-size="15" fill="${color}" ` +
          `stroke="black" stroke-width="0.6" paint-order="stroke">${label}</text>`,
      );
    });
  }
  // depth / embed: no annotation — empty svg body

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${parts.join("")}</svg>`;
}
