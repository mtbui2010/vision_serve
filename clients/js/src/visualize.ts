import { encodePalettePNG } from "./png.js";
import { Detection, Grasp, Mask, Result, bytesToBase64, topGraspsPerObject } from "./types.js";

/** Colour palette of the Go server's overlay and the Python client. */
const PALETTE: Array<[number, number, number]> = [
  [255, 59, 59],
  [255, 165, 0],
  [50, 205, 50],
  [0, 191, 255],
  [238, 130, 238],
  [255, 215, 0],
  [0, 255, 127],
  [255, 99, 71],
];

/** Highlight colour of the selected target box and grasp. */
const TARGET: [number, number, number] = [255, 0, 0];

/** Options for {@link toSVG}. */
export interface SVGOptions {
  /** Opacity of the mask colour fills, 0 (invisible) to 1. Default 0.45 (Python's `alpha`). */
  alpha?: number;
  /** Paint the mask pixels (default true). `false` draws only their boxes. */
  masks?: boolean;
  /**
   * Draw a box and a `mask conf%` label for each mask that no detection already labels (default
   * true). `false` keeps the fills only (for the dozens of overlapping masks of an automatic-mask
   * result); a target mask still gets its red box.
   */
  maskBoxes?: boolean;
  /**
   * Draw at most this many highest-quality grasps per object (grouped as
   * {@link Result.filterGrasps} does). Default 3, like Python; `null` or `<= 0` draws every grasp.
   */
  maxGraspsPerObject?: number | null;
  /** A grasp of this result (the same object, e.g. from `selectTargetGrasp`) to draw in red, on top. */
  targetGrasp?: Grasp | null;
  /**
   * The selected object to highlight in red with a thicker outline: a `Detection` / `Mask` of this
   * result (matched by identity or by box) or a raw `[x, y, w, h]`. A box that matches none of this
   * result's items is drawn as a standalone red rectangle.
   */
  targetBox?: Detection | Mask | readonly number[] | null;
  /**
   * `"class"` (default): every object of one label has one colour, from a stable hash of the
   * label, so a class keeps its colour across frames and results. `"index"`: the i-th
   * detection / mask / classification gets palette colour i (the Python SDK's colouring).
   * Masks without a label (no detection with the same box) always use the index colour.
   */
  colorBy?: "class" | "index";
}

/**
 * An SVG string drawing a result over its photo, scalable to any CSS size.
 *
 * `width` and `height` are the size of the ORIGINAL photo in pixels (the frame every result
 * coordinate is in; `probeHeader(bytes)` reads it from a JPEG / PNG header), NOT the size it is
 * shown at. The SVG gets `width`/`height` attributes of that size and a `viewBox="0 0 width
 * height"`, so as a standalone file it opens at the photo's size, and laid over an `<img>` with
 * CSS `width: 100%; height: 100%` it follows the image at any display size.
 *
 * What is drawn (the same layers as Python's `draw()`, whatever the task): mask PIXELS as
 * translucent colour fills (one embedded PNG for all masks), detection boxes with `class conf%`
 * labels, a box and `mask conf%` label for each mask no detection labels (Grounded-SAM and the
 * grasp pipelines copy the detection's box onto its mask, so such a mask is labelled once, by its
 * class), grasps as parallel-jaw glyphs (closing line, two jaw plates, centre dot, `class q0.87`;
 * coloured red→yellow→green by quality), and classification labels in the top-left corner.
 * Depth and embedding results have nothing to draw: an empty `<svg>`.
 *
 * A mask whose run counts do not add up to `width * height` (a result for another photo size)
 * gets its box only. The SVG does not contain the photo: lay it over an `<img>`, or insert an
 * `<image href="data:...">` after the opening tag for one standalone file.
 *
 * @param result the prediction result.
 * @param width  the ORIGINAL photo's width in pixels.
 * @param height the ORIGINAL photo's height in pixels.
 */
export function toSVG(result: Result, width: number, height: number, opts: SVGOptions = {}): string {
  const alpha = clamp01(opts.alpha ?? 0.45);
  const byClass = (opts.colorBy ?? "class") === "class";
  const maxGrasps = opts.maxGraspsPerObject === undefined ? 3 : opts.maxGraspsPerObject;
  const parts: string[] = [];

  const target = opts.targetBox ?? null;
  const targetBBox = target == null ? null : bboxOf(target);
  const isTarget = (it: Detection | Mask) =>
    target != null && (it === target || (targetBBox != null && sameBox(it.bbox, targetBBox)));

  // Masks carry no class: one whose box equals a detection's box takes that detection's label.
  const detLabel = new Map<string, string>();
  for (const d of result.detections) if (!detLabel.has(boxKey(d.bbox))) detLabel.set(boxKey(d.bbox), d.cls);
  const detColour = (d: Detection, i: number) => (byClass ? classColour(d.cls) : indexColour(i));
  const maskColour = (m: Mask, i: number) => {
    const label = detLabel.get(boxKey(m.bbox));
    return byClass && label !== undefined ? classColour(label) : indexColour(i);
  };

  // 1. Mask fills: every mask in ONE palette PNG (later masks win where they overlap).
  if (opts.masks !== false && result.masks.length && width > 0 && height > 0) {
    const img = maskImage(result.masks, width, height, alpha, maskColour);
    if (img) {
      parts.push(
        `<image href="data:image/png;base64,${img}" x="0" y="0" width="${width}" height="${height}" ` +
          `preserveAspectRatio="none" style="image-rendering:pixelated"/>`,
      );
    }
  }

  // 2. Detections; the target last, on top, in red.
  let targetDet: Detection | null = null;
  result.detections.forEach((det, i) => {
    if (!targetDet && isTarget(det)) {
      targetDet = det;
      return;
    }
    parts.push(box(det.bbox, detColour(det, i), 2), label(det.bbox[0] ?? 0, det.bbox[1] ?? 0, detText(det), detColour(det, i)));
  });
  if (targetDet) {
    const d: Detection = targetDet;
    parts.push(box(d.bbox, TARGET, 4), label(d.bbox[0] ?? 0, d.bbox[1] ?? 0, detText(d), TARGET));
  }

  // 3. Boxes of the masks no detection labels.
  const maskBoxes = opts.maskBoxes !== false;
  result.masks.forEach((m, i) => {
    if (detLabel.has(boxKey(m.bbox))) return;
    const t = isTarget(m);
    if (!maskBoxes && !t) return;
    const c = t ? TARGET : maskColour(m, i);
    parts.push(box(m.bbox, c, t ? 4 : 2), label(m.bbox[0] ?? 0, m.bbox[1] ?? 0, `mask ${pct(m.conf)}`, c));
  });

  // 4. Grasps: the best few per object; the target on top.
  if (result.grasps.length) {
    const shown =
      maxGrasps == null || maxGrasps <= 0 ? result.grasps.slice() : topGraspsPerObject(result.grasps, result.detections, result.masks, maxGrasps);
    for (const g of shown) if (g !== opts.targetGrasp) parts.push(graspGlyph(g, false));
    if (opts.targetGrasp && shown.includes(opts.targetGrasp)) parts.push(graspGlyph(opts.targetGrasp, true));
  }

  // 5. Classification labels, top-left, on a dark band.
  result.classifications.forEach((c, i) => {
    const text = `${c.cls} ${pct(c.conf)}`;
    const fs = 16;
    const ty = 20 + i * 30;
    const tw = textWidth(text, fs);
    parts.push(
      `<rect x="16" y="${ty - 3}" width="${n(tw + 8)}" height="${fs + 8}" fill="rgb(20,20,20)"/>` +
        `<text x="20" y="${n(ty + fs * 0.82)}" font-family="sans-serif" font-size="${fs}" fill="${rgb(byClass ? classColour(c.cls) : indexColour(i))}" ` +
        `textLength="${n(tw)}" lengthAdjust="spacingAndGlyphs">${escapeXML(text)}</text>`,
    );
  });

  // 6. A target box that is none of this result's items (e.g. chosen on another result).
  if (targetBBox && ![...result.detections, ...result.masks].some(isTarget)) parts.push(box(targetBBox, TARGET, 4));

  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">` +
    parts.join("") +
    `</svg>`
  );
}

/** Palette colour of a class label: FNV-1a (32-bit) of its UTF-8 bytes, modulo the palette. */
export function classColour(label: string): [number, number, number] {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(label)) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return PALETTE[h % PALETTE.length]!;
}

function indexColour(i: number): [number, number, number] {
  return PALETTE[i % PALETTE.length]!;
}

/** Grasp quality in [0, 1] → red → yellow → green (Python's `_quality_colour`). */
function qualityColour(q: number): [number, number, number] {
  q = clamp01(q);
  if (q < 0.5) return [255, Math.trunc(255 * (q / 0.5)), 0];
  return [Math.trunc(255 * (1 - (q - 0.5) / 0.5)), 255, 0];
}

/** All masks as one base64 palette PNG of `width x height`, or null when none could be decoded. */
function maskImage(
  masks: readonly Mask[],
  width: number,
  height: number,
  alpha: number,
  colourOf: (m: Mask, i: number) => [number, number, number],
): string | null {
  const pixels = new Uint8Array(width * height);
  const palette: Array<[number, number, number, number]> = [[0, 0, 0, 0]];
  const slot = new Map<string, number>();
  const a = Math.round(alpha * 255);
  let any = false;
  masks.forEach((m, i) => {
    const c = colourOf(m, i);
    const key = c.join(",");
    let idx = slot.get(key);
    if (idx === undefined) {
      if (palette.length === 256) return; // cannot happen with an 8-colour palette + red
      idx = palette.length;
      palette.push([c[0], c[1], c[2], a]);
      slot.set(key, idx);
    }
    if (paintMask(pixels, m, width, height, idx)) any = true;
  });
  if (!any) return null;
  return bytesToBase64(encodePalettePNG(width, height, pixels, palette));
}

/** Set `value` on the mask's pixels, straight from the column-major RLE; false if it does not fit. */
function paintMask(pixels: Uint8Array, m: Mask, width: number, height: number, value: number): boolean {
  if (!m.rle.trim()) return false;
  if (m.rleSize && (m.rleSize[0] !== width || m.rleSize[1] !== height)) {
    // Encoded at the size the client sent (client-side resize): decode and scale it up.
    let bits: Uint8Array;
    try {
      bits = m.toMask(width, height);
    } catch {
      return false;
    }
    for (let k = 0; k < bits.length; k++) if (bits[k]) pixels[k] = value;
    return true;
  }
  const counts = m.rle.trim().split(/\s+/).map((c) => parseInt(c, 10));
  let sum = 0;
  for (const c of counts) {
    if (!(c >= 0)) return false;
    sum += c;
  }
  if (sum !== width * height) return false;
  let idx = 0;
  counts.forEach((c, r) => {
    if (r % 2 === 1 && c > 0) {
      // Run [idx, idx + c) in column-major order: k -> (x = k / H, y = k % H).
      let x = Math.floor(idx / height);
      let y = idx % height;
      for (let k = 0; k < c; k++) {
        pixels[y * width + x] = value;
        if (++y === height) {
          y = 0;
          x++;
        }
      }
    }
    idx += c;
  });
  return true;
}

function graspGlyph(g: Grasp, isTarget: boolean): string {
  const c = rgb(isTarget ? TARGET : qualityColour(g.quality));
  const lw = isTarget ? 3 : 2;
  const [x0, y0, x1, y1] = g.contactsFlat();
  const plate = Math.max(6, Math.min(g.width * 0.35, 22));
  const px = (-Math.sin(g.theta) * plate) / 2;
  const py = (Math.cos(g.theta) * plate) / 2;
  const line = (ax: number, ay: number, bx: number, by: number, w: number) =>
    `<line x1="${n(ax)}" y1="${n(ay)}" x2="${n(bx)}" y2="${n(by)}" stroke="${c}" stroke-width="${w}" stroke-linecap="round"/>`;
  const text = (g.cls ? `${g.cls} ` : "") + `q${g.quality.toFixed(2)}`;
  return (
    `<g>` +
    line(x0, y0, x1, y1, lw) +
    line(x0 - px, y0 - py, x0 + px, y0 + py, lw + 1) +
    line(x1 - px, y1 - py, x1 + px, y1 + py, lw + 1) +
    `<circle cx="${n(g.x)}" cy="${n(g.y)}" r="${isTarget ? 3 : 2}" fill="${c}"/>` +
    label(g.x, g.y - 4, text, isTarget ? TARGET : qualityColour(g.quality)) +
    `</g>`
  );
}

/** A box outline `[x, y, w, h]`. */
function box(bbox: readonly number[], colour: [number, number, number], thickness: number): string {
  const [x = 0, y = 0, w = 0, h = 0] = bbox;
  return `<rect x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" fill="none" stroke="${rgb(colour)}" stroke-width="${thickness}"/>`;
}

/** A label on a coloured band just above `(x, y)`, white text (Python's `_draw_label`). */
function label(x: number, y: number, text: string, colour: [number, number, number]): string {
  const fs = 14;
  const tw = textWidth(text, fs);
  const th = fs;
  const ty = Math.max(0, y - th - 4);
  // textLength makes the browser fit the text to the band whatever font it picks.
  return (
    `<rect x="${n(x)}" y="${n(ty)}" width="${n(tw + 4)}" height="${th + 4}" fill="${rgb(colour)}" fill-opacity="0.78"/>` +
    `<text x="${n(x + 2)}" y="${n(ty + 2 + fs * 0.82)}" font-family="sans-serif" font-size="${fs}" fill="#fff" ` +
    `textLength="${n(tw)}" lengthAdjust="spacingAndGlyphs">${escapeXML(text)}</text>`
  );
}

/** A width estimate of `text` in a sans-serif font (per-glyph classes of Helvetica/DejaVu widths). */
function textWidth(text: string, fontSize: number): number {
  let em = 0;
  for (const ch of text) {
    if (" .,:;!|'il".includes(ch)) em += 0.3;
    else if ("fjrt()[]-".includes(ch)) em += 0.38;
    else if ("mwMW%@".includes(ch)) em += 0.86;
    else if (ch >= "A" && ch <= "Z") em += 0.68;
    else em += 0.57;
  }
  return em * fontSize;
}

const detText = (d: Detection) => `${d.cls} ${pct(d.conf)}`;
const pct = (conf: number) => `${Math.round(conf * 100)}%`;
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;
/** A coordinate with at most 2 decimals (shorter SVG; a hundredth of a pixel is invisible). */
const n = (v: number) => String(Math.round(v * 100) / 100);
const boxKey = (b: readonly number[]) => b.map(Number).join(",");

function sameBox(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => Number(v) === Number(b[i]));
}

function bboxOf(t: Detection | Mask | readonly number[]): number[] | null {
  const b = Array.isArray(t) ? (t as readonly number[]) : (t as Detection | Mask).bbox;
  return Array.isArray(b) && b.length === 4 ? b.map(Number) : null;
}

/** Escape a string so it is safe inside an SVG text element. */
function escapeXML(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}
