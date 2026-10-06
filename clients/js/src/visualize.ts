import { encodePalettePNG } from "./png.js";
import { Detection, Grasp, Mask, Result, bytesToBase64, topGraspsPerObject } from "./types.js";

type RGB = [number, number, number];

/** Index palette (`colorBy: "index"`): the Go server's overlay palette, as in the Python client. @internal */
export const PALETTE: RGB[] = [
  [255, 59, 59],
  [255, 165, 0],
  [50, 205, 50],
  [0, 191, 255],
  [238, 130, 238],
  [255, 215, 0],
  [0, 255, 127],
  [255, 99, 71],
];

/**
 * Class palette (`colorBy: "class"`): the Python client's 16 well-separated colours
 * (Trubetskoy's list without near-white, grey, black, navy and maroon).
 */
export const CLASS_PALETTE: RGB[] = [
  [230, 25, 75], // red
  [60, 180, 75], // green
  [255, 225, 25], // yellow
  [0, 130, 200], // blue
  [245, 130, 48], // orange
  [145, 30, 180], // purple
  [70, 240, 240], // cyan
  [240, 50, 230], // magenta
  [210, 245, 60], // lime
  [250, 190, 212], // pink
  [0, 128, 128], // teal
  [220, 190, 255], // lavender
  [170, 110, 40], // brown
  [170, 255, 195], // mint
  [128, 128, 0], // olive
  [255, 215, 180], // apricot
];

/** Highlight colour of the selected target box and grasp. */
const TARGET: RGB = [255, 0, 0];

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
   * `"class"` (default, as Python's `draw()`): every item of one class gets one colour of a
   * 16-colour palette, picked by a stable hash of the class name (FNV-1a), so a class keeps its
   * colour across pictures and frames; when two classes of one picture land on the same colour,
   * the one later in alphabetical order takes the next free one. Items without a class (masks
   * no detection labels) fall back to their index. `"index"`: item i gets colour i of the
   * server's 8-colour overlay palette.
   */
  colorBy?: "class" | "index";
  /**
   * Label font size in pixels of the photo. Default: scaled with the photo's shorter side
   * (`max(12, min(160, round(short / 30)))`: 14 on a 640 x 426 photo), as Python's `draw()`.
   */
  fontSize?: number;
  /** Box line width in pixels of the photo. Default `max(1, min(40, round(short / 210)))`: 2 on 640 x 426. */
  lineWidth?: number;
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
  const colorBy = opts.colorBy ?? "class";
  if (colorBy !== "class" && colorBy !== "index") throw new Error(`colorBy must be "class" or "index", got ${JSON.stringify(colorBy)}`);
  const maxGrasps = opts.maxGraspsPerObject === undefined ? 3 : opts.maxGraspsPerObject;
  const st = style(width, height, opts.fontSize, opts.lineWidth);
  const parts: string[] = [];

  const target = opts.targetBox ?? null;
  const targetBBox = target == null ? null : bboxOf(target);
  const isTarget = (it: Detection | Mask) =>
    target != null && (it === target || (targetBBox != null && sameBox(it.bbox, targetBBox)));

  // Colours: by class (a per-picture map, so two classes never share one) or by index.
  const classCols =
    colorBy === "class"
      ? classColourMap([...result.detections.map((d) => d.cls), ...result.classifications.map((c) => c.cls)].filter(Boolean))
      : new Map<string, RGB>();
  const itemColour = (i: number, cls: string): RGB =>
    colorBy === "class" && cls ? (classCols.get(cls) ?? classColour(cls)) : PALETTE[i % PALETTE.length]!;
  // Masks carry no class: one whose box equals a detection's box takes that detection's label.
  const detLabel = new Map<string, string>();
  for (const d of result.detections) if (!detLabel.has(boxKey(d.bbox))) detLabel.set(boxKey(d.bbox), d.cls);
  const maskColour = (m: Mask, i: number) => itemColour(i, detLabel.get(boxKey(m.bbox)) ?? "");

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
    const c = itemColour(i, det.cls);
    parts.push(box(det.bbox, c, st.lw), label(st, det.bbox[0] ?? 0, det.bbox[1] ?? 0, detText(det), c));
  });
  if (targetDet) {
    const d: Detection = targetDet;
    parts.push(box(d.bbox, TARGET, 2 * st.lw), label(st, d.bbox[0] ?? 0, d.bbox[1] ?? 0, detText(d), TARGET));
  }

  // 3. Boxes of the masks no detection labels.
  const maskBoxes = opts.maskBoxes !== false;
  result.masks.forEach((m, i) => {
    if (detLabel.has(boxKey(m.bbox))) return;
    const t = isTarget(m);
    if (!maskBoxes && !t) return;
    const c = t ? TARGET : maskColour(m, i);
    parts.push(box(m.bbox, c, t ? 2 * st.lw : st.lw), label(st, m.bbox[0] ?? 0, m.bbox[1] ?? 0, `mask ${pct(m.conf)}`, c));
  });

  // 4. Grasps: the best few per object; the target on top.
  if (result.grasps.length) {
    const shown =
      maxGrasps == null || maxGrasps <= 0 ? result.grasps.slice() : topGraspsPerObject(result.grasps, result.detections, result.masks, maxGrasps);
    for (const g of shown) if (g !== opts.targetGrasp) parts.push(graspGlyph(st, g, false));
    if (opts.targetGrasp && shown.includes(opts.targetGrasp)) parts.push(graspGlyph(st, opts.targetGrasp, true));
  }

  // 5. Classification labels, top-left, on a dark band.
  const cfs = Math.round((st.font * 16) / 14);
  const margin = Math.round((st.font * 20) / 14);
  const lineHeight = Math.round((st.font * 30) / 14);
  const cpad = Math.max(2, Math.round((st.font * 4) / 14));
  result.classifications.forEach((c, i) => {
    const text = `${c.cls} ${pct(c.conf)}`;
    const ty = margin + i * lineHeight;
    const tw = textWidth(text, cfs);
    parts.push(
      `<rect x="${n(margin - cpad)}" y="${n(ty - cpad)}" width="${n(tw + 2 * cpad)}" height="${n(cfs + 2 * cpad)}" fill="rgb(20,20,20)"/>` +
        `<text x="${margin}" y="${n(ty + cfs * 0.8)}" font-family="sans-serif" font-size="${cfs}" fill="${rgb(itemColour(i, c.cls))}" ` +
        `textLength="${n(tw)}" lengthAdjust="spacingAndGlyphs">${escapeXML(text)}</text>`,
    );
  });

  // 6. A target box that is none of this result's items (e.g. chosen on another result).
  if (targetBBox && ![...result.detections, ...result.masks].some(isTarget)) parts.push(box(targetBBox, TARGET, 2 * st.lw));

  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">` +
    parts.join("") +
    `</svg>`
  );
}

/** Font size, line width and picture size shared by every drawing step. @internal */
export interface Style {
  font: number;
  lw: number;
  width: number;
  height: number;
}

/**
 * Sizes for a picture (Python's `_auto_sizes`): proportional to its shorter side, clamped, so
 * labels read the same on a thumbnail and on a 4000 x 3000 photo. Explicit values win.
 * @internal
 */
export function style(width: number, height: number, fontSize?: number, lineWidth?: number): Style {
  const short = Math.max(1, Math.min(Math.trunc(width), Math.trunc(height)));
  const font = fontSize != null ? Math.trunc(fontSize) : Math.max(12, Math.min(160, roundHalfEven(short / 30)));
  const lw = lineWidth != null ? Math.trunc(lineWidth) : Math.max(1, Math.min(40, roundHalfEven(short / 210)));
  if (!(font >= 1) || !(lw >= 1)) throw new Error(`fontSize and lineWidth must be >= 1, got ${fontSize} and ${lineWidth}`);
  return { font, lw, width, height };
}

/** Python's `round()`: half to even. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** 32-bit FNV-1a of the UTF-8 bytes: a stable hash, the same in every SDK. @internal */
export function fnv1a(text: string): number {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(text)) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/**
 * `{ class: colour }` for the classes of one picture (Python's `_class_colour_map`): each class
 * starts at `CLASS_PALETTE[fnv1a(name) % 16]`; when two land on the same colour, the one later in
 * alphabetical order takes the next free colour (up to 16 classes are always distinct).
 */
export function classColourMap(names: Iterable<string>): Map<string, RGB> {
  const k = CLASS_PALETTE.length;
  const used = new Set<number>();
  const out = new Map<string, RGB>();
  // Python's sorted(): by code point (not localeCompare).
  for (const name of [...new Set(names)].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))) {
    let idx = fnv1a(name) % k;
    while (used.has(idx) && used.size < k) idx = (idx + 1) % k;
    used.add(idx);
    out.set(name, CLASS_PALETTE[idx]!);
  }
  return out;
}

/**
 * A class's own colour (`[r, g, b]`): `CLASS_PALETTE[fnv1a(utf8(label)) % 16]`, the same in the
 * Python SDK. In one picture `toSVG` may move a class to the next free colour when two classes
 * collide ({@link classColourMap}).
 */
export function classColour(label: string): RGB {
  return CLASS_PALETTE[fnv1a(label) % CLASS_PALETTE.length]!;
}

/** Black or white, whichever reads better on `bg` (Python's `_text_colour`). @internal */
export function textColour(bg: RGB): RGB {
  const lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2];
  return lum > 150 ? [0, 0, 0] : [255, 255, 255];
}

/** Grasp quality in [0, 1] → red → yellow → green (Python's `_quality_colour`). */
function qualityColour(q: number): RGB {
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
  colourOf: (m: Mask, i: number) => RGB,
): string | null {
  const pixels = new Uint8Array(width * height);
  const palette: Array<[number, number, number, number]> = [[0, 0, 0, 0]];
  const slot = new Map<string, number>();
  const a = roundHalfEven(alpha * 255); // Python: int(round(alpha * 255))
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

function graspGlyph(st: Style, g: Grasp, isTarget: boolean): string {
  const colour = isTarget ? TARGET : qualityColour(g.quality);
  const c = rgb(colour);
  const extra = Math.max(1, Math.floor(st.lw / 2));
  const lw = isTarget ? st.lw + extra : st.lw;
  const [x0, y0, x1, y1] = g.contactsFlat();
  const plate = Math.max(3 * st.lw, Math.min(g.width * 0.35, 11 * st.lw));
  const px = (-Math.sin(g.theta) * plate) / 2;
  const py = (Math.cos(g.theta) * plate) / 2;
  const line = (ax: number, ay: number, bx: number, by: number, w: number) =>
    `<line x1="${n(ax)}" y1="${n(ay)}" x2="${n(bx)}" y2="${n(by)}" stroke="${c}" stroke-width="${w}" stroke-linecap="round"/>`;
  const text = (g.cls ? `${g.cls} ` : "") + `q${fixed(g.quality, 2)}`;
  return (
    `<g>` +
    line(x0, y0, x1, y1, lw) +
    line(x0 - px, y0 - py, x0 + px, y0 + py, lw + 1) +
    line(x1 - px, y1 - py, x1 + px, y1 + py, lw + 1) +
    `<circle cx="${n(g.x)}" cy="${n(g.y)}" r="${isTarget ? st.lw + extra : st.lw}" fill="${c}"/>` +
    label(st, g.x, g.y - 2 * st.lw, text, colour) +
    `</g>`
  );
}

/** A box outline `[x, y, w, h]`. */
function box(bbox: readonly number[], colour: RGB, thickness: number): string {
  const [x = 0, y = 0, w = 0, h = 0] = bbox;
  return `<rect x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" fill="none" stroke="${rgb(colour)}" stroke-width="${thickness}"/>`;
}

/**
 * A label on a band of `colour` just above `(x, y)`, kept inside the picture (at the top edge it
 * overlaps the box), black or white text (Python's `_draw_label`).
 */
function label(st: Style, x: number, y: number, text: string, colour: RGB): string {
  const fs = st.font;
  const pad = Math.max(2, Math.floor(fs / 7));
  const tw = textWidth(text, fs);
  const bw = tw + 2 * pad;
  const bh = fs + 2 * pad;
  const tx = Math.max(0, Math.min(x, st.width - bw));
  const ty = Math.max(0, Math.floor(y) - bh);
  // textLength makes the browser fit the text to the band whatever font it picks.
  return (
    `<rect x="${n(tx)}" y="${n(ty)}" width="${n(bw)}" height="${n(bh)}" fill="${rgb(colour)}"/>` +
    `<text x="${n(tx + pad)}" y="${n(ty + pad + fs * 0.8)}" font-family="sans-serif" font-size="${fs}" fill="${rgb(textColour(colour))}" ` +
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
/** Python's `"%.0f%%" % (conf * 100)`. */
const pct = (conf: number) => `${fixed(conf * 100, 0)}%`;

/**
 * Python's `"%.Nf"`: correct rounding of the exact binary value, ties to even (`toFixed` rounds
 * ties up: 0.125 is "0.13" in JS, "0.12" in Python).
 */
function fixed(x: number, digits: number): string {
  const r = x * 10 ** digits;
  if (Number.isFinite(r) && Math.abs(r % 1) === 0.5) return (roundHalfEven(r) / 10 ** digits).toFixed(digits);
  return x.toFixed(digits);
}
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
