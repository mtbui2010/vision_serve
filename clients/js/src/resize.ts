/**
 * Client-side resizing: shrink a large image to what the model can use before uploading it
 * (the same rules as the Python SDK's `visionserve/resize.py`).
 *
 * A model resizes every image to its own fixed input on the server (RF-DETR 560x560,
 * GroundingDINO 800x800, CLIP 224x224, ...), so most of a 12 MP photo's upload is thrown away.
 * `GET /api/models` publishes per model how far an image can be shrunk without changing what the
 * model sees (`max_useful_side` bounds the LONGER side, `max_useful_short_side` the SHORTER side;
 * null = never shrink). The {@link Client} shrinks to it, sends JPEG, and maps the results back
 * to ORIGINAL image pixels.
 *
 * Decoding needs an {@link ImageCodec}. The SDK has no image dependency of its own:
 * - in a browser (or a worker) it uses `createImageBitmap` + `OffscreenCanvas`;
 * - in Node it uses `sharp` when the application has installed it (an optional peer, never
 *   installed by this package);
 * - otherwise there is no codec and every image is sent exactly as given (results are just as
 *   correct; only the upload is larger).
 */

import { Detection, Grasp, Mask, Result } from "./types.js";

/** What the client did to the image before uploading it (`Result.clientResize`). */
export class ClientResize {
  /** The image as the server would have seen it (after EXIF rotation): every returned coordinate is in this frame. */
  readonly originalWidth: number;
  readonly originalHeight: number;
  /** The image actually uploaded. */
  readonly sentWidth: number;
  readonly sentHeight: number;
  /** JPEG quality it was encoded at, or `null` when it was sent as PNG (`jpeg: false`). */
  readonly jpegQuality: number | null;

  constructor(originalWidth: number, originalHeight: number, sentWidth: number, sentHeight: number, jpegQuality: number | null) {
    this.originalWidth = originalWidth;
    this.originalHeight = originalHeight;
    this.sentWidth = sentWidth;
    this.sentHeight = sentHeight;
    this.jpegQuality = jpegQuality;
  }

  /** True when the uploaded image is smaller than the original (else it was only re-encoded). */
  get resized(): boolean {
    return this.sentWidth !== this.originalWidth || this.sentHeight !== this.originalHeight;
  }

  // Multiply first, then divide: 300 * 1680 / 3000 is exactly 168, 300 * (1680 / 3000) is not.
  toSentX(x: number): number {
    return (x * this.sentWidth) / this.originalWidth;
  }
  toSentY(y: number): number {
    return (y * this.sentHeight) / this.originalHeight;
  }
  toOriginalX(x: number): number {
    return (x * this.originalWidth) / this.sentWidth;
  }
  toOriginalY(y: number): number {
    return (y * this.originalHeight) / this.sentHeight;
  }
}

/** `"auto"` (the server's hint), `"off"` (send as given), or a longest side in pixels. */
export type ResizeOption = "auto" | "off" | number;

/** Validate a `resize` option. */
export function checkResize(v: unknown): ResizeOption {
  if (v === "auto" || v === "off") return v;
  if (typeof v === "number" && Number.isInteger(v) && v > 0) return v;
  throw new TypeError(`resize must be "auto", "off" or a positive integer, got ${JSON.stringify(v) ?? String(v)}`);
}

/** Validate `jpegQuality`: an integer in 1..100. */
export function checkQuality(v: unknown): number {
  if (typeof v === "number" && Number.isInteger(v) && v >= 1 && v <= 100) return v;
  throw new TypeError(`jpegQuality must be an integer in 1..100, got ${JSON.stringify(v) ?? String(v)}`);
}

const roundHalfUp = (x: number) => Math.floor(x + 0.5);
/** Go's math.Round (half away from zero), as the server's ROI clamp uses. */
const roundGo = (x: number) => Math.sign(x) * Math.floor(Math.abs(x) + 0.5);

/**
 * The size to send a `width` x `height` image at (unchanged when small enough). `maxSide` bounds
 * the LONGER side, `maxShortSide` the SHORTER side (give one); with `region` (the ROI's `[w, h]`)
 * the bound applies to the region and the whole image is scaled by the same factor. Each side is
 * rounded half up, never below 1.
 */
export function targetSize(
  width: number,
  height: number,
  opts: { maxSide?: number | null; maxShortSide?: number | null; region?: [number, number] | null },
): [number, number] {
  const [rw, rh] = opts.region ?? [width, height];
  let s: number;
  if (opts.maxSide) s = opts.maxSide / Math.max(rw, rh);
  else if (opts.maxShortSide) s = opts.maxShortSide / Math.min(rw, rh);
  else return [width, height];
  if (s >= 1) return [width, height];
  return [Math.max(1, roundHalfUp(width * s)), Math.max(1, roundHalfUp(height * s))];
}

/**
 * `[w, h]` in pixels of the region the server crops for `roi` (`internal/roi.Clamp`: fractions
 * when both w and h are <= 1, rounded, clamped), or `null` when unset or degenerate.
 */
export function roiRegion(roi: number[] | null | undefined, width: number, height: number): [number, number] | null {
  if (roi == null || roi.length !== 4) return null;
  let [x, y, w, h] = roi as [number, number, number, number];
  if (!(w > 0) || !(h > 0)) return null;
  if (w <= 1 && h <= 1) [x, y, w, h] = [x * width, y * height, w * width, h * height];
  const x0 = Math.max(0, roundGo(x));
  const y0 = Math.max(0, roundGo(y));
  const x1 = Math.min(width, roundGo(x + w));
  const y1 = Math.min(height, roundGo(y + h));
  if (x1 - x0 < 1 || y1 - y0 < 1) return null;
  return [x1 - x0, y1 - y0];
}

// ---------------------------------------------------------------------- //
// Header probe (JPEG / PNG): size, format and EXIF orientation without decoding
// ---------------------------------------------------------------------- //

/** What a codec reads from encoded bytes. `width`/`height` are AFTER the EXIF rotation the server applies. */
export interface ImageProbe {
  width: number;
  height: number;
  /** True for a JPEG — the only format whose EXIF orientation the server applies. */
  isJpeg: boolean;
}

/** Size + format from a JPEG or PNG header (EXIF orientation applied for JPEG), or `null`. */
export function probeHeader(b: Uint8Array): ImageProbe | null {
  if (b.length >= 24 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) {
    const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
    return { width: v.getUint32(16), height: v.getUint32(20), isJpeg: false };
  }
  if (b.length < 4 || b[0] !== 0xff || b[1] !== 0xd8) return null;
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let orientation = 1;
  let i = 2;
  while (i + 4 <= b.length) {
    if (b[i] !== 0xff) return null;
    const marker = b[i + 1]!;
    if (marker === 0xff) {
      i++; // fill byte
      continue;
    }
    if (marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01) {
      i += 2;
      continue;
    }
    const len = v.getUint16(i + 2);
    if (len < 2 || i + 2 + len > b.length) return null;
    if (marker === 0xe1 && orientation === 1) orientation = exifOrientation(b, i + 4, len - 2);
    // SOF0..SOF15 except DHT (C4), JPG (C8), DAC (CC)
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      if (len < 7) return null;
      const h = v.getUint16(i + 5);
      const w = v.getUint16(i + 7);
      const swap = orientation >= 5 && orientation <= 8;
      return { width: swap ? h : w, height: swap ? w : h, isJpeg: true };
    }
    if (marker === 0xda) return null; // start of scan before any frame header
    i += 2 + len;
  }
  return null;
}

/** The EXIF Orientation tag (1..8) of an APP1 payload at b[off .. off+len), or 1. */
function exifOrientation(b: Uint8Array, off: number, len: number): number {
  if (len < 14 || String.fromCharCode(...b.subarray(off, off + 4)) !== "Exif") return 1;
  const t = off + 6; // TIFF header
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const end = off + len;
  const le = b[t] === 0x49 && b[t + 1] === 0x49;
  if (!le && !(b[t] === 0x4d && b[t + 1] === 0x4d)) return 1;
  const ifd = t + v.getUint32(t + 4, le);
  if (ifd + 2 > end) return 1;
  const n = v.getUint16(ifd, le);
  for (let k = 0; k < n; k++) {
    const e = ifd + 2 + 12 * k;
    if (e + 12 > end) return 1;
    if (v.getUint16(e, le) === 0x0112) {
      const o = v.getUint16(e + 8, le);
      return o >= 1 && o <= 8 ? o : 1;
    }
  }
  return 1;
}

// ---------------------------------------------------------------------- //
// Codecs
// ---------------------------------------------------------------------- //

/**
 * Decodes, shrinks and encodes images for client-side resizing. Plug your own into
 * `new Client(host, { codec })` (e.g. a WASM decoder), or pass `codec: null` to disable it.
 */
export interface ImageCodec {
  /** Size (after EXIF rotation) and format of encoded bytes, or `null` if unreadable. */
  probe(bytes: Uint8Array): Promise<ImageProbe | null>;
  /**
   * Decode `bytes` (applying a JPEG's EXIF orientation, dropping alpha), resize to exactly
   * `width` x `height` with an antialiasing filter, and encode as JPEG at `quality` (1..100)
   * or, with `jpeg: false`, as PNG.
   */
  transcode(bytes: Uint8Array, width: number, height: number, opts: { jpeg: boolean; quality: number }): Promise<Uint8Array>;
}

/**
 * The browser codec: `createImageBitmap` (EXIF-aware) + `OffscreenCanvas` (high-quality
 * smoothing). Two limits of the canvas: its JPEG encoder picks its own chroma subsampling
 * (typically 4:2:0, which the Python SDK and `sharp` avoid — see the docs for what it costs
 * an embedding model), and it premultiplies alpha, so transparent pixels come out black (the
 * server would have read their stored colour). Use `jpeg: false` or `resize: "off"` where that
 * matters. `null` outside a browser/worker.
 */
export function browserCodec(): ImageCodec | null {
  const g = globalThis as unknown as {
    createImageBitmap?: typeof createImageBitmap;
    OffscreenCanvas?: typeof OffscreenCanvas;
  };
  if (typeof g.createImageBitmap !== "function" || typeof g.OffscreenCanvas !== "function") return null;
  const toBlob = (bytes: Uint8Array) => new Blob([bytes as unknown as BlobPart]);
  return {
    async probe(bytes) {
      const h = probeHeader(bytes);
      if (h) return h;
      try {
        const bmp = await g.createImageBitmap!(toBlob(bytes), { imageOrientation: "from-image" });
        const out = { width: bmp.width, height: bmp.height, isJpeg: false };
        bmp.close();
        return out;
      } catch {
        return null;
      }
    },
    async transcode(bytes, width, height, opts) {
      const bmp = await g.createImageBitmap!(toBlob(bytes), { imageOrientation: "from-image" });
      try {
        const canvas = new g.OffscreenCanvas!(width, height);
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("OffscreenCanvas 2d context unavailable");
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, width, height);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(bmp, 0, 0, width, height);
        const blob = await canvas.convertToBlob(
          opts.jpeg ? { type: "image/jpeg", quality: opts.quality / 100 } : { type: "image/png" },
        );
        return new Uint8Array(await blob.arrayBuffer());
      } finally {
        bmp.close();
      }
    },
  };
}

/** The shape of the `sharp` API this file uses (sharp is an optional peer, not a dependency). */
interface SharpLike {
  (input: Uint8Array): SharpPipeline;
}
interface SharpPipeline {
  metadata(): Promise<{ width?: number; height?: number; format?: string; orientation?: number }>;
  rotate(): SharpPipeline;
  removeAlpha(): SharpPipeline;
  resize(w: number, h: number, o: { fit: string }): SharpPipeline;
  jpeg(o: { quality: number; chromaSubsampling: string }): SharpPipeline;
  png(o: { compressionLevel: number }): SharpPipeline;
  toBuffer(): Promise<Uint8Array>;
}

/** A codec over the `sharp` module (pass `(await import("sharp")).default`). */
export function sharpCodec(sharp: SharpLike): ImageCodec {
  return {
    async probe(bytes) {
      const h = probeHeader(bytes);
      if (h) return h;
      try {
        const m = await sharp(bytes).metadata();
        if (!m.width || !m.height) return null;
        return { width: m.width, height: m.height, isJpeg: m.format === "jpeg" };
      } catch {
        return null;
      }
    },
    async transcode(bytes, width, height, opts) {
      let p = sharp(bytes);
      if (probeHeader(bytes)?.isJpeg) p = p.rotate(); // EXIF orientation: the server applies it to JPEG only
      p = p.removeAlpha().resize(width, height, { fit: "fill" }); // sharp's default kernel: Lanczos-3
      // 4:4:4 chroma: at ~2x the model's input, 4:2:0 would halve the colour to the model's own
      // resolution (measured with the Python SDK on CLIP: mean cosine 0.976 vs 0.995).
      p = opts.jpeg ? p.jpeg({ quality: opts.quality, chromaSubsampling: "4:4:4" }) : p.png({ compressionLevel: 1 });
      return new Uint8Array(await p.toBuffer());
    },
  };
}

let defaultCodecPromise: Promise<ImageCodec | null> | null = null;

/**
 * The codec the client uses when none is given: the browser's, else `sharp` if the application
 * installed it, else `null` (no client-side resizing). Resolved once per process.
 */
export function defaultCodec(): Promise<ImageCodec | null> {
  if (!defaultCodecPromise) {
    defaultCodecPromise = (async () => {
      const b = browserCodec();
      if (b) return b;
      try {
        const name = "sharp"; // a variable: bundlers must not try to resolve it
        const mod = (await import(/* @vite-ignore */ /* webpackIgnore: true */ name)) as { default?: unknown };
        const sharp = (mod.default ?? mod) as SharpLike;
        return typeof sharp === "function" ? sharpCodec(sharp) : null;
      } catch {
        return null;
      }
    })();
  }
  return defaultCodecPromise;
}

// ---------------------------------------------------------------------- //
// The upload decision
// ---------------------------------------------------------------------- //

export interface PreparedUpload {
  bytes: Uint8Array;
  /** null: the bytes are the input's own, untouched. */
  clientResize: ClientResize | null;
}

/**
 * The bytes to upload for an encoded image under a hint (the Python SDK's `prepare_upload`):
 * no hint or no codec -> the input untouched; larger than the hint (measured on the ROI region
 * when one is given) -> shrunk and sent as JPEG (PNG with `jpeg: false`); within it -> a JPEG
 * untouched, anything else re-encoded as JPEG when `jpeg` is on, else untouched.
 */
export async function prepareUpload(
  bytes: Uint8Array,
  opts: {
    maxSide: number | null;
    maxShortSide: number | null;
    jpeg: boolean;
    quality: number;
    roi?: number[] | null;
    codec: ImageCodec | null;
  },
): Promise<PreparedUpload> {
  const untouched = { bytes, clientResize: null };
  if ((!opts.maxSide && !opts.maxShortSide) || !opts.codec) return untouched;
  const info = await opts.codec.probe(bytes);
  if (!info || !(info.width > 0) || !(info.height > 0)) return untouched;
  const { width: w, height: h } = info;
  const [tw, th] = targetSize(w, h, {
    maxSide: opts.maxSide,
    maxShortSide: opts.maxShortSide,
    region: opts.roi ? roiRegion(opts.roi, w, h) : null,
  });
  const resized = tw !== w || th !== h;
  if (!resized && (info.isJpeg || !opts.jpeg)) return untouched;
  const out = await opts.codec.transcode(bytes, tw, th, { jpeg: opts.jpeg, quality: opts.quality });
  return { bytes: out, clientResize: new ClientResize(w, h, tw, th, opts.jpeg ? opts.quality : null) };
}

/** Boxes `[x, y, w, h]` from ORIGINAL to sent pixels (a malformed entry is left for the serializer to refuse). */
export function scaleBoxes(boxes: number[][], cr: ClientResize): number[][] {
  return boxes.map((b) =>
    Array.isArray(b) && b.length === 4 && b.every((v) => typeof v === "number")
      ? [cr.toSentX(b[0]!), cr.toSentY(b[1]!), cr.toSentX(b[2]!), cr.toSentY(b[3]!)]
      : b,
  );
}

/** Points `[x, y(, label)]` from ORIGINAL to sent pixels (the label is kept). */
export function scalePoints(points: number[][], cr: ClientResize): number[][] {
  return points.map((p) =>
    Array.isArray(p) && (p.length === 2 || p.length === 3) && p.every((v) => typeof v === "number")
      ? [cr.toSentX(p[0]!), cr.toSentY(p[1]!), ...p.slice(2)]
      : p,
  );
}

/** A pixel ROI to sent pixels; a fractional one (w and h <= 1) is unchanged. */
export function scaleRoi(roi: number[], cr: ClientResize): number[] {
  if (roi.length === 4 && roi[2]! <= 1 && roi[3]! <= 1) return roi;
  return scaleBoxes([roi], cr)[0]!;
}

/**
 * A Result computed on the sent image, mapped back to ORIGINAL pixels: detection and mask
 * boxes, grasp centres, jaw widths and angles (exactly, per axis). Masks keep their RLE at the
 * sent size and record it (`Mask.rleSize`), so `toMask(originalW, originalH)` returns an
 * original-size mask. The depth map stays at the model's resolution, as always.
 */
export function mapResult(r: Result, cr: ClientResize): Result {
  const box = (b: number[]) => [cr.toOriginalX(b[0] ?? 0), cr.toOriginalY(b[1] ?? 0), cr.toOriginalX(b[2] ?? 0), cr.toOriginalY(b[3] ?? 0)];
  const sx = cr.sentWidth / cr.originalWidth;
  const sy = cr.sentHeight / cr.originalHeight;
  const resized = cr.resized;
  return new Result(
    r.task,
    r.model,
    resized ? r.detections.map((d) => new Detection(box(d.bbox), d.cls, d.conf)) : r.detections,
    resized
      ? r.masks.map((m) => new Mask(m.rle, box(m.bbox), m.conf, m.rle ? [cr.sentWidth, cr.sentHeight] : null))
      : r.masks,
    r.classifications,
    r.depthMap,
    r.depthWidth,
    r.depthHeight,
    r.embeddings,
    r.durationMs,
    resized
      ? r.grasps.map((g) => {
          const dx = Math.cos(g.theta) / sx;
          const dy = Math.sin(g.theta) / sy;
          return new Grasp(cr.toOriginalX(g.x), cr.toOriginalY(g.y), Math.atan2(dy, dx), g.width * Math.hypot(dx, dy), g.quality, g.cls, g.conf);
        })
      : r.grasps,
    r.device,
    r.hint,
    cr,
  );
}
