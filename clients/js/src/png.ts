/**
 * A tiny, synchronous, dependency-free PNG encoder for palette images: what `toSVG` uses to put
 * mask pixels in an SVG as `<image href="data:image/png;base64,...">`.
 *
 * Why this and not the alternatives: `node:zlib` exists only in Node, the browser's
 * `CompressionStream` is asynchronous (and `toSVG` returns a string), and a canvas is browser
 * only. One-pixel-high `<rect>`s or path runs per mask row grow with every mask (tens of KB each,
 * thousands of DOM nodes for an automatic-mask result). Mask pictures are long runs of the same
 * byte, so a deflate stream that only emits literals and back-references at distance 1 (the run)
 * or one row up (the row above), with the fixed Huffman code, is enough: rows are PNG "Up"
 * filtered so a mask's interior becomes zeros and only its edges cost bits. A 640 x 480 picture
 * with a few masks is a few KB; encoding is one pass over the pixels.
 *
 * @internal
 */

/** CRC-32 (PNG chunks), table driven. */
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes: Uint8Array, start: number, end: number): number {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]!) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function adler32(bytes: Uint8Array): number {
  let a = 1;
  let b = 0;
  for (let i = 0; i < bytes.length; ) {
    const end = Math.min(bytes.length, i + 5552); // the largest block without overflow before % 65521
    for (; i < end; i++) {
      a += bytes[i]!;
      b += a;
    }
    a %= 65521;
    b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

/** LSB-first bit writer (deflate's bit order). */
class BitWriter {
  private buf = new Uint8Array(1024);
  private len = 0;
  private acc = 0;
  private nbits = 0;

  /** Write the low `n` bits of `value`, least significant first (n <= 24). */
  bits(value: number, n: number): void {
    this.acc |= value << this.nbits;
    this.nbits += n;
    while (this.nbits >= 8) {
      this.byte(this.acc & 0xff);
      this.acc >>>= 8;
      this.nbits -= 8;
    }
  }

  /** Write a Huffman code of `n` bits, most significant first (deflate's rule for codes). */
  code(code: number, n: number): void {
    let rev = 0;
    for (let i = 0; i < n; i++) rev |= ((code >> i) & 1) << (n - 1 - i);
    this.bits(rev, n);
  }

  private byte(b: number): void {
    if (this.len === this.buf.length) {
      const bigger = new Uint8Array(this.buf.length * 2);
      bigger.set(this.buf);
      this.buf = bigger;
    }
    this.buf[this.len++] = b;
  }

  finish(): Uint8Array {
    if (this.nbits > 0) this.byte(this.acc & 0xff);
    this.acc = 0;
    this.nbits = 0;
    return this.buf.slice(0, this.len);
  }
}

const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/** Index of the last base <= v. */
function bucket(bases: number[], v: number): number {
  let i = bases.length - 1;
  while (bases[i]! > v) i--;
  return i;
}

/** Fixed-Huffman literal/length symbol. */
function writeSymbol(w: BitWriter, sym: number): void {
  if (sym < 144) w.code(0x30 + sym, 8);
  else if (sym < 256) w.code(0x190 + sym - 144, 9);
  else if (sym < 280) w.code(sym - 256, 7);
  else w.code(0xc0 + sym - 280, 8);
}

function writeMatch(w: BitWriter, length: number, dist: number): void {
  const li = bucket(LEN_BASE, length);
  writeSymbol(w, 257 + li);
  if (LEN_EXTRA[li]) w.bits(length - LEN_BASE[li]!, LEN_EXTRA[li]!);
  const di = bucket(DIST_BASE, dist);
  w.code(di, 5);
  if (DIST_EXTRA[di]) w.bits(dist - DIST_BASE[di]!, DIST_EXTRA[di]!);
}

/**
 * zlib stream (RFC 1950/1951) of `data`: one fixed-Huffman block whose matches are at distance 1
 * or `stride` (pass the row length for an image; 0 = runs only).
 */
export function zlibCompress(data: Uint8Array, stride = 0): Uint8Array {
  const w = new BitWriter();
  w.bits(1, 1); // BFINAL
  w.bits(1, 2); // BTYPE = 01, fixed Huffman
  const n = data.length;
  const dists = stride > 1 && stride <= 32768 ? [1, stride] : [1];
  let i = 0;
  while (i < n) {
    let bestLen = 0;
    let bestDist = 0;
    for (const d of dists) {
      if (i < d) continue;
      let len = 0;
      while (len < 258 && i + len < n && data[i + len] === data[i + len - d]) len++;
      if (len > bestLen) {
        bestLen = len;
        bestDist = d;
      }
    }
    if (bestLen >= 3) {
      writeMatch(w, bestLen, bestDist);
      i += bestLen;
    } else {
      writeSymbol(w, data[i]!);
      i++;
    }
  }
  writeSymbol(w, 256); // end of block
  const body = w.finish();
  const out = new Uint8Array(2 + body.length + 4);
  out[0] = 0x78; // deflate, 32K window
  out[1] = 0x01; // no preset dictionary, fastest; (0x7801 % 31 === 0)
  out.set(body, 2);
  const a = adler32(data);
  out[out.length - 4] = a >>> 24;
  out[out.length - 3] = (a >>> 16) & 0xff;
  out[out.length - 2] = (a >>> 8) & 0xff;
  out[out.length - 1] = a & 0xff;
  return out;
}

/**
 * An 8-bit palette PNG. `pixels` holds one palette index per pixel (row-major, `width * height`);
 * `palette` is `[r, g, b, a]` per index (index 0 is usually the transparent background).
 */
export function encodePalettePNG(width: number, height: number, pixels: Uint8Array, palette: Array<[number, number, number, number]>): Uint8Array {
  if (pixels.length !== width * height) throw new Error(`pixels has ${pixels.length} entries, width * height = ${width * height}`);
  if (palette.length < 1 || palette.length > 256) throw new Error("a PNG palette has 1 to 256 entries");
  // Rows with filter type 2 (Up): each byte minus the byte above, so repeated rows become zeros.
  const stride = width + 1;
  const raw = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    const o = y * stride;
    raw[o] = 2;
    const row = y * width;
    if (y === 0) raw.set(pixels.subarray(0, width), o + 1);
    else for (let x = 0; x < width; x++) raw[o + 1 + x] = (pixels[row + x]! - pixels[row - width + x]!) & 0xff;
  }
  const idat = zlibCompress(raw, stride);

  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 3; // colour type: palette
  const plte = new Uint8Array(palette.length * 3);
  const trns = new Uint8Array(palette.length);
  palette.forEach(([r, g, b, a], i) => {
    plte.set([r, g, b], i * 3);
    trns[i] = a;
  });
  const chunks: Array<[string, Uint8Array]> = [
    ["IHDR", ihdr],
    ["PLTE", plte],
    ["tRNS", trns],
    ["IDAT", idat],
    ["IEND", new Uint8Array(0)],
  ];
  const total = 8 + chunks.reduce((s, [, d]) => s + 12 + d.length, 0);
  const out = new Uint8Array(total);
  out.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ov = new DataView(out.buffer);
  let p = 8;
  for (const [type, data] of chunks) {
    ov.setUint32(p, data.length);
    for (let k = 0; k < 4; k++) out[p + 4 + k] = type.charCodeAt(k);
    out.set(data, p + 8);
    ov.setUint32(p + 8 + data.length, crc32(out, p + 4, p + 8 + data.length));
    p += 12 + data.length;
  }
  return out;
}
