import { test } from "node:test";
import assert from "node:assert/strict";
import { inflateSync } from "node:zlib";

import { ClientResize, Mask, Result, classColour, selectTargetGrasp, toSVG } from "../src/index.js";
import { encodePalettePNG, zlibCompress } from "../src/png.js";
import { mapResult } from "../src/resize.js";

/** assert.ok with a message: without one, a failing assert.ok re-parses the TS source and can hang under tsx. */
const ok = (cond: unknown, msg = "expected a truthy value"): void => assert.ok(cond, msg);

/** Decode the palette PNG `toSVG` embeds (test-only, with node:zlib): indices, palette, alpha. */
function decodePNG(png: Uint8Array): { width: number; height: number; pixels: Uint8Array; plte: Uint8Array; trns: Uint8Array } {
  assert.deepEqual([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const dv = new DataView(png.buffer, png.byteOffset, png.byteLength);
  let p = 8;
  const chunks: Record<string, Uint8Array> = {};
  while (p < png.length) {
    const len = dv.getUint32(p);
    const type = String.fromCharCode(...png.subarray(p + 4, p + 8));
    chunks[type] = png.subarray(p + 8, p + 8 + len);
    p += 12 + len;
  }
  const ihdr = new DataView(chunks.IHDR!.buffer, chunks.IHDR!.byteOffset, 13);
  const width = ihdr.getUint32(0);
  const height = ihdr.getUint32(4);
  assert.equal(chunks.IHDR![8], 8); // bit depth
  assert.equal(chunks.IHDR![9], 3); // palette
  const raw = inflateSync(chunks.IDAT!); // also checks the zlib header and Adler-32
  const pixels = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    assert.equal(raw[y * (width + 1)], 2); // Up filter
    for (let x = 0; x < width; x++) {
      const above = y ? pixels[(y - 1) * width + x]! : 0;
      pixels[y * width + x] = (raw[y * (width + 1) + 1 + x]! + above) & 0xff;
    }
  }
  return { width, height, pixels, plte: chunks.PLTE!, trns: chunks.tRNS! };
}

/** The base64 PNG of the first <image> of an SVG, decoded. */
function maskPNG(svg: string) {
  const m = svg.match(/<image href="data:image\/png;base64,([^"]+)"/);
  ok(m, "no mask image in the SVG");
  return decodePNG(new Uint8Array(Buffer.from(m[1]!, "base64")));
}

/** Column-major RLE of a row-major 0/1 mask (the server's encoder). */
function rle(bits: number[], w: number, h: number): string {
  const runs: number[] = [];
  let cur = 0;
  let n = 0;
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const v = bits[y * w + x]!;
      if (v !== cur) {
        runs.push(n);
        n = 0;
        cur = v;
      }
      n++;
    }
  }
  runs.push(n);
  return runs.join(" ");
}

const count = (svg: string, tag: string) => (svg.match(new RegExp("<" + tag + "[ >]", "g")) ?? []).length;
const rgb = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;

test("zlibCompress is a valid zlib stream for runs, rows and noise", () => {
  const cases: Array<[Uint8Array, number]> = [
    [new Uint8Array(0), 0],
    [new Uint8Array([7]), 0],
    [new Uint8Array(100_000), 0],
    [Uint8Array.from({ length: 70_001 }, (_, i) => (i % 641 < 300 ? 0 : (i * 7) % 5)), 641],
    [Uint8Array.from({ length: 50_000 }, (_, i) => (i * 2654435761) >>> 24), 0],
    [Uint8Array.from({ length: 40_000 }, (_, i) => (Math.floor(i / 1000) % 3) * 100), 33_000], // stride > 32768: runs only
  ];
  for (const [data, stride] of cases) {
    const z = zlibCompress(data, stride);
    assert.deepEqual(new Uint8Array(inflateSync(z)), data);
  }
  // Runs compress to almost nothing.
  ok(zlibCompress(new Uint8Array(100_000)).length < 700); // 13 bits per 258 bytes
});

test("encodePalettePNG round-trips indices, palette and transparency", () => {
  const w = 37;
  const h = 23;
  const px = Uint8Array.from({ length: w * h }, (_, i) => ((i % w) > 10 && Math.floor(i / w) > 5 ? 1 + ((i % w) % 3) : 0));
  const png = encodePalettePNG(w, h, px, [[0, 0, 0, 0], [255, 0, 0, 115], [0, 255, 0, 115], [0, 0, 255, 115]]);
  const d = decodePNG(png);
  assert.deepEqual([d.width, d.height], [w, h]);
  assert.deepEqual(d.pixels, px);
  assert.deepEqual([...d.plte], [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
  assert.deepEqual([...d.trns], [0, 115, 115, 115]);
  assert.throws(() => encodePalettePNG(2, 2, new Uint8Array(3), [[0, 0, 0, 0]]), /width \* height/);
});

test("toSVG has a viewBox and the photo's size, so it scales to any CSS size", () => {
  const svg = toSVG(Result.fromJSON({ task: "depth", model: "midas", depth_map: [0, 1], depth_width: 2, depth_height: 1 }), 640, 426);
  assert.equal(svg, '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="426" viewBox="0 0 640 426"></svg>');
});

test("toSVG paints mask pixels, one colour per class, from the column-major RLE", () => {
  const [W, H] = [12, 8];
  const a = Array.from({ length: W * H }, (_, i) => (i % W >= 2 && i % W < 6 && Math.floor(i / W) >= 1 && Math.floor(i / W) < 7 ? 1 : 0));
  const b = Array.from({ length: W * H }, (_, i) => (i % W >= 5 && i % W < 10 && Math.floor(i / W) >= 3 ? 1 : 0));
  const res = Result.fromJSON({
    task: "open_vocab", model: "grounded-sam",
    detections: [{ bbox: [2, 1, 4, 6], class: "dog", conf: 0.6 }, { bbox: [5, 3, 5, 5], class: "cat", conf: 0.7 }],
    masks: [{ rle: rle(a, W, H), bbox: [2, 1, 4, 6], conf: 0.9 }, { rle: rle(b, W, H), bbox: [5, 3, 5, 5], conf: 0.8 }],
  });
  const svg = toSVG(res, W, H);
  const png = maskPNG(svg);
  assert.deepEqual([png.width, png.height], [W, H]);
  const dog = classColour("dog");
  const cat = classColour("cat");
  for (let k = 0; k < W * H; k++) {
    const want = b[k] ? cat : a[k] ? dog : null; // the later mask wins where they overlap
    const idx = png.pixels[k]!;
    if (!want) assert.equal(png.trns[idx], 0, `pixel ${k} should be transparent`);
    else {
      assert.deepEqual([...png.plte.subarray(idx * 3, idx * 3 + 3)], want, `pixel ${k}`);
      assert.equal(png.trns[idx], Math.round(0.45 * 255));
    }
  }
  // Boxes and labels in the class colours; masks with a detection's box get no second label.
  assert.equal(count(svg, "text"), 2);
  ok(svg.includes(`stroke="${rgb(dog)}"`) && svg.includes(`stroke="${rgb(cat)}"`));
  ok(svg.includes(">dog 60%<") && svg.includes(">cat 70%<"));
  // alpha, masks: false.
  assert.equal(maskPNG(toSVG(res, W, H, { alpha: 1 })).trns[1], 255);
  assert.equal(count(toSVG(res, W, H, { masks: false }), "image"), 0);
  // A class keeps its colour in another result; colorBy "index" colours by position instead.
  const other = Result.fromJSON({ task: "detection", detections: [{ bbox: [0, 0, 1, 1], class: "x", conf: 1 }, { bbox: [0, 0, 2, 2], class: "dog", conf: 1 }] });
  ok(toSVG(other, W, H).includes(`stroke="${rgb(dog)}"`));
  ok(toSVG(other, W, H, { colorBy: "index" }).includes(`stroke="rgb(255,165,0)"`)); // palette[1]
});

test("toSVG: unlabelled masks get a box and 'mask conf%'; maskBoxes: false hides it", () => {
  const res = Result.fromJSON({ task: "segmentation", masks: [{ rle: "3 3", bbox: [1, 0, 1, 3], conf: 0.98 }] });
  const svg = toSVG(res, 2, 3);
  assert.equal(count(svg, "image"), 1);
  ok(svg.includes(">mask 98%<"));
  const fills = maskPNG(svg).pixels;
  assert.deepEqual([...fills], [0, 1, 0, 1, 0, 1]); // column x = 1
  const bare = toSVG(res, 2, 3, { maskBoxes: false });
  assert.equal(count(bare, "rect"), 0);
  assert.equal(count(bare, "image"), 1);
  // RLE for another size: no fill, the box only.
  const wrong = toSVG(res, 4, 4);
  assert.equal(count(wrong, "image"), 0);
  assert.equal(count(wrong, "rect"), 2); // the box + the label band
});

test("toSVG fills a mask the client shrank at the ORIGINAL size", () => {
  const sent = Result.fromJSON({ task: "segmentation", masks: [{ rle: "3 3", bbox: [1, 0, 1, 3], conf: 0.9 }] }); // 2x3 sent
  const res = mapResult(sent, new ClientResize(4, 6, 2, 3, 90));
  const png = maskPNG(toSVG(res, 4, 6));
  assert.deepEqual([png.width, png.height], [4, 6]);
  assert.deepEqual([...png.pixels].map((v) => (v ? 1 : 0)), [...res.masks[0]!.toMask(4, 6)]);
});

test("toSVG draws grasps: the best 3 per object by default, the target in red on top", () => {
  const grasps = Array.from({ length: 5 }, (_, i) => ({ x: 20 + i, y: 20, theta: 0.3 * i, width: 30, quality: 0.5 + i / 10, class: "bowl", conf: 0.8 }));
  const res = Result.fromJSON({
    task: "grasp", model: "grasp-rfdetr",
    detections: [{ bbox: [0, 0, 100, 100], class: "bowl", conf: 0.8 }],
    grasps: [...grasps, { x: 300, y: 300, theta: 0, width: 10, quality: 0.2 }],
  });
  const svg = toSVG(res, 400, 400);
  // A glyph = closing line + two jaw plates + centre dot + label.
  assert.equal(count(svg, "line"), 3 * 4); // 3 bowl grasps + the class-agnostic one
  assert.equal(count(svg, "circle"), 4);
  ok(svg.includes(">bowl q0.90<") && !svg.includes(">bowl q0.60<"));
  assert.equal(count(toSVG(res, 400, 400, { maxGraspsPerObject: null }), "circle"), 6);
  assert.equal(count(toSVG(res, 400, 400, { maxGraspsPerObject: 1 }), "circle"), 2);
  // The jaw line runs between the two contacts.
  const g = res.grasps[4]!;
  const [x0, y0, x1, y1] = g.contactsFlat().map((v) => String(Math.round(v * 100) / 100));
  ok(svg.includes(`<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}"`));
  // Quality colours as Python computes them (int() of 50.99... for 0.9): green-ish, red-ish.
  ok(svg.includes('stroke="rgb(50,255,0)"'));
  ok(svg.includes('stroke="rgb(255,102,0)"'));

  const target = selectTargetGrasp(res.grasps, { targetPoint: [300, 300] })!;
  const t = toSVG(res, 400, 400, { targetGrasp: target });
  const lastGlyph = t.slice(t.lastIndexOf("<g>"));
  ok(lastGlyph.includes('stroke="rgb(255,0,0)" stroke-width="3"'));
  ok(lastGlyph.includes(">q0.20<"));
});

test("toSVG draws instance_detection and classification results; labels are escaped", () => {
  const inst = Result.fromJSON({ task: "instance_detection", detections: [{ bbox: [215, 230, 57, 89], class: "<dog&co>", conf: 0.93 }] });
  const svg = toSVG(inst, 640, 426);
  assert.equal(count(svg, "rect"), 2);
  ok(svg.includes(">&lt;dog&amp;co&gt; 93%<"));
  const cls = Result.fromJSON({ task: "classification", classifications: [{ class: "tusker", conf: 0.5 }, { class: "elephant", conf: 0.3 }] });
  const c = toSVG(cls, 640, 426);
  assert.equal(count(c, "text"), 2);
  ok(c.includes('fill="rgb(20,20,20)"') && c.includes(">tusker 50%<"));
});

test("toSVG highlights the target box, its own item or a standalone box", () => {
  const res = Result.fromJSON({
    task: "detection",
    detections: [{ bbox: [0, 0, 10, 10], class: "a", conf: 0.9 }, { bbox: [20, 0, 10, 10], class: "b", conf: 0.8 }],
  });
  const own = toSVG(res, 50, 50, { targetBox: res.detections[1]! });
  assert.equal(count(own, "rect"), 4);
  // 50 x 50: line width 1, the target twice as thick.
  ok(own.includes('<rect x="20" y="0" width="10" height="10" fill="none" stroke="rgb(255,0,0)" stroke-width="2"/>'), own);
  const byBox = toSVG(res, 50, 50, { targetBox: [20, 0, 10, 10] });
  assert.equal(byBox, own);
  const standalone = toSVG(res, 50, 50, { targetBox: [1, 2, 3, 4] });
  assert.equal(count(standalone, "rect"), 5);
  ok(standalone.endsWith('<rect x="1" y="2" width="3" height="4" fill="none" stroke="rgb(255,0,0)" stroke-width="2"/></svg>'), standalone);
  // A target mask keeps its red box even with maskBoxes: false.
  const seg = Result.fromJSON({ task: "segmentation", masks: [{ bbox: [1, 1, 2, 2], conf: 0.5 }, { bbox: [3, 3, 1, 1], conf: 0.4 }] });
  const m = toSVG(seg, 8, 8, { maskBoxes: false, targetBox: seg.masks[1] as Mask });
  assert.equal(count(m, "rect"), 2);
  ok(m.includes('stroke="rgb(255,0,0)" stroke-width="2"'), m);
});
