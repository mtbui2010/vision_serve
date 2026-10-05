import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { Client, ClientResize, Grasp, Mask, Result, ModelInfo, isLoopback, probeHeader, targetSize, type ImageCodec } from "../src/index.js";
import { mapResult, roiRegion } from "../src/resize.js";

const fixtures = JSON.parse(
  readFileSync(new URL("../../testdata/client_resize.json", import.meta.url), "utf8"),
) as {
  target_size: Array<{ width: number; height: number; max_side?: number; max_short_side?: number; region?: [number, number]; want: [number, number] }>;
  roi_region: Array<{ roi: number[]; width: number; height: number; want: [number, number] | null }>;
  is_loopback: Array<{ host: string; want: boolean }>;
};

test("targetSize matches the shared cases (Python runs the same file)", () => {
  for (const c of fixtures.target_size) {
    const got = targetSize(c.width, c.height, { maxSide: c.max_side, maxShortSide: c.max_short_side, region: c.region ?? null });
    assert.deepEqual(got, c.want, JSON.stringify(c));
  }
});

test("roiRegion matches the shared cases", () => {
  for (const c of fixtures.roi_region) {
    assert.deepEqual(roiRegion(c.roi, c.width, c.height), c.want, JSON.stringify(c));
  }
});

// ------------------------------------------------------------------ //
// Header probe
// ------------------------------------------------------------------ //

/** A JPEG header: SOI, optional APP1 EXIF with an Orientation tag, SOF0 with the size, SOS. */
function jpegHeader(w: number, h: number, orientation?: number, littleEndian = false): Uint8Array {
  const out: number[] = [0xff, 0xd8];
  if (orientation) {
    const u16 = (v: number) => (littleEndian ? [v & 0xff, v >> 8] : [v >> 8, v & 0xff]);
    const u32 = (v: number) => (littleEndian ? [v & 0xff, (v >> 8) & 0xff, 0, 0] : [0, 0, (v >> 8) & 0xff, v & 0xff]);
    const tiff = [
      ...(littleEndian ? [0x49, 0x49] : [0x4d, 0x4d]), ...u16(42), ...u32(8),
      ...u16(1), // one IFD entry
      ...u16(0x0112), ...u16(3), ...u32(1), ...u16(orientation), 0, 0,
      ...u32(0),
    ];
    const payload = [0x45, 0x78, 0x69, 0x66, 0, 0, ...tiff]; // "Exif\0\0"
    const len = payload.length + 2;
    out.push(0xff, 0xe1, len >> 8, len & 0xff, ...payload);
  }
  out.push(0xff, 0xc0, 0, 17, 8, h >> 8, h & 0xff, w >> 8, w & 0xff, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1);
  out.push(0xff, 0xda, 0, 2);
  return new Uint8Array(out);
}

function pngHeader(w: number, h: number): Uint8Array {
  const b = new Uint8Array(33);
  b.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52]);
  const v = new DataView(b.buffer);
  v.setUint32(16, w);
  v.setUint32(20, h);
  return b;
}

test("probeHeader reads JPEG / PNG sizes and applies the JPEG's EXIF rotation", () => {
  assert.deepEqual(probeHeader(jpegHeader(3000, 2000)), { width: 3000, height: 2000, isJpeg: true, orientation: 1 });
  assert.deepEqual(probeHeader(jpegHeader(3000, 2000, 6)), { width: 2000, height: 3000, isJpeg: true, orientation: 6 });
  assert.deepEqual(probeHeader(jpegHeader(3000, 2000, 8, true)), { width: 2000, height: 3000, isJpeg: true, orientation: 8 });
  assert.deepEqual(probeHeader(jpegHeader(3000, 2000, 3)), { width: 3000, height: 2000, isJpeg: true, orientation: 3 });
  assert.deepEqual(probeHeader(pngHeader(640, 480)), { width: 640, height: 480, isJpeg: false });
  assert.equal(probeHeader(new Uint8Array([1, 2, 3])), null);
  // a real JPEG from the repository
  const real = new Uint8Array(readFileSync(new URL("../../../test/testdata/sample.jpg", import.meta.url)));
  const p = probeHeader(real);
  assert.ok(p && p.isJpeg && p.width > 0 && p.height > 0);
});

// ------------------------------------------------------------------ //
// Through the client, with a fake codec and a fake server
// ------------------------------------------------------------------ //

/** A codec that reads headers and "encodes" by returning a marker naming the size it was asked for. */
function fakeCodec() {
  const calls: Array<{ width: number; height: number; jpeg: boolean; quality: number }> = [];
  const codec: ImageCodec = {
    probe: async (b) => probeHeader(b),
    transcode: async (_b, width, height, opts) => {
      calls.push({ width, height, ...opts });
      return new TextEncoder().encode(`sent ${width}x${height} ${opts.jpeg ? "jpeg" : "png"}`);
    },
  };
  return { codec, calls };
}

const MODELS = [
  { name: "rf-detr", task: "detection", license: "Apache-2.0", state: "loaded", max_useful_side: null, max_useful_short_side: 1120 },
  { name: "rt-detr", task: "detection", license: "Apache-2.0", state: "loaded", max_useful_side: 1280, max_useful_short_side: null },
  { name: "mobile-sam", task: "segmentation", license: "Apache-2.0", state: "loaded", max_useful_side: null, max_useful_short_side: null },
  { name: "old", task: "detection", license: "MIT", state: "loaded" },
];

function fakeServer(reply: Record<string, unknown> = { task: "detection", model: "m", duration_ms: 1 }) {
  const state = { modelsCalls: 0, form: null as FormData | null, models: MODELS as unknown };
  const original = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL, init: RequestInit = {}) => {
    if (String(url).endsWith("/api/models")) {
      state.modelsCalls++;
      return new Response(JSON.stringify(state.models));
    }
    state.form = init.body as FormData;
    return new Response(JSON.stringify(reply));
  }) as typeof fetch;
  return {
    state,
    async image(): Promise<string> {
      const f = state.form!.get("image") as Blob;
      return new TextDecoder().decode(await f.arrayBuffer());
    },
    async imageBytes(): Promise<Uint8Array> {
      return new Uint8Array(await (state.form!.get("image") as Blob).arrayBuffer());
    },
    field: (k: string) => state.form!.get(k),
    restore: () => {
      globalThis.fetch = original;
    },
  };
}

test("predict() shrinks to the hint, scales prompts and maps results back", async () => {
  const srv = fakeServer({ task: "detection", model: "rf-detr", duration_ms: 1, detections: [{ bbox: [56, 56, 112, 56], class: "cat", conf: 0.9 }] });
  const { codec, calls } = fakeCodec();
  try {
    const c = new Client("http://x", { codec });
    const res = await c.predict("rf-detr", jpegHeader(3000, 2000), {
      box: [300, 300, 600, 300],
      point: [[100, 200, 1]],
      roi: [0, 0, 3000, 2000],
    });
    assert.deepEqual(calls, [{ width: 1680, height: 1120, jpeg: true, quality: 90 }]);
    assert.equal(await srv.image(), "sent 1680x1120 jpeg");
    assert.equal(srv.field("box"), "168,168,336,168");
    assert.equal(srv.field("point"), "56,112,1");
    assert.equal(srv.field("roi"), "0,0,1680,1120");
    assert.equal(srv.field("resize"), null); // client-side options are never sent
    assert.deepEqual(res.detections[0]!.bbox, [100, 100, 200, 100]);
    assert.deepEqual(res.clientResize, new ClientResize(3000, 2000, 1680, 1120, 90));
    await c.predict("rt-detr", jpegHeader(3000, 2000));
    assert.equal(srv.state.modelsCalls, 1); // the listing is cached
    assert.deepEqual(calls[1], { width: 1280, height: 853, jpeg: true, quality: 90 });
  } finally {
    srv.restore();
  }
});

test("a JPEG within the hint, or a model without one, is sent untouched", async () => {
  const srv = fakeServer();
  const { codec, calls } = fakeCodec();
  try {
    const c = new Client("http://x", { codec });
    const small = jpegHeader(640, 480);
    let res = await c.predict("rf-detr", small);
    assert.deepEqual(await srv.imageBytes(), small);
    assert.equal(res.clientResize, null);
    const big = jpegHeader(3000, 2000);
    for (const model of ["mobile-sam", "old", "unknown"]) {
      res = await c.predict(model, big);
      assert.deepEqual(await srv.imageBytes(), big, model);
      assert.equal(res.clientResize, null);
    }
    assert.equal(calls.length, 0);
  } finally {
    srv.restore();
  }
});

test("only a shrunk photo is re-encoded; jpeg picks its format", async () => {
  const srv = fakeServer();
  const { codec, calls } = fakeCodec();
  try {
    const png = pngHeader(300, 200);
    const res = await new Client("http://x", { codec, jpegQuality: 80 }).predict("rf-detr", png);
    assert.deepEqual(await srv.imageBytes(), png); // not shrunk: sent as given, even with jpeg on
    assert.equal(res.clientResize, null);
    assert.equal(calls.length, 0);
    await new Client("http://x", { codec, jpegQuality: 80 }).predict("rf-detr", pngHeader(3000, 2000));
    assert.equal(await srv.image(), "sent 1680x1120 jpeg");
    await new Client("http://x", { codec }).predict("rf-detr", pngHeader(3000, 2000), { jpeg: false });
    assert.equal(await srv.image(), "sent 1680x1120 png");
  } finally {
    srv.restore();
  }
});

test("isLoopback matches the shared cases (Python runs the same file)", () => {
  for (const c of fixtures.is_loopback) assert.equal(isLoopback(c.host), c.want, c.host);
});

test("the loopback rule keeps a mild shrink whole and says so", async () => {
  const srv = fakeServer();
  const { codec, calls } = fakeCodec();
  try {
    const c = new Client("http://127.0.0.1:11435", { codec });
    const big = jpegHeader(3000, 2000); // rf-detr: 1120 / 2000 = 0.56 > 0.5
    let res = await c.predict("rf-detr", big);
    assert.deepEqual(await srv.imageBytes(), big);
    assert.equal(res.clientResize?.resized, false);
    assert.equal(res.clientResize?.reason, "loopback: scale 0.56 > 0.5, sent as is");
    res = await c.predict("rf-detr", jpegHeader(4000, 2400)); // 1120 / 2400 = 0.47: shrunk
    assert.equal(await srv.image(), "sent 1867x1120 jpeg");
    assert.equal(res.clientResize?.reason, "hint");
    res = await c.predict("rf-detr", big, { resize: 1500 }); // explicit: no loopback rule
    assert.equal(await srv.image(), "sent 1500x1000 jpeg");
    assert.equal(res.clientResize?.reason, "resize=1500");
    assert.equal(calls.length, 2);
  } finally {
    srv.restore();
  }
});

test("resize off / int, full-resolution options and a missing codec", async () => {
  const srv = fakeServer();
  const { codec, calls } = fakeCodec();
  const big = jpegHeader(3000, 2000);
  try {
    await new Client("http://x", { codec, resize: "off" }).predict("rf-detr", big);
    assert.deepEqual(await srv.imageBytes(), big);
    await new Client("http://x", { codec, resize: 1000 }).predict("mobile-sam", big);
    assert.equal(await srv.image(), "sent 1000x667 jpeg");
    assert.equal(srv.state.modelsCalls, 0); // neither needs the listing
    const c = new Client("http://x", { codec });
    await c.predict("rf-detr", big, { resize: 500 });
    assert.equal(await srv.image(), "sent 500x333 jpeg");
    for (const opts of [{ dilate: 3 }, { gripperMax: 80 }, { templateName: "mug" }, { depth: new Float32Array(6e6) }]) {
      const res = await c.predict("rf-detr", big, opts);
      assert.deepEqual(await srv.imageBytes(), big, JSON.stringify(Object.keys(opts)));
      assert.equal(res.clientResize, null);
    }
    await new Client("http://x", { codec: null }).predict("rf-detr", big);
    assert.deepEqual(await srv.imageBytes(), big);
    assert.equal(calls.length, 2);
    await assert.rejects(() => c.predict("rf-detr", big, { resize: 0 }), /resize must be/);
    assert.throws(() => new Client("http://x", { jpegQuality: 101 }), /jpegQuality/);
  } finally {
    srv.restore();
  }
});

test("a failed listing means full resolution", async () => {
  const srv = fakeServer();
  srv.state.models = "not a list";
  const { codec } = fakeCodec();
  try {
    const big = jpegHeader(3000, 2000);
    const res = await new Client("http://x", { codec }).predict("rf-detr", big);
    assert.deepEqual(await srv.imageBytes(), big);
    assert.equal(res.clientResize, null);
  } finally {
    srv.restore();
  }
});

test("listModels() reads the hint", async () => {
  const srv = fakeServer();
  try {
    const infos = await new Client("http://x").listModels();
    assert.deepEqual(infos[0], new ModelInfo("rf-detr", "detection", "Apache-2.0", "loaded", null, 1120));
    assert.equal(infos[3]!.maxUsefulSide, null);
    assert.equal(infos[3]!.maxUsefulShortSide, null);
  } finally {
    srv.restore();
  }
});

test("mapResult maps masks, grasps and keeps the mask at the sent size", () => {
  // sent 4x2 with the right half set; original 8x4
  const sent = [0, 0, 1, 1, 0, 0, 1, 1]; // row-major 4x2
  const runs: number[] = [];
  let cur = 0;
  let n = 0;
  for (let x = 0; x < 4; x++) {
    for (let y = 0; y < 2; y++) {
      const v = sent[y * 4 + x]!;
      if (v === cur) n++;
      else {
        runs.push(n);
        cur = v;
        n = 1;
      }
    }
  }
  runs.push(n);
  const cr = new ClientResize(8, 4, 4, 2, 90);
  const theta = 0.7;
  const r = mapResult(
    new Result("segmentation", "m", [], [new Mask(runs.join(" "), [2, 0, 2, 2], 1)], [], [], 0, 0, [], 1, [
      new Grasp(2, 1, theta, 1, 0.5),
    ]),
    cr,
  );
  const m = r.masks[0]!;
  assert.deepEqual(m.bbox, [4, 0, 4, 4]);
  const full = m.toMask(8, 4);
  for (let y = 0; y < 4; y++) for (let x = 0; x < 8; x++) assert.equal(full[y * 8 + x], x >= 4 ? 1 : 0);
  assert.deepEqual(Array.from(m.toMask(4, 2)), sent);
  const g = r.grasps[0]!;
  assert.deepEqual([g.x, g.y], [4, 2]);
  const vx = Math.cos(theta) / 0.5;
  const vy = Math.sin(theta) / 0.5;
  assert.ok(Math.abs(g.width - Math.hypot(vx, vy)) < 1e-12 && Math.abs(g.theta - Math.atan2(vy, vx)) < 1e-12);
  assert.equal(r.clientResize, cr);
  assert.equal(r.filterByConf(0, 1).clientResize, cr); // carried through the helpers
});
