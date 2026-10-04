import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  Client,
  VisionServeError,
  Result,
  Mask,
  Detection,
  filterBySize,
  getDepthAtDetection,
  normalizePrompt,
} from "../src/index.js";

/** Install a fake global fetch; returns a handle to inspect the last request. */
function mockFetch(responder: (url: string, init: RequestInit) => Response) {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL, init: RequestInit = {}) => {
    calls.push({ url: String(url), init });
    return responder(String(url), init);
  }) as typeof fetch;
  return {
    calls,
    restore() {
      globalThis.fetch = original;
    },
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

test("health() hits GET /api/health", async () => {
  const m = mockFetch(() => json({ status: "ok" }));
  try {
    const client = new Client("http://localhost:11435");
    const res = await client.health();
    assert.deepEqual(res, { status: "ok" });
    assert.equal(m.calls[0]!.url, "http://localhost:11435/api/health");
    assert.equal(m.calls[0]!.init.method, "GET");
  } finally {
    m.restore();
  }
});

test("host trailing slash is stripped", async () => {
  const m = mockFetch(() => json({ status: "ok" }));
  try {
    await new Client("http://localhost:11435///").health();
    assert.equal(m.calls[0]!.url, "http://localhost:11435/api/health");
  } finally {
    m.restore();
  }
});

test("listModels() parses ModelInfo and ps() filters loaded", async () => {
  const m = mockFetch(() =>
    json([
      { name: "rf-detr", task: "detection", license: "Apache-2.0", state: "loaded" },
      { name: "mobile-sam", task: "segmentation", license: "Apache-2.0", state: "available" },
    ]),
  );
  try {
    const client = new Client();
    const models = await client.listModels();
    assert.equal(models.length, 2);
    assert.equal(models[0]!.name, "rf-detr");
    assert.equal(models[0]!.isLoaded, true);
    const loaded = await client.ps();
    assert.deepEqual(loaded.map((x) => x.name), ["rf-detr"]);
  } finally {
    m.restore();
  }
});

test("predict() builds multipart with model + formatted box", async () => {
  const m = mockFetch(() => json({ task: "segmentation", model: "mobile-sam", masks: [] }));
  try {
    const client = new Client();
    await client.predict("mobile-sam", new Uint8Array([1, 2, 3]), { box: [34, 58, 120, 240] });
    const body = m.calls[0]!.init.body as FormData;
    assert.ok(body instanceof FormData);
    assert.equal(body.get("model"), "mobile-sam");
    assert.equal(body.get("box"), "34,58,120,240");
    assert.equal(m.calls[0]!.url.endsWith("/api/predict"), true);
    // image part is a Blob/File
    assert.ok(body.get("image"));
    // No Content-Type header — fetch must set the multipart boundary itself.
    const headers = (m.calls[0]!.init.headers ?? {}) as Record<string, string>;
    assert.equal(headers["Content-Type"], undefined);
  } finally {
    m.restore();
  }
});

test("predict() serializes multiple boxes, points, prompt", async () => {
  const m = mockFetch(() => json({ task: "open_vocab", model: "grounded-sam" }));
  try {
    const client = new Client();
    await client.predict("grounded-sam", new Uint8Array([0]), {
      prompt: "cat. remote.",
      box: [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      point: [[95, 180, 1], [10, 20]],
    });
    const body = m.calls[0]!.init.body as FormData;
    assert.equal(body.get("prompt"), "cat. remote.");
    assert.equal(body.get("box"), "1,2,3,4;5,6,7,8");
    assert.equal(body.get("point"), "95,180,1;10,20");
  } finally {
    m.restore();
  }
});

test("predict() drops empty prompt and formats float without trailing .0", async () => {
  const m = mockFetch(() => json({ task: "segmentation", model: "mobile-sam" }));
  try {
    const client = new Client();
    await client.predict("mobile-sam", new Uint8Array([0]), { prompt: "   ", box: [1.5, 2.0, 3, 4] });
    const body = m.calls[0]!.init.body as FormData;
    assert.equal(body.get("prompt"), null);
    assert.equal(body.get("box"), "1.5,2,3,4");
  } finally {
    m.restore();
  }
});

test("predict() validates box / point arity", async () => {
  const m = mockFetch(() => json({}));
  try {
    const client = new Client();
    await assert.rejects(() => client.predict("m", new Uint8Array([0]), { box: [1, 2, 3] }), /4 values/);
    await assert.rejects(
      () => client.predict("m", new Uint8Array([0]), { point: [1, 2, 3, 4] }),
      /2 or 3 values/,
    );
  } finally {
    m.restore();
  }
});

test("Result.fromJSON maps `class` -> cls and duration_ms -> durationMs", () => {
  const res = Result.fromJSON({
    task: "detection",
    model: "rf-detr",
    detections: [{ bbox: [1, 2, 3, 4], class: "cat", conf: 0.91 }],
    duration_ms: 18.4,
  });
  assert.equal(res.task, "detection");
  assert.equal(res.durationMs, 18.4);
  assert.equal(res.detections.length, 1);
  const d = res.detections[0]!;
  assert.ok(d instanceof Detection);
  assert.equal(d.cls, "cat");
  assert.deepEqual(d.bbox, [1, 2, 3, 4]);
});

test("Mask.toMask decodes column-major RLE to row-major", () => {
  // W=2, H=3, total=6. Foreground = entire column x=1 (run order k=3,4,5).
  const mask = new Mask("3 3", [1, 0, 1, 3], 0.98);
  const flat = mask.toMask(2, 3);
  assert.deepEqual(Array.from(flat), [0, 1, 0, 1, 0, 1]);
  const grid = mask.toMask2D(2, 3);
  assert.deepEqual(grid, [
    [false, true],
    [false, true],
    [false, true],
  ]);
});

test("Mask.toMask rejects a run-count sum mismatch", () => {
  const mask = new Mask("3 2", [0, 0, 0, 0], 1.0);
  assert.throws(() => mask.toMask(2, 3), /sum to 5 but width\*height = 6/);
});

test("non-2xx surfaces the server error field", async () => {
  const m = mockFetch(() => json({ error: "model not found" }, 404));
  try {
    const client = new Client();
    await assert.rejects(
      () => client.health(),
      (e: unknown) => e instanceof VisionServeError && e.status === 404 && /model not found/.test(String(e)),
    );
  } finally {
    m.restore();
  }
});

test("filterBySize keeps grasps and device (only detections and masks are filtered)", () => {
  const res = Result.fromJSON({
    task: "grasp",
    model: "grasp-rfdetr",
    device: "gpu:0",
    detections: [
      { bbox: [0, 0, 10, 10], class: "cup", conf: 0.9 },
      { bbox: [0, 0, 1, 1], class: "crumb", conf: 0.5 },
    ],
    grasps: [{ x: 5, y: 5, theta: 0.5, width: 8, quality: 0.9, class: "cup", conf: 0.9 }],
    duration_ms: 3,
  });
  const out = filterBySize(res, { minSize: 50 });
  assert.deepEqual(out.detections.map((d) => d.cls), ["cup"]);
  assert.equal(out.grasps.length, 1);
  assert.equal(out.grasps[0]!.cls, "cup");
  assert.equal(out.device, "gpu:0");
});

// --------------------------------------------------------------------------------------------
// Parity with the Python SDK: options, prompt normalisation, errors, base64 arrays, depth
// --------------------------------------------------------------------------------------------

/** The text fields of the last request, as a plain object (file parts left out). */
function fields(m: { calls: Array<{ init: RequestInit }> }): Record<string, string> {
  const body = m.calls.at(-1)!.init.body as FormData;
  const out: Record<string, string> = {};
  for (const [k, v] of body.entries()) if (typeof v === "string") out[k] = v;
  return out;
}

test("predict() sends every option under the server's snake_case field name", async () => {
  const m = mockFetch(() => json({ task: "detection", model: "m" }));
  try {
    await new Client().predict("background", new Uint8Array([0]), {
      boxThreshold: 0.35,
      textThreshold: 0.2,
      bgMaxArea: 40,
      fgMinArea: 0.5,
      gridSize: 12,
      method: "automask",
      roi: [0.1, 0.2, 0.5, 0.5],
      dilate: -3,
      minSize: 0.1,
      maxSize: 90,
      gripperMin: 10,
      gripperMax: 150.5,
      claimThreshold: 0.05,
      cropTemp: 0.02,
      templateName: "dog",
    });
    assert.deepEqual(fields(m), {
      model: "background",
      roi: "0.1,0.2,0.5,0.5",
      box_threshold: "0.35",
      text_threshold: "0.2",
      bg_max_area: "40",
      fg_min_area: "0.5",
      min_size: "0.1",
      max_size: "90",
      gripper_min: "10",
      gripper_max: "150.5",
      claim_threshold: "0.05",
      crop_temp: "0.02",
      grid_size: "12",
      dilate: "-3",
      method: "automask",
      template_name: "dog",
    });
  } finally {
    m.restore();
  }
});

test("predict() leaves out undefined / null options (server defaults)", async () => {
  const m = mockFetch(() => json({ task: "detection", model: "rf-detr" }));
  try {
    await new Client().predict("rf-detr", new Uint8Array([0]), { minSize: undefined, method: null, roi: null });
    assert.deepEqual(fields(m), { model: "rf-detr" });
  } finally {
    m.restore();
  }
});

test("predict() refuses malformed values before sending anything", async () => {
  const m = mockFetch(() => json({}));
  try {
    const c = new Client();
    const img = new Uint8Array([0]);
    await assert.rejects(() => c.predict("m", img, { gridSize: 2.5 }), /gridSize must be an integer/);
    await assert.rejects(() => c.predict("m", img, { dilate: 1.5 }), /dilate must be an integer/);
    await assert.rejects(() => c.predict("m", img, { boxThreshold: Number.NaN }), /finite number/);
    await assert.rejects(() => c.predict("m", img, { minSize: "1" as unknown as number }), /finite number/);
    await assert.rejects(() => c.predict("m", img, { roi: [1, 2, 3] }), /roi must be one box/);
    // snake_case keys are a TypeError naming the camelCase option, not silently ignored.
    await assert.rejects(
      () => c.predict("m", img, { min_size: 1 } as unknown as Record<string, number>),
      (e: unknown) => e instanceof TypeError && /unknown option "min_size".*"minSize"/.test(e.message),
    );
    assert.equal(m.calls.length, 0);
  } finally {
    m.restore();
  }
});

test("normalizePrompt() follows the shared fixtures (same as the Python SDK)", () => {
  const url = new URL("../../testdata/normalize_prompt.json", import.meta.url);
  const { cases } = JSON.parse(readFileSync(url, "utf8")) as {
    cases: Array<{ model: string; prompt: string | null; want: string | null }>;
  };
  assert.ok(cases.length >= 20);
  for (const c of cases) {
    assert.equal(normalizePrompt(c.model, c.prompt), c.want, `${c.model} ${JSON.stringify(c.prompt)}`);
  }
});

test("predict() defaults GroundingDINO to 'object.' and normalises separators", async () => {
  const m = mockFetch(() => json({ task: "open_vocab", model: "grounding-dino" }));
  try {
    const c = new Client();
    await c.predict("grounding-dino", new Uint8Array([0]));
    assert.equal(fields(m).prompt, "object.");
    await c.predict("grounding-dino", new Uint8Array([0]), { prompt: "cat, remote" });
    assert.equal(fields(m).prompt, "cat. remote");
    await c.predict("rf-detr", new Uint8Array([0]));
    assert.equal(fields(m).prompt, undefined);
  } finally {
    m.restore();
  }
});

test("a 503 carries retryAfter (seconds) from the Retry-After header", async () => {
  const m = mockFetch(
    () => new Response(JSON.stringify({ error: "queue full" }), { status: 503, headers: { "Retry-After": "1" } }),
  );
  try {
    await assert.rejects(
      () => new Client().predict("rf-detr", new Uint8Array([0])),
      (e: unknown) =>
        e instanceof VisionServeError && e.status === 503 && e.retryAfter === 1 && /queue full/.test(e.message),
    );
  } finally {
    m.restore();
  }
});

test("retryAfter is undefined without a usable Retry-After", async () => {
  for (const headers of [{}, { "Retry-After": "soon" }, { "Retry-After": "-1" }] as Record<string, string>[]) {
    const m = mockFetch(() => new Response("{}", { status: 503, headers }));
    try {
      await assert.rejects(
        () => new Client().health(),
        (e: unknown) => e instanceof VisionServeError && e.status === 503 && e.retryAfter === undefined,
      );
    } finally {
      m.restore();
    }
  }
});

test("a timeout says 'timed out after N ms', not 'failed to reach'", async () => {
  // A server that never answers: fetch only settles when the abort signal fires.
  const original = globalThis.fetch;
  globalThis.fetch = ((_url: string | URL, init: RequestInit = {}) =>
    new Promise((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("This operation was aborted", "AbortError")));
    })) as typeof fetch;
  try {
    await assert.rejects(
      () => new Client("http://127.0.0.1:1", { timeoutMs: 30 }).health(),
      (e: unknown) =>
        e instanceof VisionServeError &&
        e.status === undefined &&
        /timed out after 30 ms/.test(e.message) &&
        !/failed to reach/.test(e.message),
    );
  } finally {
    globalThis.fetch = original;
  }
});

test("a body that stalls past the timeout is a timeout too", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = (async (_url: string | URL, init: RequestInit = {}) => {
    const stream = new ReadableStream({
      start(controller) {
        init.signal?.addEventListener("abort", () => controller.error(new DOMException("aborted", "AbortError")));
      },
    });
    return new Response(stream, { status: 200 });
  }) as typeof fetch;
  try {
    await assert.rejects(
      () => new Client("http://127.0.0.1:1", { timeoutMs: 30 }).health(),
      (e: unknown) => e instanceof VisionServeError && /timed out after 30 ms/.test(e.message),
    );
  } finally {
    globalThis.fetch = original;
  }
});

test("an unreachable server still says 'failed to reach'", async () => {
  const m = mockFetch(() => {
    throw new TypeError("fetch failed", { cause: new Error("connect ECONNREFUSED 127.0.0.1:1") });
  });
  try {
    await assert.rejects(
      () => new Client("http://127.0.0.1:1").health(),
      (e: unknown) =>
        e instanceof VisionServeError && /failed to reach VisionServe/.test(e.message) && /ECONNREFUSED/.test(e.message),
    );
  } finally {
    m.restore();
  }
});

test("Result keeps the server's hint through the helpers", () => {
  const res = Result.fromJSON({
    task: "detection",
    model: "rf-detr",
    hint: "install TensorRT for ~1.5x",
    detections: [{ bbox: [0, 0, 10, 10], class: "cup", conf: 0.9 }],
  });
  assert.equal(res.hint, "install TensorRT for ~1.5x");
  assert.equal(res.filterByConf(0.5).hint, res.hint);
  assert.equal(res.topK(1).hint, res.hint);
  assert.equal(res.nms(0.5).sortByConf().hint, res.hint);
  assert.equal(filterBySize(res, { minSize: 1 }).hint, res.hint);
  assert.equal(res.groupByClass()["cup"]!.hint, res.hint);
  assert.equal(Result.fromJSON({ task: "detection" }).hint, "");
});

test("base64Arrays sends encoding=base64 and decodes float32 depth and embeddings", async () => {
  const m = mockFetch(() =>
    json({
      task: "depth",
      model: "midas",
      depth_map_base64: "zczMPQAAAD8AAIA/AAAQwA==", // <f4: 0.1, 0.5, 1.0, -2.25
      depth_width: 2,
      depth_height: 2,
      embeddings_base64: "AACAPwAAAEAAAEBAAACAQAAAoEAAAMBA", // <f4: 1..6
      embeddings_shape: [2, 3],
    }),
  );
  try {
    const res = await new Client("http://x", { base64Arrays: true }).predict("midas", new Uint8Array([0]));
    assert.equal(fields(m).encoding, "base64");
    assert.deepEqual(res.depthMap, [Math.fround(0.1), 0.5, 1, -2.25]);
    assert.ok(Array.isArray(res.depthMap));
    assert.deepEqual(res.embeddings, [
      [1, 2, 3],
      [4, 5, 6],
    ]);
    await new Client("http://x").predict("midas", new Uint8Array([0]));
    assert.equal(fields(m).encoding, undefined); // opt-in
  } finally {
    m.restore();
  }
});

test("depth: typed arrays are sent little-endian with dtype and size", async () => {
  const m = mockFetch(() => json({ task: "segmentation", model: "background" }));
  try {
    const c = new Client();
    await c.predict("background", new Uint8Array([0]), {
      method: "depth",
      depth: new Uint16Array([1, 0x0102, 65535, 0]),
      depthWidth: 2,
      depthHeight: 2,
    });
    let body = m.calls.at(-1)!.init.body as FormData;
    assert.deepEqual(
      { dtype: body.get("depth_dtype"), w: body.get("depth_width"), h: body.get("depth_height") },
      { dtype: "uint16", w: "2", h: "2" },
    );
    const part = body.get("depth") as Blob;
    assert.deepEqual(Array.from(new Uint8Array(await part.arrayBuffer())), [1, 0, 2, 1, 255, 255, 0, 0]);

    await c.predict("background", new Uint8Array([0]), { depth: new Float32Array([1.5]) });
    body = m.calls.at(-1)!.init.body as FormData;
    assert.equal(body.get("depth_dtype"), "float32");
    assert.equal(body.get("depth_width"), null); // server default: the image's size
    const f32 = new DataView(await (body.get("depth") as Blob).arrayBuffer()).getFloat32(0, true);
    assert.equal(f32, 1.5);

    // raw bytes need an explicit dtype
    await c.predict("background", new Uint8Array([0]), { depth: new Uint8Array([1, 0]), depthDtype: "uint16" });
    assert.equal((m.calls.at(-1)!.init.body as FormData).get("depth_dtype"), "uint16");
  } finally {
    m.restore();
  }
});

test("depth: inconsistent dtype / size is refused before sending", async () => {
  const m = mockFetch(() => json({}));
  try {
    const c = new Client();
    const img = new Uint8Array([0]);
    await assert.rejects(() => c.predict("background", img, { depth: new Uint8Array([1, 2]) }), /need depthDtype/);
    await assert.rejects(
      () => c.predict("background", img, { depth: new Uint16Array([1]), depthDtype: "float32" }),
      /contradicts/,
    );
    await assert.rejects(
      () => c.predict("background", img, { depth: new Uint16Array(5), depthWidth: 2, depthHeight: 2 }),
      /5 values but depthWidth\*depthHeight = 4/,
    );
    await assert.rejects(() => c.predict("background", img, { depth: new Uint16Array(4), depthWidth: 2 }), /go together/);
    await assert.rejects(() => c.predict("background", img, { depthWidth: 2, depthHeight: 2 }), /need a depth map/);
    assert.equal(m.calls.length, 0);
  } finally {
    m.restore();
  }
});

/** A 4x2 depth Result (model resolution) whose value at (x, y) is 10*y + x. */
function depthResult(): Result {
  const w = 4;
  const h = 2;
  const map: number[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) map.push(10 * y + x);
  return Result.fromJSON({ task: "depth", model: "midas", depth_map: map, depth_width: w, depth_height: h });
}

test("getDepthAtDetection scales image-pixel boxes onto a model-sized depth map", () => {
  // The image is 400x200, the map 4x2: one map pixel covers 100x100 image pixels.
  const det = Result.fromJSON({
    task: "detection",
    model: "rf-detr",
    detections: [
      { bbox: [300, 100, 100, 100], class: "a", conf: 1 }, // bottom-right map pixel (3, 1) = 13
      { bbox: [0, 0, 200, 100], class: "b", conf: 1 }, // map pixels (0,0), (1,0) = 0, 1
      { bbox: [500, 0, 50, 50], class: "c", conf: 1 }, // outside the image
    ],
  });
  const opts = { imageWidth: 400, imageHeight: 200 };
  assert.deepEqual(getDepthAtDetection(depthResult(), det, opts), [13, 0.5, null]);
  assert.deepEqual(getDepthAtDetection(depthResult(), det, { ...opts, mode: "min" }), [13, 0, null]); // 0 is a value
  assert.deepEqual(getDepthAtDetection(depthResult(), det, { ...opts, mode: "max" }), [13, 1, null]);
  // A map at the image's own size is read 1:1.
  const small = Result.fromJSON({ task: "detection", detections: [{ bbox: [1, 1, 2, 1], class: "x", conf: 1 }] });
  assert.deepEqual(getDepthAtDetection(depthResult(), small, { imageWidth: 4, imageHeight: 2, mode: "mean" }), [11.5]);
});

test("getDepthAtDetection requires the image size and a depth map", () => {
  const det = Result.fromJSON({ task: "detection", detections: [{ bbox: [0, 0, 1, 1], class: "x", conf: 1 }] });
  assert.throws(
    () => getDepthAtDetection(depthResult(), det, undefined as unknown as { imageWidth: number; imageHeight: number }),
    /model's resolution \(4x2\).*imageWidth, imageHeight/,
  );
  assert.throws(() => getDepthAtDetection(depthResult(), det, { imageWidth: 0, imageHeight: 2 }), /must be > 0/);
  assert.throws(() => getDepthAtDetection(det, det, { imageWidth: 4, imageHeight: 2 }), /no depth map/);
  assert.throws(
    () => getDepthAtDetection(depthResult(), det, { imageWidth: 4, imageHeight: 2, mode: "p90" as "min" }),
    /unknown mode/,
  );
});
