import { test } from "node:test";
import assert from "node:assert/strict";

import { Client, VisionServeError, Result, Mask, Detection } from "../src/index.js";

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
