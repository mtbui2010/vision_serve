import { test } from "node:test";
import assert from "node:assert/strict";

import { ClientResize, Classification, Detection, Grasp, Mask, Result } from "../src/index.js";
import { mapResult } from "../src/resize.js";

/** assert.ok with a message: without one, a failing assert.ok re-parses the TS source and can hang under tsx. */
const ok = (cond: unknown, msg = "expected a truthy value"): void => assert.ok(cond, msg);

/** float32 values (what the server computes), so base64 round trips are exact. */
const f32 = (xs: number[]) => Array.from(Float32Array.from(xs));
const b64 = (xs: number[]) => Buffer.from(Float32Array.from(xs).buffer).toString("base64");

// One server answer per task, in the server's own field order with its omitempty rules
// (pkg/api/types.go), so JSON.stringify(Result.fromJSON(wire)) must give it back unchanged.
const WIRE: Record<string, Record<string, unknown>> = {
  detection: {
    task: "detection", model: "rf-detr", device: "gpu:0",
    detections: [{ bbox: [441.66, 31.67, 84.6, 131.9], class: "person", conf: 0.937 }, { bbox: [216, 227.2, 57.4, 92.5], class: "dog", conf: 0.922 }],
    duration_ms: 12.5,
  },
  segmentation: {
    task: "segmentation", model: "mobile-sam", device: "cpu",
    masks: [{ rle: "3 3", bbox: [1, 0, 1, 3], conf: 0.98 }, { bbox: [0, 0, 0, 0], conf: 0.1 }],
    duration_ms: 230,
  },
  open_vocab: {
    task: "open_vocab", model: "grounded-sam", device: "gpu:0", hint: "install TensorRT for ~1.5x",
    detections: [{ bbox: [0, 0, 2, 3], class: "dog", conf: 0.6 }],
    masks: [{ rle: "0 6", bbox: [0, 0, 2, 3], conf: 0.97 }],
    duration_ms: 480.25,
  },
  classification: {
    task: "classification", model: "efficientnet-b0",
    classifications: [{ class: "African elephant", conf: 0.81 }, { class: "tusker", conf: 0.12 }],
    duration_ms: 3,
  },
  depth: { task: "depth", model: "midas", depth_map: f32([0, 0.25, 0.1, 1, 0.3333, 0.7]), depth_width: 3, depth_height: 2, duration_ms: 20 },
  embed: { task: "embed", model: "clip", embeddings: [f32([0.1, -0.2, 0.3]), f32([1, 2, 3])], duration_ms: 5 },
  grasp: {
    task: "grasp", model: "grasp-rfdetr", device: "gpu:0",
    detections: [{ bbox: [10, 10, 50, 40], class: "bowl", conf: 0.72 }],
    masks: [{ rle: "0 6", bbox: [10, 10, 50, 40], conf: 0.9 }],
    grasps: [
      { x: 30, y: 25, theta: 0.5, width: 42, quality: 0.99, class: "bowl", conf: 0.72 },
      { x: 5, y: 5, theta: -1, width: 20, quality: 0.4 }, // class-agnostic: no class / conf on the wire
    ],
    duration_ms: 31,
  },
  instance_detection: {
    task: "instance_detection", model: "owlv2_base_patch16",
    detections: [{ bbox: [215, 230, 57, 89], class: "dog", conf: 0.93 }],
    duration_ms: 77,
  },
  empty: { task: "detection", model: "rf-detr", duration_ms: 0 },
};

test("JSON.stringify(result) is the server's wire format for every task", () => {
  for (const [name, wire] of Object.entries(WIRE)) {
    const res = Result.fromJSON(wire);
    assert.deepStrictEqual(JSON.parse(JSON.stringify(res)), wire, name);
    // Key order too (a diff against a saved server answer stays clean).
    assert.equal(JSON.stringify(res), JSON.stringify(wire), name);
    assert.deepStrictEqual(Result.fromJSON(JSON.parse(JSON.stringify(res))), res, name);
  }
});

test("element toJSON: `class` not `cls`, omitempty like the server", () => {
  assert.equal(JSON.stringify(new Detection([1, 2, 3, 4], "cup", 0.5)), '{"bbox":[1,2,3,4],"class":"cup","conf":0.5}');
  assert.equal(JSON.stringify(new Mask("", [0, 0, 1, 1], 0.2)), '{"bbox":[0,0,1,1],"conf":0.2}');
  assert.equal(JSON.stringify(new Classification("cat", 0.9)), '{"class":"cat","conf":0.9}');
  assert.equal(JSON.stringify(new Grasp(1, 2, 0, 3, 0.5)), '{"x":1,"y":2,"theta":0,"width":3,"quality":0.5}');
  assert.equal(JSON.stringify(new Grasp(1, 2, 0, 3, 0.5, "cup", 0.8)), '{"x":1,"y":2,"theta":0,"width":3,"quality":0.5,"class":"cup","conf":0.8}');
  // The JSON is a copy: changing it does not change the result.
  const d = new Detection([1, 2, 3, 4], "cup", 0.5);
  (d.toJSON().bbox as number[])[0] = 99;
  assert.equal(d.bbox[0], 1);
});

test("base64 arrays: read, written back with { encoding: 'base64' }, and round-tripped", () => {
  const depthWire = { task: "depth", model: "midas", depth_width: 3, depth_height: 2, duration_ms: 20, depth_map_base64: b64([0, 0.25, 0.1, 1, 0.3333, 0.7]) };
  const embWire = { task: "embed", model: "clip", duration_ms: 5, embeddings_base64: b64([0.1, -0.2, 0.3, 1, 2, 3]), embeddings_shape: [2, 3] };
  for (const wire of [depthWire, embWire]) {
    const res = Result.fromJSON(wire);
    assert.deepStrictEqual(res.toJSON({ encoding: "base64" }), wire);
    assert.deepStrictEqual(Result.fromJSON(JSON.parse(JSON.stringify(res))), res); // as numbers
    assert.deepStrictEqual(Result.fromJSON(JSON.parse(JSON.stringify(res.toJSON({ encoding: "base64" })))), res); // as base64
  }
  assert.deepStrictEqual(Result.fromJSON(depthWire).depthMap, WIRE.depth!.depth_map);
  assert.deepStrictEqual(Result.fromJSON(embWire).embeddings, WIRE.embed!.embeddings);
  // Ragged rows have no [N, D] shape: they stay numbers.
  const ragged = new Result("embed", "m", [], [], [], [], 0, 0, [[1, 2], [3]], 1);
  assert.deepStrictEqual(ragged.toJSON({ encoding: "base64" }), { task: "embed", model: "m", embeddings: [[1, 2], [3]], duration_ms: 1 });
  assert.throws(() => ragged.toJSON({ encoding: "hex" as "json" }), /encoding must be/);
});

test("clientResize survives the round trip, masks keep their RLE size", () => {
  // A mask computed on a 2x3 photo sent for a 4x6 original.
  const sent = Result.fromJSON({ ...WIRE.open_vocab, hint: "" });
  const cr = new ClientResize(4, 6, 2, 3, 90, "resize=3");
  const res = mapResult(sent, cr);
  assert.deepEqual(res.masks[0]!.rleSize, [2, 3]);
  const json = JSON.parse(JSON.stringify(res));
  assert.deepStrictEqual(json.client_resize, { original_width: 4, original_height: 6, sent_width: 2, sent_height: 3, jpeg_quality: 90, reason: "resize=3" });
  const back = Result.fromJSON(json);
  assert.deepStrictEqual(back, res);
  assert.deepEqual(back.masks[0]!.toMask(4, 6), res.masks[0]!.toMask(4, 6));
  // Sent whole (the loopback rule): recorded, masks untouched.
  const whole = mapResult(Result.fromJSON(WIRE.segmentation!), new ClientResize(2, 3, 2, 3, null, "loopback: scale 0.6 > 0.5, sent as is"));
  assert.deepStrictEqual(Result.fromJSON(JSON.parse(JSON.stringify(whole))), whole);
  // A server answer never has the field.
  assert.equal("client_resize" in JSON.parse(JSON.stringify(Result.fromJSON(WIRE.grasp!))), false);
});

test("Grasp pose and jaw contacts", () => {
  const g = new Grasp(10, 20, Math.PI / 2, 8, 0.9, "cup", 0.7);
  assert.deepEqual(g.pose, [10, 20, 8, Math.PI / 2]);
  const [[x0, y0], [x1, y1]] = g.contacts();
  ok(Math.abs(x0 - 10) < 1e-12 && Math.abs(x1 - 10) < 1e-12);
  assert.deepEqual([y0, y1], [16, 24]);
  assert.deepEqual(g.contactsFlat(), [x0, y0, x1, y1]);
});

test("groupByClass: each group holds only its class's detections, masks and grasps", () => {
  const res = Result.fromJSON({
    task: "grasp", model: "grasp-rfdetr", device: "gpu:0", hint: "h",
    detections: [
      { bbox: [0, 0, 10, 10], class: "cup", conf: 0.9 },
      { bbox: [20, 0, 10, 10], class: "bowl", conf: 0.8 },
      { bbox: [40, 0, 10, 10], class: "cup", conf: 0.7 },
    ],
    masks: [
      { rle: "", bbox: [20, 0, 10, 10], conf: 0.95 }, // bowl's box
      { rle: "", bbox: [0, 0, 10, 10], conf: 0.94 }, // first cup's box
      { rle: "", bbox: [1, 2, 3, 4], conf: 0.5 }, // no detection: ""
    ],
    grasps: [
      { x: 5, y: 5, theta: 0, width: 4, quality: 0.9, class: "cup", conf: 0.9 },
      { x: 25, y: 5, theta: 0, width: 4, quality: 0.8, class: "bowl", conf: 0.8 },
      { x: 45, y: 5, theta: 0, width: 4, quality: 0.7, class: "cup", conf: 0.7 },
      { x: 2, y: 3, theta: 0, width: 4, quality: 0.6 }, // class-agnostic, inside the first cup's box
      { x: 99, y: 99, theta: 0, width: 4, quality: 0.5 }, // class-agnostic, in no box: ""
    ],
    classifications: [{ class: "cup", conf: 0.5 }],
    depth_map: [0, 1], depth_width: 2, depth_height: 1,
    embeddings: [[1, 2]],
    duration_ms: 9,
  });
  const groups = res.groupByClass();
  assert.deepEqual(Object.keys(groups), ["cup", "bowl", ""]);
  const cup = groups.cup!;
  assert.deepEqual(cup.detections.map((d) => d.conf), [0.9, 0.7]);
  assert.deepEqual(cup.masks.map((m) => m.conf), [0.94]);
  assert.deepEqual(cup.grasps.map((g) => g.quality), [0.9, 0.7, 0.6]);
  assert.deepEqual(groups.bowl!.grasps.map((g) => g.quality), [0.8]);
  assert.deepEqual(groups[""]!.masks.map((m) => m.conf), [0.5]);
  assert.deepEqual(groups[""]!.grasps.map((g) => g.quality), [0.5]);
  assert.equal(groups[""]!.detections.length, 0);
  for (const g of Object.values(groups)) {
    // Per-image data is not copied into every group.
    assert.deepEqual([g.classifications, g.depthMap, g.depthWidth, g.depthHeight, g.embeddings], [[], [], 0, 0, []]);
    assert.deepEqual([g.task, g.model, g.device, g.hint, g.durationMs], ["grasp", "grasp-rfdetr", "gpu:0", "h", 9]);
  }
  // The items are the result's own objects.
  assert.equal(cup.detections[0], res.detections[0]);
});

test("filterGrasps: the best k per source detection, class-agnostic grasps by position", () => {
  const res = Result.fromJSON({
    task: "grasp", model: "m",
    detections: [
      { bbox: [0, 0, 100, 100], class: "bowl", conf: 0.8 },
      { bbox: [10, 10, 20, 20], class: "carrot", conf: 0.7 },
    ],
    grasps: [
      // A bowl grasp whose centre lies inside the carrot's box is still the bowl's.
      { x: 15, y: 15, theta: 0, width: 4, quality: 0.5, class: "bowl", conf: 0.8 },
      { x: 50, y: 50, theta: 0, width: 4, quality: 0.9, class: "bowl", conf: 0.8 },
      { x: 60, y: 60, theta: 0, width: 4, quality: 0.7, class: "bowl", conf: 0.8 },
      { x: 20, y: 20, theta: 0, width: 4, quality: 0.6, class: "carrot", conf: 0.7 },
      { x: 21, y: 21, theta: 0, width: 4, quality: 0.65, class: "carrot", conf: 0.7 },
      // Class-agnostic: the smallest box containing the centre (the carrot's: third of that
      // group, so not kept with k = 2), else a label bucket.
      { x: 12, y: 12, theta: 0, width: 4, quality: 0.3 },
      { x: 500, y: 500, theta: 0, width: 4, quality: 0.2 },
      // Class-aware, its detection filtered out: grouped by (class, conf).
      { x: 1, y: 1, theta: 0, width: 4, quality: 0.1, class: "cup", conf: 0.5 },
      { x: 2, y: 2, theta: 0, width: 4, quality: 0.15, class: "cup", conf: 0.5 },
    ],
    duration_ms: 1,
  });
  const q = (r: Result) => r.grasps.map((g) => g.quality);
  assert.deepEqual(q(res.filterGrasps(2)), [0.9, 0.7, 0.65, 0.6, 0.2, 0.15, 0.1]);
  assert.deepEqual(q(res.filterGrasps(3)), [0.9, 0.7, 0.5, 0.65, 0.6, 0.3, 0.2, 0.15, 0.1]);
  assert.deepEqual(q(res.filterGrasps(1)), [0.9, 0.65, 0.2, 0.15]);
  assert.equal(res.filterGrasps(0), res);
  assert.equal(res.filterGrasps(null), res);
  assert.equal(res.filterGrasps(1).detections, res.detections);
});
