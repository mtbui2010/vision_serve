import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  Grasp,
  Result,
  backproject,
  cameraDistance,
  getDepthAtDetection,
  graspDistances,
  objectDistances,
  selectTargetGrasp,
  selectTargetGraspIndex,
  selectTargetObject,
  selectTargetObjectIndex,
  type DepthImage,
  type SelectGraspOptions,
  type SelectObjectOptions,
} from "../src/index.js";

// Shared cases generated from the Python SDK (clients/testdata/gen_postprocess_sync.py); Python
// checks the same file in clients/python/tests/test_postprocess_sync.py.
const fx = JSON.parse(readFileSync(new URL("../../testdata/postprocess_sync.json", import.meta.url), "utf8"));

type Kw = Record<string, unknown>;

function depthImage(name: string): DepthImage {
  const d = fx.depth[name];
  const data = d.dtype === "uint16" ? Uint16Array.from(d.data) : Float32Array.from(d.data);
  return { data, width: d.width, height: d.height };
}
const detResult = () => Result.fromJSON({ task: "detection", model: "m", detections: fx.detections });
const maskResult = () => Result.fromJSON({ task: "segmentation", model: "m", masks: fx.masks });
const grasps = (): Grasp[] => fx.grasps.map((g: Record<string, unknown>) => Grasp.fromJSON(g));

/** Python computes in float32 where numpy does; the JS port matches to float32 precision. */
function close(got: unknown, want: unknown, what: string, rel = 1e-6): void {
  if (typeof want === "number") {
    assert.equal(typeof got, "number", `${what}: got ${String(got)}, want ${want}`);
    const g = got as number;
    assert.ok(Math.abs(g - want) <= rel * Math.max(1, Math.abs(want)), `${what}: got ${g}, want ${want}`);
  } else if (Array.isArray(want)) {
    assert.ok(Array.isArray(got) && got.length === want.length, `${what}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
    want.forEach((w, i) => close((got as unknown[])[i], w, `${what}[${i}]`, rel));
  } else {
    assert.deepEqual(got, want, what);
  }
}

test("backproject / cameraDistance match the shared cases", () => {
  const K = fx.intrinsics as number[];
  const Kobj = { fx: K[0]!, fy: K[1]!, cx: K[2]!, cy: K[3]! };
  for (const c of fx.backproject) {
    close(backproject(c.u, c.v, c.z, K), c.want, `backproject ${JSON.stringify(c)}`, 1e-12);
    close(backproject(c.u, c.v, c.z, Kobj), c.want, "backproject (object intrinsics)", 1e-12);
    close(cameraDistance(c.u, c.v, c.z, K), c.distance, "cameraDistance", 1e-12);
  }
  assert.throws(() => backproject(0, 0, 1, [1, 2, 3]), /4 values/);
  assert.throws(() => backproject(0, 0, 1, { fx: 1 } as never), /intrinsics must be/);
});

test("Grasp.pose / contacts / contactsFlat match the shared cases", () => {
  for (const c of fx.grasp_geometry) {
    const g = Grasp.fromJSON(c.grasp);
    close(g.pose, c.pose, "pose", 1e-12);
    close(g.contacts(), c.contacts, "contacts", 1e-12);
    close(g.contactsFlat(), c.contacts_flat, "contactsFlat", 1e-12);
  }
});

test("getDepthAtDetection with a metric depth image matches the shared cases", () => {
  for (const c of fx.depth_at_detection) {
    const res = c.items === "masks" ? maskResult() : detResult();
    const got = getDepthAtDetection(depthImage(c.depth), res, {
      mode: c.mode,
      depthScale: c.depth_scale,
      imageWidth: c.image_size?.[0],
      imageHeight: c.image_size?.[1],
    });
    close(got, c.want, `getDepthAtDetection ${JSON.stringify({ ...c, want: undefined })}`);
  }
});

test("objectDistances matches the shared cases", () => {
  for (const c of fx.object_distances) {
    const res = c.items === "masks" ? maskResult() : detResult();
    const got = objectDistances(depthImage(c.depth), res, fx.intrinsics, { mode: c.mode, depthScale: c.depth_scale });
    close(got, c.want, `objectDistances ${JSON.stringify({ ...c, want: undefined })}`);
  }
});

test("graspDistances matches the shared cases (half-to-even rounding of the centre)", () => {
  for (const c of fx.grasp_distances) {
    const got = graspDistances(depthImage(c.depth), grasps(), fx.intrinsics, { window: c.window, depthScale: c.depth_scale });
    close(got, c.want, `graspDistances ${JSON.stringify({ ...c, want: undefined })}`);
  }
});

/** The Python kwargs of a case as the JS options. */
function jsOptions(kw: Kw): Record<string, unknown> {
  const o: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(kw)) {
    if (k === "items") continue;
    if (k === "depth") o.depth = depthImage(v as string);
    else if (k === "intrinsics") o.intrinsics = fx.intrinsics;
    else if (k === "image_size") [o.imageWidth, o.imageHeight] = v as number[];
    else o[k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase())] = v;
  }
  return o;
}

test("selectTargetObject picks what Python picks in every shared case", () => {
  for (const c of fx.select_target_object) {
    const res = c.kwargs.items === "masks" ? maskResult() : detResult();
    const opts = jsOptions(c.kwargs) as SelectObjectOptions;
    const [obj, idx] = selectTargetObjectIndex(res, opts);
    assert.equal(idx, c.want_index, `selectTargetObject ${JSON.stringify(c.kwargs)}`);
    const items = res.detections.length ? res.detections : res.masks;
    assert.equal(obj, idx >= 0 ? items[idx] : null);
    assert.equal(selectTargetObject(res, opts), obj);
  }
});

test("selectTargetGrasp picks what Python picks in every shared case", () => {
  for (const c of fx.select_target_grasp) {
    const gs = grasps();
    const opts = jsOptions(c.kwargs) as SelectGraspOptions;
    const [g, idx] = selectTargetGraspIndex(gs, opts);
    assert.equal(idx, c.want_index, `selectTargetGrasp ${JSON.stringify(c.kwargs)}`);
    assert.equal(g, idx >= 0 ? gs[idx] : null); // the same object, so toSVG({ targetGrasp }) finds it
    assert.equal(selectTargetGrasp(gs, opts), g);
  }
});

test("a server depth Result is refused for metric distances; options are checked", () => {
  const depth = Result.fromJSON({ task: "depth", model: "midas", depth_map: [0, 0.5, 1, 1], depth_width: 2, depth_height: 2 });
  assert.throws(() => objectDistances(depth as never, detResult(), fx.intrinsics), /RELATIVE inverse depth/);
  assert.throws(() => graspDistances(depth as never, grasps(), fx.intrinsics), /RELATIVE inverse depth/);
  assert.throws(
    () => selectTargetObject(detResult(), { targetDistance: 1, depth: depth as never, intrinsics: fx.intrinsics }),
    /RELATIVE inverse depth/,
  );
  assert.throws(() => selectTargetObject(detResult(), { targetDistance: 1 }), /requires depth and intrinsics/);
  assert.throws(() => selectTargetGrasp(grasps(), { targetDistance: 1, depth: depthImage("mm") }), /requires depth and intrinsics/);
  assert.throws(() => selectTargetObject(detResult(), { nearPoint: "center" }), /imageWidth and imageHeight/);
  assert.throws(() => getDepthAtDetection(depth, detResult(), { imageWidth: 8, imageHeight: 6, depthScale: 0.001 }), /cannot turn a server depth Result/);
  assert.throws(() => getDepthAtDetection(depthImage("mm"), detResult(), { mode: "toString" as "min" }), /unknown mode/);
  assert.throws(() => graspDistances({ data: [1, 2, 3], width: 2, height: 2 }, grasps(), fx.intrinsics), /3 values, width \* height = 4/);
  // A plain number[] is metres (scale 1), like a float array.
  const m = depthImage("m");
  assert.deepEqual(
    graspDistances({ data: Array.from(m.data), width: m.width, height: m.height }, grasps(), fx.intrinsics),
    graspDistances(m, grasps(), fx.intrinsics),
  );
});
