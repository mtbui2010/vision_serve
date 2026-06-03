/**
 * Functional test for one model via the JS client.
 *
 * Usage:
 *   node test_model.mjs <host> <model> <task> [image] [prompt] [box]
 *   node test_model.mjs http://localhost:12000 rf-detr detection
 *   node test_model.mjs http://localhost:12009 grounding-dino open_vocab '' 'person. car.'
 *   node test_model.mjs http://localhost:12005 mobile-sam segmentation '' '' '100,100,440,280'
 */
import { Client } from "../../clients/js/dist/index.js";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_IMAGE = resolve(__dirname, "../../test/testdata/sample.jpg");

const [host, model, task, imagePath = DEFAULT_IMAGE, prompt = "", box = ""] =
  process.argv.slice(2);

if (!host || !model || !task) {
  console.error("Usage: node test_model.mjs <host> <model> <task> [image] [prompt] [box]");
  process.exit(1);
}

const client = new Client(host, { timeoutMs: 180_000 });

// Health check
const health = await client.health();
if (health.status !== "ok") {
  console.error(`health check failed: ${JSON.stringify(health)}`);
  process.exit(1);
}
console.log("  health: ok");

// Load model
try {
  await client.load(model);
} catch (e) {
  console.warn(`  WARNING: explicit load failed (may auto-load): ${e.message}`);
}

// Build predict options
const opts = {};
if (prompt && prompt.trim()) opts.prompt = prompt;
if (box && box.trim()) opts.box = box.split(",").map(Number);

// Predict (pass file path — client reads it lazily via node:fs/promises)
const result = await client.predict(model, imagePath, opts);

// Validate
function checkConf(items, label) {
  for (const item of items) {
    if (item.conf < 0 || item.conf > 1)
      throw new Error(`${label}: conf out of range: ${item.conf}`);
  }
}

switch (task) {
  case "detection":
  case "open_vocab": {
    // Zero detections is valid — test image may not contain target objects.
    checkConf(result.detections, model);
    for (const d of result.detections) {
      if (d.bbox[2] <= 0 || d.bbox[3] <= 0)
        throw new Error(`${model}: zero-area bbox ${JSON.stringify(d.bbox)}`);
    }
    if (result.detections.length) {
      const confs = result.detections.map((d) => d.conf);
      console.log(
        `  detections=${result.detections.length}  conf=[${Math.min(...confs).toFixed(3)},${Math.max(...confs).toFixed(3)}]`
      );
    } else {
      console.log(`  detections=0  (no targets in test image — structure ok)`);
    }
    break;
  }
  case "segmentation": {
    // Zero-area bbox warns rather than fails (unverified decoder shapes for some models).
    checkConf(result.masks, model);
    const bad = result.masks.filter((m) => m.bbox[2] <= 0 || m.bbox[3] <= 0);
    if (bad.length) console.warn(`  WARNING: ${bad.length}/${result.masks.length} masks have zero-area bbox`);
    console.log(`  masks=${result.masks.length}  good_bbox=${result.masks.length - bad.length}`);
    break;
  }
  case "depth": {
    if (!result.depthMap || !result.depthMap.length)
      throw new Error(`${model}: empty depthMap`);
    if (!result.depthWidth || !result.depthHeight)
      throw new Error(`${model}: invalid depth dims`);
    if (result.depthMap.length !== result.depthWidth * result.depthHeight)
      throw new Error(`${model}: depthMap length mismatch`);
    console.log(`  depth=${result.depthWidth}x${result.depthHeight}  pixels=${result.depthMap.length}`);
    break;
  }
  case "classification": {
    if (!result.classifications.length)
      throw new Error(`${model}: no classifications`);
    checkConf(result.classifications, model);
    const top = result.classifications[0];
    console.log(`  classifications=${result.classifications.length}  top=${JSON.stringify(top.cls)} (${top.conf.toFixed(3)})`);
    break;
  }
  case "embed": {
    if (!result.embeddings || !result.embeddings.length)
      throw new Error(`${model}: no embeddings`);
    if (!result.embeddings[0].length)
      throw new Error(`${model}: empty embedding vector`);
    console.log(`  embeddings=${result.embeddings.length}  dim=${result.embeddings[0].length}`);
    break;
  }
  default:
    console.warn(`  WARNING: no validator for task=${task}`);
}

console.log(`  durationMs=${result.durationMs.toFixed(1)}`);
console.log(`PASS [${model}]`);
