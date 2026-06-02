/** Detection example: run RF-DETR on an image and print the boxes. */
import { Client } from "../src/index.js";

const client = new Client("http://localhost:11435");

const res = await client.predict("rf-detr", "image.jpg");
console.log(`task=${res.task} model=${res.model} (${res.durationMs.toFixed(1)} ms)`);
for (const d of res.detections) {
  console.log(d.cls, d.conf.toFixed(3), d.bbox); // bbox = [x, y, w, h] in original pixels
}
