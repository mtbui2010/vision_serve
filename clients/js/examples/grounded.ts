/** Open-vocab segmentation example: Grounded-SAM (text → boxes → masks). */
import { Client } from "../src/index.js";

const client = new Client("http://localhost:11435");

const res = await client.predict("grounded-sam", "image.jpg", { prompt: "cat. remote." });
console.log(
  res.detections.map((d) => d.cls),
  "→",
  res.masks.length,
  "masks",
);
