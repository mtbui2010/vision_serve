/** Segmentation example: MobileSAM with a box prompt → a decoded mask. */
import { Client } from "../src/index.js";

const client = new Client("http://localhost:11435");

// Box is [x, y, w, h] in ORIGINAL image coordinates.
const res = await client.predict("mobile-sam", "image.jpg", { box: [34, 58, 120, 240] });
console.log(`${res.masks.length} mask(s)`);

const mask = res.masks[0];
if (mask) {
  // Decode the column-major RLE to a row-major Uint8Array (1 = inside the mask).
  const flat = mask.toMask(640, 480); // pass the ORIGINAL width, height
  const area = flat.reduce((a, b) => a + b, 0);
  console.log("mask area (px):", area, "conf:", mask.conf.toFixed(3));
}
