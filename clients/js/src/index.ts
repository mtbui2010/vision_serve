/**
 * VisionServe TypeScript/JavaScript client SDK.
 *
 * Talks to the VisionServe HTTP server (the Go runtime) over REST. See the repo
 * README and `clients/js/README.md` for usage.
 */

export { Client, VisionServeError } from "./client.js";
export type { ImageInput, BoxInput, PointInput, PredictOptions, ClientOptions } from "./client.js";
export { Result, Detection, Mask, ModelInfo } from "./types.js";
export type { Task, ModelState } from "./types.js";
