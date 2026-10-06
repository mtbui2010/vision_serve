/**
 * VisionServe TypeScript/JavaScript client SDK.
 *
 * Talks to the VisionServe HTTP server (the Go runtime) over REST. See the repo
 * README and `clients/js/README.md` for usage.
 */

export { Client, VisionServeError, normalizePrompt, isLoopback } from "./client.js";
export type { ImageInput, BoxInput, PointInput, DepthInput, PredictOptions, ClientOptions } from "./client.js";
export { Result, Detection, Mask, Grasp, ModelInfo, Classification } from "./types.js";
export type { Task, ModelState } from "./types.js";
export { filterBySize, getDepthAtDetection } from "./filter.js";
export type { SizeFilterOptions, DepthAtDetectionOptions, DepthMode } from "./filter.js";
export { toSVG, classColour, classColourMap } from "./visualize.js";
export type { SVGOptions } from "./visualize.js";
export {
  backproject,
  cameraDistance,
  objectDistances,
  graspDistances,
  selectTargetObject,
  selectTargetObjectIndex,
  selectTargetGrasp,
  selectTargetGraspIndex,
} from "./postprocess.js";
export type {
  CameraIntrinsics,
  IntrinsicsInput,
  DepthImage,
  ObjectDistanceOptions,
  GraspDistanceOptions,
  SelectObjectOptions,
  SelectGraspOptions,
} from "./postprocess.js";
export { ClientResize, browserCodec, sharpCodec, probeHeader, targetSize } from "./resize.js";
export type { ImageCodec, ImageProbe, ResizeOption } from "./resize.js";
