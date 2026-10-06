// Package api defines the public structs that clients can import.
// This is the unified schema (wire format) for every task — do NOT create a separate
// schema per model (see CLAUDE.md).
package api

// Task classifies the type of CV task.
type Task string

const (
	TaskDetection         Task = "detection"
	TaskSegmentation      Task = "segmentation"
	TaskOpenVocab         Task = "open_vocab"         // open-vocab feature (Grounding DINO)
	TaskDepth             Task = "depth"              // monocular depth estimation
	TaskClassification    Task = "classification"     // image classification
	TaskEmbed             Task = "embed"              // image/text embedding (CLIP, ArcFace)
	TaskGrasp             Task = "grasp"              // planar parallel-jaw grasp synthesis
	TaskInstanceDetection Task = "instance_detection" // one-shot / template-based object detection
)

// Result is the normalized output — a unified schema across tasks.
type Result struct {
	Task            Task             `json:"task"`
	Model           string           `json:"model"`
	Device          string           `json:"device,omitempty"` // "cpu" | "gpu:0" | "gpu:0+trt"
	Hint            string           `json:"hint,omitempty"`   // setup recommendation (e.g. install TRT)
	Detections      []Detection      `json:"detections,omitempty"`
	Masks           []Mask           `json:"masks,omitempty"`
	Grasps          []Grasp          `json:"grasps,omitempty"`
	Classifications []Classification `json:"classifications,omitempty"` // top-K class predictions
	Embeddings      [][]float32      `json:"embeddings,omitempty"`      // one embedding vector per image/input
	DepthMap        []float32        `json:"depth_map,omitempty"`       // row-major HxW relative depth
	DepthWidth      int              `json:"depth_width,omitempty"`
	DepthHeight     int              `json:"depth_height,omitempty"`
	DurationMs      float64          `json:"duration_ms"`

	// Binary-friendly form of the two large float arrays, sent INSTEAD of depth_map /
	// embeddings when the request asks for encoding=base64 (see EncodeArraysBase64): the
	// base64 of the little-endian float32 bytes, row-major. DepthMapBase64 has shape
	// [depth_height, depth_width]; EmbeddingsBase64 has shape EmbeddingsShape = [N, D].
	// numpy.frombuffer(base64.b64decode(s), "<f4").reshape(shape) restores them bit for bit.
	DepthMapBase64   string `json:"depth_map_base64,omitempty"`
	EmbeddingsBase64 string `json:"embeddings_base64,omitempty"`
	EmbeddingsShape  []int  `json:"embeddings_shape,omitempty"`
}

// Classification is a single class prediction for TaskClassification.
type Classification struct {
	Class string  `json:"class"`
	Conf  float64 `json:"conf"`
}

// Detection is a bbox with class + confidence.
// BBox is ALWAYS in ORIGINAL image coordinates: [x, y, w, h] (top-left corner + width/height).
type Detection struct {
	BBox  [4]float64 `json:"bbox"`
	Class string     `json:"class"`
	Conf  float64    `json:"conf"`
}

// Grasp is a planar parallel-jaw grasp in ORIGINAL image coordinates.
//
// Class/Conf carry the source object's label and detector confidence when the grasp
// was produced in box mode (a detector found a named object, which was then segmented);
// both are empty/zero for class-agnostic grasps (no detector — whole-image automask).
// Quality is the analytic mask2grasp score and is always present.
type Grasp struct {
	X       float64 `json:"x"`
	Y       float64 `json:"y"`
	Theta   float64 `json:"theta"`           // in-plane gripper-closing angle (radians)
	Width   float64 `json:"width"`           // jaw opening, original-image pixels
	Quality float64 `json:"quality"`         // analytic grasp score in [0,1]
	Class   string  `json:"class,omitempty"` // source object label (box mode); "" if class-agnostic
	Conf    float64 `json:"conf,omitempty"`  // source detector confidence (box mode); 0 if class-agnostic
}

// Mask is a segmentation result. The mask is encoded with RLE (COCO-style, column-major counts).
type Mask struct {
	RLE  string     `json:"rle,omitempty"`
	BBox [4]float64 `json:"bbox,omitempty"`
	Conf float64    `json:"conf"`
}

// --- Wire types for the HTTP API ---

// ModelInfo describes a model in the registry + its runtime state.
type ModelInfo struct {
	Name    string `json:"name"`
	Task    Task   `json:"task"`
	License string `json:"license"`
	State   string `json:"state"` // "not_downloaded" | "available" | "loaded"

	// Client-resize hint: how far an SDK may shrink an image before uploading it without changing
	// what the model can see (the model resizes to its own fixed input anyway). At most one is
	// set; both null = send the image at full resolution (masks, OCR, depth-aligned grasping,
	// templates, keep_aspect, …). MaxUsefulSide bounds the image's LONGER side (models that fit
	// the image inside their input: letterbox, long_side); MaxUsefulShortSide bounds its SHORTER
	// side (models that fill their input on both axes: squash, center_crop). Each is 2 × the
	// model's larger input side, or the manifest's runtime.max_useful_side (longer side). A
	// region of interest is bounded the same way, measured on the region.
	MaxUsefulSide      *int `json:"max_useful_side"`
	MaxUsefulShortSide *int `json:"max_useful_short_side"`

	// AcceptsDepth: the model reads an uploaded depth map (the depth / depth_base64 fields of
	// /api/predict, pixel-aligned to the photo). SDKs upload a camera's depth frame only when it
	// is true (Python Client.watch(depth="auto")); every other model ignores the map. Only
	// `background` with a MiDaS session sets it today; a model that accepts depth never gets a
	// client-resize hint (the map is aligned to the full photo).
	AcceptsDepth bool `json:"accepts_depth"`
}

// PredictJSONRequest is the option list of /api/predict (and /api/preprocess). It is the JSON
// body, and it is ALSO the multipart form: every multipart field has the same name as the JSON
// key here (the server fills one from the other by the json tag), so a new option is declared
// once, in this struct. Only the binary parts differ: multipart sends the image and the depth
// map as file parts "image" and "depth", JSON as image_base64 / depth_base64.
// Prompt/Box/Point are optional, for models that need a prompt (SAM box, GroundingDINO text).
type PredictJSONRequest struct {
	Model       string  `json:"model"`
	ImageBase64 string  `json:"image_base64"`
	Prompt      string  `json:"prompt,omitempty"`   // text: "cat. remote."
	Box         string  `json:"box,omitempty"`      // "x,y,w,h" (multiple boxes: separated by ';')
	Point       string  `json:"point,omitempty"`    // "x,y[,label]" (multiple: separated by ';')
	MinSize     float64 `json:"min_size,omitempty"` // minimum bbox area as % of image area, 0 = no limit (e.g. 0.1 = 0.1%)
	MaxSize     float64 `json:"max_size,omitempty"` // maximum bbox area as % of image area, 0 = no limit (e.g. 90 = 90%)
	// Grasp models only: parallel-jaw opening bounds in ORIGINAL-image pixels. 0 = use the
	// manifest default. A candidate grasp is kept only if gripper_min <= width <= gripper_max.
	GripperMin float64 `json:"gripper_min,omitempty"`
	GripperMax float64 `json:"gripper_max,omitempty"`
	// GroundingDINO threshold overrides (0 = manifest/default). BoxThreshold filters object
	// queries by score; TextThreshold is a second floor on the SAME score (a query is kept only
	// above both). Neither changes labels: a detection is always named by one whole prompt
	// phrase. Used by grounding-dino/grounded-sam/grasp-gd; BoxThreshold also by owlvit.
	BoxThreshold  float64 `json:"box_threshold,omitempty"`
	TextThreshold float64 `json:"text_threshold,omitempty"`
	// Foreground model knobs (percent of image area): bg_max_area = a mask ≥ this is
	// background (support surface); fg_min_area = a mask < this is dropped as noise.
	// 0 = model default. Distinct from min_size/max_size (the output bbox-area filter).
	BgMaxArea float64 `json:"bg_max_area,omitempty"`
	FgMinArea float64 `json:"fg_min_area,omitempty"`
	// GridSize overrides the MobileSAM automask grid (N×N → N² decoder calls); larger =
	// catches more small objects but slower. 0 = model default. Used by mobile-sam (no prompt),
	// grasp (no box) and background (method automask).
	GridSize int `json:"grid_size,omitempty"`
	// Method selects the algorithm for models that offer several (the `background` model:
	// "auto" | "depth" | "sam" | "cv" | "automask"). "" = model default (auto for background).
	Method string `json:"method,omitempty"`
	// ClaimThreshold applies to textalign's `method: dual`. It is the probability in (0,1)
	// the supervised closed head must reach on a requested word before it names a detection
	// instead of the open head. 0 = model default; >= 1 means it never claims. It is NOT
	// conf_threshold, which decides
	// what is reported and is set to 0.001 by evaluation runs.
	ClaimThreshold float64 `json:"claim_threshold,omitempty"`
	// CropTemp is the softmax temperature the crop namer uses to turn cosines into a
	// probability over the requested words; that probability multiplies the reported
	// confidence. 0 = model default. Lower is more decisive.
	CropTemp float64 `json:"crop_temp,omitempty"`
	// ROI restricts processing to a region of interest "x,y,w,h" in ORIGINAL image pixels:
	// the server crops to it, runs the model on the crop, and maps results back. "" = full image.
	ROI string `json:"roi,omitempty"`
	// Dilate post-processes every output mask by a square-kernel morphology of |Dilate| px:
	// >0 enlarges (dilate), <0 shrinks (erode), 0 = off.
	Dilate int `json:"dilate,omitempty"`
	// Optional external depth map (RGB-D), raw little-endian array, base64-encoded. DepthDtype
	// is "uint16" (→ /65535) or "float32" (as-is); DepthWidth/Height give its resolution
	// (default = image size). Used by the background model's depth method instead of MiDaS.
	DepthBase64 string `json:"depth_base64,omitempty"`
	DepthDtype  string `json:"depth_dtype,omitempty"`
	DepthWidth  int    `json:"depth_width,omitempty"`
	DepthHeight int    `json:"depth_height,omitempty"`
	// TemplateName selects a named template set registered via POST /api/templates.
	// Used by instance_detection models (OWL-ViT, SiamRPN, …); ignored by all others.
	TemplateName string `json:"template_name,omitempty"`
	// Encoding of the large float arrays in the response (depth_map, embeddings): "json"
	// (default, number arrays) or "base64" (EncodingBase64). Also accepted as the query
	// parameter ?encoding=base64.
	Encoding string `json:"encoding,omitempty"`
}

// LoadRequest / UnloadRequest are used by /api/load and /api/unload.
type LoadRequest struct {
	Model string `json:"model"`
}

// ErrorResponse is the JSON body returned on error.
type ErrorResponse struct {
	Error string `json:"error"`
}
