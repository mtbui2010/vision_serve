// Package api defines the public structs that clients can import.
// This is the unified schema (wire format) for every task — do NOT create a separate
// schema per model (see CLAUDE.md).
package api

// Task classifies the type of CV task.
type Task string

const (
	TaskDetection      Task = "detection"
	TaskSegmentation   Task = "segmentation"
	TaskOpenVocab      Task = "open_vocab"      // open-vocab feature (Grounding DINO)
	TaskDepth          Task = "depth"            // monocular depth estimation
	TaskClassification Task = "classification"   // image classification
	TaskEmbed          Task = "embed"            // image/text embedding (CLIP, ArcFace)
	TaskGrasp          Task = "grasp"            // planar parallel-jaw grasp synthesis
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
	Embeddings      [][]float32      `json:"embeddings,omitempty"`       // one embedding vector per image/input
	DepthMap        []float32        `json:"depth_map,omitempty"`        // row-major HxW relative depth
	DepthWidth      int              `json:"depth_width,omitempty"`
	DepthHeight     int              `json:"depth_height,omitempty"`
	DurationMs      float64          `json:"duration_ms"`
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
}

// PredictJSONRequest is the JSON body alternative to multipart when calling /api/predict.
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
	// queries by score; TextThreshold controls token→label assignment (lower => richer labels,
	// e.g. "canned coffee" instead of just "coffee"). Used by grounding-dino/grounded-sam/grasp-gd.
	BoxThreshold  float64 `json:"box_threshold,omitempty"`
	TextThreshold float64 `json:"text_threshold,omitempty"`
}

// LoadRequest / UnloadRequest are used by /api/load and /api/unload.
type LoadRequest struct {
	Model string `json:"model"`
}

// ErrorResponse is the JSON body returned on error.
type ErrorResponse struct {
	Error string `json:"error"`
}
