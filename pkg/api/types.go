// Package api defines the public structs that clients can import.
// This is the unified schema (wire format) for every task — do NOT create a separate
// schema per model (see CLAUDE.md).
package api

// Task classifies the type of CV task.
type Task string

const (
	TaskDetection    Task = "detection"
	TaskSegmentation Task = "segmentation"
	TaskOpenVocab    Task = "open_vocab" // open-vocab feature (Grounding DINO)
)

// Result is the normalized output — a unified schema across tasks.
type Result struct {
	Task       Task        `json:"task"`
	Model      string      `json:"model"`
	Detections []Detection `json:"detections,omitempty"`
	Masks      []Mask      `json:"masks,omitempty"`
	DurationMs float64     `json:"duration_ms"`
}

// Detection is a bbox with class + confidence.
// BBox is ALWAYS in ORIGINAL image coordinates: [x, y, w, h] (top-left corner + width/height).
type Detection struct {
	BBox  [4]float64 `json:"bbox"`
	Class string     `json:"class"`
	Conf  float64    `json:"conf"`
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
	Model       string `json:"model"`
	ImageBase64 string `json:"image_base64"`
	Prompt      string `json:"prompt,omitempty"` // text: "cat. remote."
	Box         string `json:"box,omitempty"`    // "x,y,w,h" (multiple boxes: separated by ';')
	Point       string `json:"point,omitempty"`  // "x,y[,label]" (multiple: separated by ';')
}

// LoadRequest / UnloadRequest are used by /api/load and /api/unload.
type LoadRequest struct {
	Model string `json:"model"`
}

// ErrorResponse is the JSON body returned on error.
type ErrorResponse struct {
	Error string `json:"error"`
}
