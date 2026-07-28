// Package models defines the Model interface (the core abstraction) and the
// registry pattern used to add new models WITHOUT modifying the core.
//
// Adding a new model = create package internal/models/<name>/ + implement Model +
// call models.Register(name, factory) in init(). See docs/contributing-models.md.
package models

import (
	"fmt"
	"image"
	"sort"
	"sync"

	"visionserve/internal/engine"
	"visionserve/pkg/api"
)

// The types below are aliases to the public schema in pkg/api, so model
// implementations can use short names while keeping a SINGLE unified schema.
type (
	Task           = api.Task
	Result         = api.Result
	Detection      = api.Detection
	Mask           = api.Mask
	Grasp          = api.Grasp
	Classification = api.Classification
)

const (
	TaskDetection      = api.TaskDetection
	TaskSegmentation   = api.TaskSegmentation
	TaskOpenVocab      = api.TaskOpenVocab
	TaskDepth          = api.TaskDepth
	TaskClassification = api.TaskClassification
	TaskEmbed          = api.TaskEmbed
	TaskGrasp          = api.TaskGrasp
)

// Model is the interface every model must implement.
//
// Design note: Infer (calling the ONNX session) is NOT part of this interface —
// that is handled by engine + lifecycle. A model only focuses on pre/postprocess
// (the part that differs between architectures). This is a clean seam for
// community contributions.
type Model interface {
	// Name identifies the model (matches the directory / manifest name).
	Name() string

	// Task is the type of CV task.
	Task() Task

	// InputName / OutputNames give the I/O tensor names the engine must bind.
	// Returning nil/"" means "let the engine auto-detect from the ONNX file".
	InputName() string
	OutputNames() []string

	// Preprocess: original image -> input tensor (resized/normalized/letterboxed).
	// Also returns metadata so postprocess can map results back to ORIGINAL image coordinates.
	Preprocess(img image.Image) (engine.Tensor, PreprocessMeta, error)

	// Postprocess: raw output tensors -> normalized Result.
	// outs follow the order of OutputNames() (or the model's export order if left empty).
	Postprocess(outs []engine.Tensor, meta PreprocessMeta) (Result, error)
}

// PreprocessMeta holds the info needed to map results back to original image coordinates.
// With letterbox, the scale on both axes is usually equal (aspect ratio preserved).
type PreprocessMeta struct {
	OrigWidth  int
	OrigHeight int
	ScaleX     float64 // input_x = orig_x * ScaleX + PadX
	ScaleY     float64
	PadX       int
	PadY       int
}

// Config is the configuration derived from the manifest, passed to the factory when
// creating a model. The registry is responsible for filling this struct (reading the
// YAML + loading the labels file).
type Config struct {
	Name      string
	Width     int
	Height    int
	Layout    string // "NCHW" | "NHWC"
	Mean      []float32
	Std       []float32
	Letterbox bool

	// Postprocess
	PostType   string // e.g. "detr"
	BoxFormat  string // e.g. "cxcywh" | "xyxy"
	ConfThresh float64
	TextThresh float64 // GroundingDINO: threshold for assigning token→label (box_threshold = ConfThresh)
	MaxDet     int
	Labels     []string

	// Multi-session / auxiliary files (for PipelineModel). Dir = model directory (resolve
	// auxiliary files like vocab.txt); Files = role → absolute ONNX path (e.g. encoder/decoder).
	Dir   string
	Files map[string]string

	// Grasp composition (architecture "grasp"). Segmenter picks the mask backbone
	// (default "mobile-sam"); Detector is optional (e.g. "rf-detr", "grounding-dino") —
	// empty means class-agnostic (automask). GripperMin/Max are the default parallel-jaw
	// opening bounds in pixels (a request may override them via the Prompt).
	Detector   string
	Segmenter  string
	GripperMin float64
	GripperMax float64

	// Instance detection config (owlvit, siamrpn, …). Populated from the manifest's
	// [instance] block by lifecycle.Manager. Models that do not use them leave them zero.
	InstanceSimThreshold float64
	InstanceMaxTemplates int
	InstancePatchSize    int
}

// Base is the minimal interface every model implements (simple Model and
// multi-session PipelineModel both satisfy it). Lifecycle type-asserts to the concrete
// interface (Model vs PipelineModel) to pick the inference path.
type Base interface {
	Name() string
	Task() Task
}

// Point is a SAM point prompt in ORIGINAL image coordinates.
// Label: 1 = foreground, 0 = background (SAM convention).
type Point struct {
	X, Y  float64
	Label int
}

// Prompt carries optional prompt data for models that need one:
//   - SAM: Boxes (each [x,y,w,h] in original-image coords) and/or Points.
//   - GroundingDINO / Grounded-SAM: Text (e.g. "cat. remote.").
//
// rf-detr ignores the prompt (it implements the plain Model interface).
//
// The trailing fields are per-request predict OPTIONS (not prompt content): size
// filters and grasp gripper bounds. Models that don't use them ignore them; the
// grasp model reads them in Infer. They live here because Infer's signature only
// carries (img, Prompt, Runner), so this is the single channel for per-call params.
type Prompt struct {
	Text   string
	Boxes  [][4]float64
	Points []Point

	// MinSize/MaxSize: object bbox-area filter as a percent of image area (0 = no
	// limit), same semantics as api.FilterBySizePct. The grasp model applies these
	// to detections (box mode) or masks (automask) BEFORE deriving grasps.
	MinSize float64
	MaxSize float64
	// GripperMin/GripperMax: parallel-jaw opening bounds in ORIGINAL-image pixels for
	// the grasp model (0 = use the manifest default).
	GripperMin float64
	GripperMax float64
	// BoxThresh/TextThresh: per-request GroundingDINO threshold overrides (0 = use the
	// manifest value, else the built-in default). BoxThresh filters object queries by
	// score (= box_threshold); TextThresh controls which prompt tokens are assigned as a
	// detection's label (= text_threshold). Used by grounding-dino, grounded-sam, grasp-gd.
	BoxThresh  float64
	TextThresh float64
	// BgMaxArea/FgMinArea: foreground model knobs (percent of image area). A MobileSAM
	// automask is treated as BACKGROUND when its area ≥ BgMaxArea (a support surface), and
	// dropped as noise when its area < FgMinArea. 0 = use the model default. These are
	// SEPARATE from MinSize/MaxSize (which are an output bbox-area post-filter applied by
	// the handler — that would drop the foreground union, whose bbox spans the image).
	BgMaxArea float64
	FgMinArea float64
	// GridSize: per-request override for the MobileSAM Automatic Mask Generator grid
	// (N×N point prompts → N² decoder calls). 0 = use the model's default. Larger N
	// catches more small objects but is slower. Read per-call (thread-safe), e.g. by the
	// foreground model to let a request trade speed for coverage.
	GridSize int
	// Method: per-request algorithm selector for models that offer several (the
	// `background` model: "depth" | "sam" | "cv" | "automask"). "" = the model default.
	Method string
	// ROI: optional region of interest [x, y, w, h] in ORIGINAL image pixels. When set
	// (w>0 && h>0), the SERVER crops the image to this rectangle, runs the model on the
	// crop only, and maps results back to original coordinates. Handled in the HTTP layer,
	// generic to every model — individual models never see it.
	ROI [4]float64
	// Dilate: post-process every output MASK by a square-kernel morphology of |Dilate|
	// pixels — >0 enlarges (dilate), <0 shrinks (erode), 0 = no-op. Generic, handled in the
	// HTTP layer (and run CLI); individual models never see it.
	Dilate int
	// Depth: optional external depth map (RGB-D sensor), row-major, length DepthW*DepthH,
	// ALIGNED to the RGB image. Values are normalized: uint16 → /65535 ([0,1]); float kept
	// as-is. Invalid pixels (uint16 0, or float ≤0/NaN) are NaN. When provided, the
	// `background` model's depth method fits the plane on THIS instead of running MiDaS.
	Depth          []float32
	DepthW, DepthH int

	// TemplateName is the name of a registered template set (see /api/templates).
	// Used by instance_detection models (OWL-ViT, SiamRPN, …). The lifecycle manager
	// resolves this to TemplateImages before calling the model.
	TemplateName string
	// TemplateImages carries the resolved template crops (set by lifecycle from the
	// template store). Individual models read this slice; they never populate it.
	TemplateImages []image.Image
}

// Empty reports whether the prompt carries no PROMPT content (options aside).
func (p Prompt) Empty() bool {
	return p.Text == "" && len(p.Boxes) == 0 && len(p.Points) == 0
}

// Runner is the gateway a PipelineModel uses to run its ONNX sessions. The heavy
// sessions are owned and kept alive by lifecycle.Manager (VRAM-safe, per CLAUDE.md);
// the model only orchestrates calls to them by "role".
type Runner interface {
	// Run executes the session registered under role, binding inputs by name.
	Run(role string, inputs map[string]engine.Tensor) ([]engine.Tensor, error)
	// InputNames returns the input names the role's session expects.
	InputNames(role string) []string
	// OutputNames returns the output order of the role's session (to index Run's result).
	OutputNames(role string) []string
}

// PoolSizer is an optional interface for PipelineModels that want multiple concurrent
// session copies for specific roles. Returning n > 1 for a role makes lifecycle create
// a SessionPool of n identical sessions, allowing n concurrent inferences on that role
// instead of serializing all callers on one mutex.
//
// Example: MobileSAM decoder pool=4 lets AMG run 4 decoder calls simultaneously.
// Models that don't implement PoolSizer get a single session per role (pool of 1).
type PoolSizer interface {
	PoolSizes() map[string]int
}

// PipelineModel is for models that need a prompt and/or multiple chained ONNX sessions
// (MobileSAM = encoder+decoder, GroundingDINO = single but prompted, Grounded-SAM =
// GroundingDINO + SAM). The model drives its own inference via Runner + Prompt.
//
// Note: this deliberately differs from Model (where engine+lifecycle drive a single
// pre→infer→post). Multi-stage prompted models need to control chaining themselves;
// lifecycle still LOADS and owns the sessions, so VRAM management stays centralized.
type PipelineModel interface {
	Base
	// Roles lists the session keys this model needs; each must exist in Config.Files.
	Roles() []string
	// Infer runs the full pipeline and returns the unified Result.
	Infer(img image.Image, prompt Prompt, r Runner) (Result, error)
}

// Factory builds a model (Model or PipelineModel) from Config (parsed manifest).
type Factory func(cfg Config) (Base, error)

var (
	mu       sync.RWMutex
	registry = map[string]Factory{}
)

// Register registers a factory for a model type. Call it in the model package's init().
// Panics on a duplicate name — this is a programming error at startup, not a runtime error.
func Register(name string, f Factory) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := registry[name]; dup {
		panic(fmt.Sprintf("models: factory %q already registered", name))
	}
	registry[name] = f
}

// New creates a model by its registered name. The name here is the ARCHITECTURE TYPE
// (e.g. "rf-detr"), usually matching manifest.name; if different, use a dedicated field
// in the manifest.
func New(name string, cfg Config) (Base, error) {
	mu.RLock()
	f, ok := registry[name]
	mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("models: no factory registered for %q (registered: %v)", name, Registered())
	}
	return f(cfg)
}

// Registered returns the list of registered model names (sorted).
func Registered() []string {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]string, 0, len(registry))
	for k := range registry {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// IsRegistered reports whether a model type already has a factory.
func IsRegistered(name string) bool {
	mu.RLock()
	defer mu.RUnlock()
	_, ok := registry[name]
	return ok
}
