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
	Classification = api.Classification
)

const (
	TaskDetection      = api.TaskDetection
	TaskSegmentation   = api.TaskSegmentation
	TaskOpenVocab      = api.TaskOpenVocab
	TaskDepth          = api.TaskDepth
	TaskClassification = api.TaskClassification
	TaskEmbed          = api.TaskEmbed
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
type Prompt struct {
	Text   string
	Boxes  [][4]float64
	Points []Point
}

// Empty reports whether the prompt carries no information.
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
