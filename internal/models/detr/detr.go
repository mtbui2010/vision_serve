// Package detr implements the DETR-style, NMS-free detectors (both Apache-2.0):
//
//   - RF-DETR — architecture "rf-detr" (the core detection model). COCO checkpoints emit
//     the 91-class "paper" label space (index 0 = N/A): pair with a 91-line labels file.
//   - RT-DETR — architecture "rt-detr" (onnx-community/RT-DETR-l-hf, 640×640). Emits the
//     contiguous COCO-80 space: pair with coco80.txt.
//
// The output is a fixed set of object queries, each with per-class logits + a box; one
// decoder serves both. Do NOT apply YOLO-style NMS here (CLAUDE.md). The two
// registrations differ only in how strictly they accept the exported outputs (see
// splitRF / splitRT) and in their error-message prefix.
//
// Registered via init() — adding a model does not require modifying the core.
package detr

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// Compile-time checks of the interfaces lifecycle type-asserts at load: a signature drift
// fails the build instead of silently changing how the model is run.
var _ models.Model = (*detr)(nil)

func init() {
	models.Register("rf-detr", NewRFDETR)
	models.Register("rt-detr", NewRTDETR)
}

// variant is what differs between the two registrations.
type variant struct {
	prefix string // error-message prefix ("rfdetr" / "rtdetr")
	// split identifies the boxes ([.., Q, 4]) and logits ([.., Q, C]) outputs.
	split func(outs []engine.Tensor) (boxes, logits engine.Tensor, err error)
}

var (
	rfVariant = variant{prefix: "rfdetr", split: splitRF}
	rtVariant = variant{prefix: "rtdetr", split: splitRT}
)

type detr struct {
	cfg models.Config
	v   variant
}

// NewRFDETR is the "rf-detr" factory the lifecycle calls after parsing the manifest.
func NewRFDETR(cfg models.Config) (models.Base, error) { return newDETR(cfg, rfVariant) }

// NewRTDETR is the "rt-detr" factory.
func NewRTDETR(cfg models.Config) (models.Base, error) { return newDETR(cfg, rtVariant) }

func newDETR(cfg models.Config, v variant) (models.Base, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("%s: invalid input width/height (%dx%d)", v.prefix, cfg.Width, cfg.Height)
	}
	m := &detr{cfg: cfg, v: v}
	if _, err := m.spec(); err != nil { // a declared preprocessing this decoder cannot map back
		return nil, err
	}
	// Labels are optional: without them classes are reported as class_<id>.
	return m, nil
}

func (m *detr) Name() string      { return m.cfg.Name }
func (m *detr) Task() models.Task { return models.TaskDetection }

// InputName/OutputNames left empty -> the engine auto-detects I/O names from the ONNX file.
// (Postprocess identifies which tensor is boxes/logits by SHAPE, not by name — more robust
// when export names differ across releases.)
func (m *detr) InputName() string     { return "" }
func (m *detr) OutputNames() []string { return nil }

func (m *detr) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return m.preprocess(img)
}

func (m *detr) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return m.postprocess(outs, meta)
}
