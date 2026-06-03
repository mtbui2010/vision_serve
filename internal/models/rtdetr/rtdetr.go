// Package rtdetr implements the RT-DETR model (detection, Apache-2.0).
//
// RT-DETR is a real-time DETR-style, NMS-free architecture. The output is a fixed set
// of "queries", each with per-class logits + a bounding box. Do NOT apply NMS here.
//
// Architecturally nearly identical to RF-DETR. Key differences:
//   - Default input: 640×640 (manifest-configurable)
//   - COCO-80 class space (indices 0-79, no N/A gap) instead of RF-DETR's COCO-91
//
// Registered under the architecture name "rt-detr" via init() — adding this model
// does not require modifying the core.
package rtdetr

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("rt-detr", New)
}

type rtDETR struct {
	cfg models.Config
}

// New is the factory the lifecycle calls after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("rtdetr: invalid input width/height (%dx%d)", cfg.Width, cfg.Height)
	}
	if len(cfg.Labels) == 0 {
		// Do not hard-fail: detection still works, classes are shown as ids.
		// The manifest should declare labels: coco80.txt (80 contiguous classes, no N/A gap).
	}
	return &rtDETR{cfg: cfg}, nil
}

func (m *rtDETR) Name() string      { return m.cfg.Name }
func (m *rtDETR) Task() models.Task { return models.TaskDetection }

// InputName/OutputNames left empty -> the engine auto-detects I/O names from the ONNX file.
// Postprocess identifies which tensor is boxes/logits by SHAPE (last dim == 4), not by name.
func (m *rtDETR) InputName() string     { return "" }
func (m *rtDETR) OutputNames() []string { return nil }

func (m *rtDETR) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return m.preprocess(img)
}

func (m *rtDETR) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return m.postprocess(outs, meta)
}
