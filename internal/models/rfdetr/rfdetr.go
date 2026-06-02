// Package rfdetr implements the RF-DETR model (detection, Apache 2.0) — the free-tier core MVP.
//
// RF-DETR is a DETR-style, NMS-free architecture: the output is a fixed set of "queries",
// each query consisting of per-class logits + a box. Do NOT apply YOLO's NMS logic here (CLAUDE.md).
//
// Registered under the architecture name "rf-detr" via init() — adding a model does not require
// modifying the core.
package rfdetr

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("rf-detr", New)
}

type rfDETR struct {
	cfg models.Config
}

// New is the factory the lifecycle calls after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("rfdetr: invalid input width/height (%dx%d)", cfg.Width, cfg.Height)
	}
	if len(cfg.Labels) == 0 {
		// Do not hard-fail: detection still works, classes are just shown as ids.
		// (the manifest should declare labels: coco.txt)
	}
	return &rfDETR{cfg: cfg}, nil
}

func (m *rfDETR) Name() string      { return m.cfg.Name }
func (m *rfDETR) Task() models.Task { return models.TaskDetection }

// InputName/OutputNames left empty -> the engine auto-detects I/O names from the ONNX file.
// (Postprocess identifies which tensor is boxes/logits by SHAPE, not by name — more robust
// when export names differ across RF-DETR releases.)
func (m *rfDETR) InputName() string     { return "" }
func (m *rfDETR) OutputNames() []string { return nil }

func (m *rfDETR) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return m.preprocess(img)
}

func (m *rfDETR) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return m.postprocess(outs, meta)
}
