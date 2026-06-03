// Package classification implements image classification models:
//   - EfficientNet-B0 (Apache-2.0) — 224×224 input
//   - MobileNet-V3-Small (Apache-2.0) — 224×224 input
//
// Both are plain Model implementations: engine+lifecycle drive the single ONNX session.
// Output: single tensor [1, num_classes] float32 logits -> softmax -> top-K predictions.
//
// Registered via init() — no changes to core required.
package classification

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("efficientnet", New)
	models.Register("mobilenet-v3", New)
}

type classificationModel struct {
	cfg models.Config
}

// New is the factory the lifecycle calls after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("classification: invalid input dimensions (%dx%d)", cfg.Width, cfg.Height)
	}
	if len(cfg.Labels) == 0 {
		// Labels are optional at model creation; class indices are returned as fallback.
	}
	return &classificationModel{cfg: cfg}, nil
}

func (m *classificationModel) Name() string      { return m.cfg.Name }
func (m *classificationModel) Task() models.Task { return models.TaskClassification }

// InputName/OutputNames left empty -> engine auto-detects from the ONNX file.
func (m *classificationModel) InputName() string     { return "" }
func (m *classificationModel) OutputNames() []string { return nil }

func (m *classificationModel) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return preprocess(img, m.cfg)
}

func (m *classificationModel) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return postprocess(outs, meta, m.cfg)
}
