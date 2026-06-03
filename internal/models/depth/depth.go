// Package depth implements monocular depth estimation models:
//   - Depth Anything V2 (Apache-2.0) — 518×518 input
//   - MiDaS (MIT) — 256×256 input
//
// Both are plain Model implementations: engine+lifecycle drive the single ONNX session.
// The model focuses only on pre/postprocess (the part that differs per architecture).
//
// Registered via init() — no changes to core required.
package depth

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("depth-anything-v2", New)
	models.Register("midas", New)
}

type depthModel struct {
	cfg models.Config
}

// New is the factory the lifecycle calls after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("depth: invalid input dimensions (%dx%d)", cfg.Width, cfg.Height)
	}
	return &depthModel{cfg: cfg}, nil
}

func (m *depthModel) Name() string      { return m.cfg.Name }
func (m *depthModel) Task() models.Task { return models.TaskDepth }

// InputName/OutputNames left empty -> engine auto-detects from the ONNX file.
func (m *depthModel) InputName() string     { return "" }
func (m *depthModel) OutputNames() []string { return nil }

func (m *depthModel) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return preprocess(img, m.cfg)
}

func (m *depthModel) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return postprocess(outs, meta, m.cfg)
}
