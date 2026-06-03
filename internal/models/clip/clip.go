// Package clip implements the CLIP image encoder (OpenAI, MIT license).
//
// V1: image encoder only — produces 512-d L2-normalised embeddings (ViT-B/32).
// Text encoder is deferred: it requires a BPE tokenizer which is too complex for v1.
//
// The model is a plain Model (single ONNX session, no prompt). Registered under the
// architecture name "clip" via init() — no core modifications required.
package clip

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("clip", New)
}

type clipModel struct {
	cfg models.Config
}

// New is the factory called by lifecycle after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("clip: invalid input width/height (%dx%d)", cfg.Width, cfg.Height)
	}
	return &clipModel{cfg: cfg}, nil
}

func (m *clipModel) Name() string      { return m.cfg.Name }
func (m *clipModel) Task() models.Task { return models.TaskEmbed }

// InputName/OutputNames left empty -> engine auto-detects from the ONNX file.
func (m *clipModel) InputName() string     { return "" }
func (m *clipModel) OutputNames() []string { return nil }

func (m *clipModel) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return preprocess(img, m.cfg)
}

func (m *clipModel) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	return postprocess(outs)
}
