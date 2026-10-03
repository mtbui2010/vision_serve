// Package clip implements BOTH CLIP towers (OpenAI ViT-B/32, MIT license), which
// produce 512-d L2-normalised embeddings in ONE shared space:
//
//   - architecture "clip"      — the IMAGE encoder. A plain Model (single ONNX
//     session, no prompt): this file + preprocess.go + postprocess.go.
//   - architecture "clip-text" — the TEXT encoder (text.go). A PipelineModel, because
//     its input is a PROMPT, not an image. Its pure-Go BPE tokenizer lives in
//     tokenizer.go — no Python at runtime (CLAUDE.md).
//
// Both register themselves via init(), so no core modification is required. Because the
// spaces match, cosine(image_embedding, text_embedding) is a plain dot product; see
// models/clip-text/README.md for the measured verification.
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
	if _, err := spec(cfg); err != nil {
		return nil, err
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
