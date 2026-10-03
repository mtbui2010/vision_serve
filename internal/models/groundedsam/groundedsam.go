// Package groundedsam implements Grounded-SAM (Apache-2.0): GroundingDINO (open-vocab,
// text-prompted detection) followed by MobileSAM (box-prompted segmentation). It is a
// fully free community pipeline — no AGPL, no Python at runtime (see CLAUDE.md).
//
// It is a configuration of two pipeline stages — pipeline.GDINO for boxes/labels, pipeline.SAM
// for one mask per box — composed by pipeline.Grounded; it duplicates neither model's logic. The
// heavy ONNX sessions (gdino, encoder, decoder) are owned by lifecycle.Manager; this model only
// orchestrates them via the Runner.
package groundedsam

import (
	"fmt"
	"image"
	"strings"

	"visionserve/internal/models"
	"visionserve/internal/pipeline"
)

func init() {
	models.Register("grounded-sam", New)
}

const (
	roleGDINO   = "gdino"
	roleEncoder = "encoder"
	roleDecoder = "decoder"
)

type groundedSAM struct {
	cfg models.Config
	gs  pipeline.Grounded
}

// New loads the GroundingDINO tokenizer once. The vocab lives next to the GroundingDINO weights;
// the manifest references those via a relative path (files.gdino), so vocab.txt is resolved from
// that file's directory (groundingdino.VocabPath). The manifest's conf_threshold/text_threshold
// are GroundingDINO's box/text thresholds.
func New(cfg models.Config) (models.Base, error) {
	for _, role := range []string{roleGDINO, roleEncoder, roleDecoder} {
		if cfg.Files[role] == "" {
			return nil, fmt.Errorf("grounded-sam: manifest must declare files.%s", role)
		}
	}
	det, err := pipeline.NewGDINO(roleGDINO, cfg.Files[roleGDINO], cfg.Dir, cfg.ConfThresh, cfg.TextThresh)
	if err != nil {
		return nil, err
	}
	return &groundedSAM{
		cfg: cfg,
		gs:  pipeline.Grounded{Detector: det, Segmenter: pipeline.SAM{Encoder: roleEncoder, Decoder: roleDecoder}},
	}, nil
}

func (m *groundedSAM) Name() string      { return m.cfg.Name }
func (m *groundedSAM) Task() models.Task { return models.TaskOpenVocab }

// Roles: GroundingDINO + the two MobileSAM sessions.
func (m *groundedSAM) Roles() []string { return []string{roleGDINO, roleEncoder, roleDecoder} }

// PoolSizes requests 4 decoder copies so multiple detected boxes can be
// segmented concurrently rather than sequentially.
func (m *groundedSAM) PoolSizes() map[string]int { return map[string]int{roleDecoder: 4} }

// Exclusive implements models.Exclusive: lifecycle runs one whole GroundingDINO → MobileSAM
// pipeline at a time on this loaded model; other models are not held up by it (it replaced the
// process-wide groundingdino.PipelineMu — see the groundingdino package doc).
func (m *groundedSAM) Exclusive() bool { return true }

// Infer runs detection then per-box segmentation; masks are index-aligned with detections.
func (m *groundedSAM) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	if strings.TrimSpace(prompt.Text) == "" {
		return models.Result{}, fmt.Errorf("grounded-sam requires a text prompt, e.g. --prompt \"cat. remote.\"")
	}
	words, err := pipeline.TextPhrases(prompt.Text)
	if err != nil {
		return models.Result{}, err
	}
	return m.gs.Infer(pipeline.Call{Img: img, Prompt: prompt, Runner: r}, words)
}
