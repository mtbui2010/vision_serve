// Package groundedsam implements Grounded-SAM (Apache-2.0): GroundingDINO (open-vocab,
// text-prompted detection) followed by MobileSAM (box-prompted segmentation). It is a
// fully free community pipeline — no AGPL, no Python at runtime (see CLAUDE.md).
//
// It composes the two existing models without duplicating their logic: it calls
// groundingdino.Detect for boxes/labels, then mobilesam.Segment for one mask per box. The
// heavy ONNX sessions (gdino, encoder, decoder) are owned by lifecycle.Manager; this model
// only orchestrates them via the Runner.
package groundedsam

import (
	"fmt"
	"image"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/groundingdino"
	"visionserve/internal/models/mobilesam"
	"visionserve/internal/vision/util"
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
	cfg   models.Config
	tok   *groundingdino.Tokenizer
	joint bool // gdino weights rebuild the text mask for any phrase count -> score in one pass
}

// New loads the GroundingDINO tokenizer once. The vocab lives next to the GroundingDINO
// weights; the manifest references those via a relative path (files.gdino), so we resolve
// vocab.txt from that file's directory and fall back to <cfg.Dir>/vocab.txt.
func New(cfg models.Config) (models.Base, error) {
	for _, role := range []string{roleGDINO, roleEncoder, roleDecoder} {
		if cfg.Files[role] == "" {
			return nil, fmt.Errorf("grounded-sam: manifest must declare files.%s", role)
		}
	}
	tok, err := groundingdino.LoadTokenizer(groundingdino.VocabPath(cfg.Files[roleGDINO], cfg.Dir))
	if err != nil {
		return nil, err
	}
	return &groundedSAM{
		cfg:   cfg,
		tok:   tok,
		joint: groundingdino.JointTextPassOrSafe(cfg.Files[roleGDINO]),
	}, nil
}

func (m *groundedSAM) Name() string      { return m.cfg.Name }
func (m *groundedSAM) Task() models.Task { return models.TaskOpenVocab }

// Roles: GroundingDINO + the two MobileSAM sessions.
func (m *groundedSAM) Roles() []string { return []string{roleGDINO, roleEncoder, roleDecoder} }

// PoolSizes requests 4 decoder copies so multiple detected boxes can be
// segmented concurrently rather than sequentially.
func (m *groundedSAM) PoolSizes() map[string]int { return map[string]int{roleDecoder: 4} }

// Infer runs detection then per-box segmentation; masks are index-aligned with detections.
func (m *groundedSAM) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	// Serialize the whole GroundingDINO pipeline (see groundingdino.PipelineMu).
	groundingdino.PipelineMu.Lock()
	defer groundingdino.PipelineMu.Unlock()
	if strings.TrimSpace(prompt.Text) == "" {
		return models.Result{}, fmt.Errorf("grounded-sam requires a text prompt, e.g. --prompt \"cat. remote.\"")
	}
	boxThresh, textThresh := groundingdino.Thresholds(m.cfg.ConfThresh, m.cfg.TextThresh, prompt)

	// 1) Open-vocab detection (GroundingDINO).
	gdinoRun := func(inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleGDINO, inputs)
	}
	dets, err := groundingdino.Detect(img, prompt.Text, m.tok, gdinoRun, r.OutputNames(roleGDINO), boxThresh, textThresh,
		groundingdino.WithJointTextPass(m.joint))
	if err != nil {
		return models.Result{}, err
	}
	if len(dets) == 0 {
		return models.Result{Detections: dets}, nil // nothing to segment
	}

	// 2) Segment one mask per detected box (MobileSAM).
	boxes := make([][4]float64, len(dets))
	for i, d := range dets {
		boxes[i] = d.BBox
	}
	encRun := func(inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleEncoder, inputs)
	}
	decRun := func(inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleDecoder, inputs)
	}
	encInName := util.FirstName(r.InputNames(roleEncoder), "input_image")
	masks, err := mobilesam.Segment(img, boxes, encRun, decRun, encInName, r.OutputNames(roleDecoder))
	if err != nil {
		return models.Result{}, err
	}

	// Keep masks index-aligned with detections; surface the detection box + score on each
	// mask so a consumer can pair them even without the detections slice.
	for i := range masks {
		if i < len(dets) {
			masks[i].BBox = dets[i].BBox
			masks[i].Conf = dets[i].Conf
		}
	}

	return models.Result{Detections: dets, Masks: masks}, nil
}
