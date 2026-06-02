// Package groundingdino implements GroundingDINO (Apache-2.0) for text-prompt-driven
// open-vocabulary detection — a fully free community feature (no AGPL, no Python at
// runtime; see CLAUDE.md).
//
// GroundingDINO is a single ONNX graph but it is PROMPTED (it needs the text), so it is a
// PipelineModel (role "model") that drives its own inference via the Runner.
//
// Verified I/O contract (real runs):
//
//	inputs:  pixel_values   [1,3,800,800] float32 (ImageNet-normalized, plain 800x800 squash)
//	         pixel_mask     [1,800,800]   int64   (all ones)
//	         input_ids      [1,L]         int64   (BERT WordPiece, [CLS]..[SEP])
//	         attention_mask [1,L]         int64   (all ones)
//	         token_type_ids [1,L]         int64   (all zeros)
//	outputs: logits     [1,900,256] float32
//	         pred_boxes [1,900,4]   float32 (cxcywh, normalized 0..1)
//
// Postprocess mirrors HF post_process_grounded_object_detection: sigmoid(logits), per
// query score = max over the 256 text dims; box_threshold filters queries; the phrase is
// decoded from the token positions whose prob exceeds text_threshold.
package groundingdino

import (
	"fmt"
	"image"
	"math"
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/pkg/api"
)

func init() {
	models.Register("grounding-dino", New)
}

const roleModel = "model"

// Default thresholds when the manifest leaves them unset.
const (
	defaultBoxThresh  = 0.3
	defaultTextThresh = 0.25
)

type groundingDINO struct {
	cfg models.Config
	tok *Tokenizer
}

// New loads the tokenizer (vocab.txt in the model directory) once and returns the model.
func New(cfg models.Config) (models.Base, error) {
	vocabPath := filepath.Join(cfg.Dir, "vocab.txt")
	tok, err := LoadTokenizer(vocabPath)
	if err != nil {
		return nil, err
	}
	return &groundingDINO{cfg: cfg, tok: tok}, nil
}

func (m *groundingDINO) Name() string      { return m.cfg.Name }
func (m *groundingDINO) Task() models.Task { return models.TaskOpenVocab }

// Roles: a single session keyed "model".
func (m *groundingDINO) Roles() []string { return []string{roleModel} }

// Infer runs the full open-vocab detection pipeline for the text prompt.
func (m *groundingDINO) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	if strings.TrimSpace(prompt.Text) == "" {
		return models.Result{}, fmt.Errorf("grounding-dino requires a text prompt, e.g. --prompt \"cat. remote.\"")
	}
	boxThresh, textThresh := m.thresholds()

	run := func(inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleModel, inputs)
	}
	dets, err := Detect(img, prompt.Text, m.tok, run, r.OutputNames(roleModel), boxThresh, textThresh)
	if err != nil {
		return models.Result{}, err
	}
	return models.Result{Detections: dets}, nil
}

func (m *groundingDINO) thresholds() (box, text float64) {
	box, text = m.cfg.ConfThresh, m.cfg.TextThresh
	if box <= 0 {
		box = defaultBoxThresh
	}
	if text <= 0 {
		text = defaultTextThresh
	}
	return box, text
}

// Detect is the REUSABLE core (also called by grounded-sam). It builds the 5 inputs from
// the image + text, runs the session via run, and postprocesses logits/pred_boxes into
// detections in ORIGINAL image coordinates.
//
//   - tok:      a tokenizer loaded from the matching vocab.txt.
//   - run:      executes the GroundingDINO session, binding inputs by name.
//   - outNames: the session's output names (used to map logits vs pred_boxes; falls back
//     to shape matching when names are unhelpful).
func Detect(
	img image.Image,
	text string,
	tok *Tokenizer,
	run func(map[string]engine.Tensor) ([]engine.Tensor, error),
	outNames []string,
	boxThresh, textThresh float64,
) ([]api.Detection, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("grounding-dino requires a non-empty text prompt")
	}
	if tok == nil {
		return nil, fmt.Errorf("grounding-dino: nil tokenizer")
	}

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	pixelValues, pixelMask := preprocessImage(img)
	enc := tok.Encode(text)
	L := int64(len(enc.InputIDs))

	inputs := map[string]engine.Tensor{
		"pixel_values":   pixelValues,
		"pixel_mask":     pixelMask,
		"input_ids":      engine.I64(enc.InputIDs, 1, L),
		"attention_mask": engine.I64(enc.AttentionMask, 1, L),
		"token_type_ids": engine.I64(enc.TokenTypeIDs, 1, L),
	}

	outs, err := run(inputs)
	if err != nil {
		return nil, fmt.Errorf("grounding-dino: inference failed: %w", err)
	}

	logits, boxes := pickLogitsAndBoxes(outNames, outs)
	if logits == nil || boxes == nil {
		return nil, fmt.Errorf("grounding-dino: could not identify logits/pred_boxes among outputs (shapes %v)", shapesOf(outs))
	}

	return postprocess(logits, boxes, enc.InputIDs, tok, origW, origH, boxThresh, textThresh)
}

// pickLogitsAndBoxes maps outputs to logits ([.,.,256]) and pred_boxes ([.,.,4]) by name
// first, then by last-dim shape.
func pickLogitsAndBoxes(names []string, outs []engine.Tensor) (logits, boxes *engine.Tensor) {
	for i := range outs {
		name := ""
		if i < len(names) {
			name = strings.ToLower(names[i])
		}
		switch {
		case strings.Contains(name, "box"):
			boxes = &outs[i]
		case strings.Contains(name, "logit"):
			logits = &outs[i]
		}
	}
	if logits == nil || boxes == nil {
		for i := range outs {
			switch outs[i].Dim(-1) {
			case 4:
				if boxes == nil {
					boxes = &outs[i]
				}
			case 256:
				if logits == nil {
					logits = &outs[i]
				}
			}
		}
	}
	return logits, boxes
}

// postprocess turns logits/pred_boxes into detections (original-image xywh).
func postprocess(
	logits, boxes *engine.Tensor,
	inputIDs []int64,
	tok *Tokenizer,
	origW, origH int,
	boxThresh, textThresh float64,
) ([]api.Detection, error) {
	nq := int(logits.Dim(1))   // 900 queries
	dim := int(logits.Dim(2))  // 256 text dims
	bDim := int(boxes.Dim(-1)) // 4
	if nq <= 0 || dim <= 0 || bDim != 4 {
		return nil, fmt.Errorf("grounding-dino: unexpected output shapes logits=%v boxes=%v", logits.Shape, boxes.Shape)
	}

	// Token positions eligible for a phrase: skip [CLS] at 0 and [SEP] at the end.
	lastTok := len(inputIDs) - 1

	var dets []api.Detection
	for q := 0; q < nq; q++ {
		base := q * dim
		// score = max sigmoid over the text dims; also collect token positions above
		// text_threshold for the phrase.
		score := 0.0
		var phraseIDs []int64
		for i := 0; i < dim; i++ {
			p := sigmoid(logits.Data[base+i])
			if p > score {
				score = p
			}
			// Only token positions in [1 .. L-2] correspond to real prompt tokens.
			if i >= 1 && i < lastTok && p > textThresh {
				phraseIDs = append(phraseIDs, inputIDs[i])
			}
		}
		if score <= boxThresh {
			continue
		}
		phrase := tok.Decode(phraseIDs)
		if phrase == "" {
			continue // no assignable phrase → drop (matches HF behavior)
		}

		// cxcywh normalized → xywh in ORIGINAL pixels (plain squash, so multiply by W/H).
		bb := boxes.Data[q*4 : q*4+4]
		cx, cy, w, h := float64(bb[0]), float64(bb[1]), float64(bb[2]), float64(bb[3])
		x := (cx - w/2) * float64(origW)
		y := (cy - h/2) * float64(origH)
		dets = append(dets, api.Detection{
			BBox:  [4]float64{x, y, w * float64(origW), h * float64(origH)},
			Class: phrase,
			Conf:  score,
		})
	}
	return dets, nil
}

func sigmoid(x float32) float64 {
	return 1.0 / (1.0 + math.Exp(-float64(x)))
}

func shapesOf(ts []engine.Tensor) [][]int64 {
	out := make([][]int64, len(ts))
	for i, t := range ts {
		out[i] = t.Shape
	}
	return out
}
