package groundingdino

import (
	"path/filepath"

	"visionserve/internal/models"
)

// Built-in thresholds, used when neither the request nor the manifest sets one.
const (
	DefaultBoxThresh  = 0.3
	DefaultTextThresh = 0.25
)

// Thresholds resolves GroundingDINO's box and text thresholds for one request, in the one
// precedence every GroundingDINO pipeline uses: the request's override (> 0), else the manifest
// value (> 0), else the built-in default.
//
// A caller whose manifest thresholds belong to ANOTHER detector passes 0 for them — the
// rfdetr-gdino router does: its conf_threshold is RF-DETR's, on a different scale, so its
// GroundingDINO pass resolves request → default only.
func Thresholds(manifestBox, manifestText float64, p models.Prompt) (box, text float64) {
	box, text = manifestBox, manifestText
	if box <= 0 {
		box = DefaultBoxThresh
	}
	if text <= 0 {
		text = DefaultTextThresh
	}
	if p.BoxThresh > 0 {
		box = p.BoxThresh
	}
	if p.TextThresh > 0 {
		text = p.TextThresh
	}
	return box, text
}

// VocabPath is where the tokenizer's vocab.txt lives for GroundingDINO weights at weights: next
// to them (a composite's manifest points at "../grounding-dino/model-fixedmask.onnx"), or
// <dir>/vocab.txt when the manifest declares no weights path.
func VocabPath(weights, dir string) string {
	if weights != "" {
		return filepath.Join(filepath.Dir(weights), "vocab.txt")
	}
	return filepath.Join(dir, "vocab.txt")
}
