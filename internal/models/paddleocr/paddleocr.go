// Package paddleocr implements PP-OCRv4 (Apache-2.0, PaddlePaddle) for VisionServe.
//
// PP-OCRv4 is a two-stage OCR pipeline:
//  1. det (DBNet++): detects text regions → binary probability map.
//  2. rec (SVTR-tiny CTC): recognizes text in each region crop.
//
// This is a PipelineModel (roles "det" and "rec"): it drives both ONNX sessions itself
// via the Runner. lifecycle.Manager loads and owns both sessions (VRAM-safe).
//
// V1 approach (no polygon extraction):
//   - Threshold the DBNet++ probability map at 0.3.
//   - Find connected components via BFS flood-fill → axis-aligned bounding boxes.
//   - Expand each box by a 1.5× dilate factor.
//   - Crop each box, resize to h=48, run SVTR-tiny, CTC-decode → text string.
//
// Result encoding: unified Detection schema (BBox = text region, Class = recognized text,
// Conf = average CTC confidence). No schema changes needed.
//
// Verified I/O contracts:
//
//	det  input  "x"                  [1, 3, H, W] float32 (dynamic H/W, padded to ×32)
//	     output "sigmoid_0.tmp_0"    [1, 1, H, W] float32 (probability map 0..1)
//	rec  input  "x"                  [1, 3, 48, W_crop] float32
//	     output "softmax_11.tmp_0"   [1, T, num_chars] float32 (softmax probabilities)
package paddleocr

import (
	"fmt"
	"image"
	"path/filepath"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("paddle-ocr", New)
}

const (
	roleDet = "det"
	roleRec = "rec"
)

// paddleOCR is the PipelineModel for PP-OCRv4.
type paddleOCR struct {
	cfg     models.Config
	charset []string
}

// New is the factory called by lifecycle after parsing the manifest.
// It validates the manifest and pre-loads the character set.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleDet] == "" {
		return nil, fmt.Errorf("paddleocr: manifest must declare files.%s", roleDet)
	}
	if cfg.Files[roleRec] == "" {
		return nil, fmt.Errorf("paddleocr: manifest must declare files.%s", roleRec)
	}

	// Load the character set from ppocr_keys_v1.txt in the model directory.
	charsetDir := cfg.Dir
	if charsetDir == "" {
		// Fall back to the directory containing the det ONNX file.
		charsetDir = filepath.Dir(cfg.Files[roleDet])
	}
	charset, err := loadCharset(charsetDir)
	if err != nil {
		return nil, err
	}

	return &paddleOCR{cfg: cfg, charset: charset}, nil
}

func (m *paddleOCR) Name() string      { return m.cfg.Name }
func (m *paddleOCR) Task() models.Task { return models.TaskDetection }

// Roles are the ONNX sessions lifecycle must load (keys into the manifest 'files' map).
func (m *paddleOCR) Roles() []string { return []string{roleDet, roleRec} }

// Infer runs the full PP-OCRv4 pipeline:
//  1. det: preprocess image → [1,3,H,W] → run DBNet++ → probability map → extract boxes.
//  2. rec: for each box, crop + resize to [1,3,48,W_crop] → run SVTR-tiny → CTC decode.
//  3. Build Result{Detections} with BBox in original coords, Class = text, Conf = avg CTC conf.
func (m *paddleOCR) Infer(img image.Image, _ models.Prompt, r models.Runner) (models.Result, error) {
	// --- Stage 1: Detection ---
	maxSide := m.cfg.Width
	if maxSide <= 0 {
		maxSide = detMaxSide
	}

	detTensor, detMeta := detPreprocess(img, maxSide)

	// Run detection session.
	detInputName := firstName(r.InputNames(roleDet), "x")
	detOuts, err := r.Run(roleDet, map[string]engine.Tensor{detInputName: detTensor})
	if err != nil {
		return models.Result{}, fmt.Errorf("paddleocr: det session failed: %w", err)
	}
	if len(detOuts) == 0 {
		return models.Result{}, fmt.Errorf("paddleocr: det session returned no output")
	}

	probMapTensor := pickProbMap(r.OutputNames(roleDet), detOuts)
	if probMapTensor == nil {
		return models.Result{}, fmt.Errorf("paddleocr: could not find probability map in det outputs (shapes %v)", shapesOf(detOuts))
	}

	// Probability map shape: [1, 1, H, W] — extract H, W.
	var mapH, mapW int
	switch len(probMapTensor.Shape) {
	case 4:
		mapH = int(probMapTensor.Shape[2])
		mapW = int(probMapTensor.Shape[3])
	case 3:
		mapH = int(probMapTensor.Shape[1])
		mapW = int(probMapTensor.Shape[2])
	case 2:
		mapH = int(probMapTensor.Shape[0])
		mapW = int(probMapTensor.Shape[1])
	default:
		return models.Result{}, fmt.Errorf("paddleocr: unexpected prob map shape %v", probMapTensor.Shape)
	}

	thresh := m.cfg.ConfThresh
	if thresh <= 0 {
		thresh = defaultDetThresh
	}

	detBoxes := extractBBoxes(probMapTensor.Data, mapH, mapW, thresh, defaultDilate)

	if len(detBoxes) == 0 {
		// No text regions found — return empty result.
		return models.Result{Detections: []models.Detection{}}, nil
	}

	// Map det-space boxes back to original image coordinates.
	origBoxes := make([][4]float64, len(detBoxes))
	for i, b := range detBoxes {
		origBoxes[i] = mapBoxToOriginal(b, detMeta)
	}

	// --- Stage 2: Recognition ---
	recInputName := firstName(r.InputNames(roleRec), "x")
	dets := make([]models.Detection, 0, len(origBoxes))

	for _, bbox := range origBoxes {
		// Skip degenerate boxes.
		if bbox[2] < 1 || bbox[3] < 1 {
			continue
		}

		recTensor, _ := recPreprocess(img, bbox)

		recOuts, err := r.Run(roleRec, map[string]engine.Tensor{recInputName: recTensor})
		if err != nil {
			// Skip this crop on error (log-worthy but not fatal).
			continue
		}
		if len(recOuts) == 0 {
			continue
		}

		logitsTensor := pickRecLogits(r.OutputNames(roleRec), recOuts)
		if logitsTensor == nil {
			continue
		}

		// logits shape: [1, T, C].
		var T, C int
		switch len(logitsTensor.Shape) {
		case 3:
			T = int(logitsTensor.Shape[1])
			C = int(logitsTensor.Shape[2])
		case 2:
			T = int(logitsTensor.Shape[0])
			C = int(logitsTensor.Shape[1])
		default:
			continue
		}

		text, conf := ctcDecode(logitsTensor.Data, T, C, m.charset)
		if text == "" {
			// No recognized text — still include box with empty class.
			text = ""
		}

		dets = append(dets, models.Detection{
			BBox:  bbox,
			Class: text,
			Conf:  conf,
		})
	}

	return models.Result{Detections: dets}, nil
}

// firstName returns the first element of names, or fallback if names is empty.
func firstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}

// shapesOf returns the shapes of all tensors (for error messages).
func shapesOf(ts []engine.Tensor) [][]int64 {
	out := make([][]int64, len(ts))
	for i, t := range ts {
		out[i] = t.Shape
	}
	return out
}
