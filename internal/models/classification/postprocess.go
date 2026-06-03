package classification

import (
	"fmt"
	"math"
	"sort"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// postprocess decodes classification output logits -> top-K predictions.
//
// ===================== OUTPUT FORMAT =====================
// EfficientNet-B0 and MobileNet-V3 emit a single output tensor of shape:
//   [1, num_classes]   — raw logits (before softmax), float32
//
// We apply softmax to obtain probabilities, then select the top-K entries.
// K = cfg.MaxDet if > 0, otherwise defaults to 5.
// =========================================================
func postprocess(outs []engine.Tensor, _ models.PreprocessMeta, cfg models.Config) (models.Result, error) {
	if len(outs) == 0 {
		return models.Result{}, fmt.Errorf("classification: no output tensors returned by the model")
	}

	out := outs[0]

	// Accept [1, C] or [C] — some ONNX exports drop the batch dimension.
	var numClasses int
	switch len(out.Shape) {
	case 2:
		if out.Shape[0] != 1 {
			return models.Result{}, fmt.Errorf("classification: expected batch size 1, got shape %v", out.Shape)
		}
		numClasses = int(out.Shape[1])
	case 1:
		numClasses = int(out.Shape[0])
	default:
		return models.Result{}, fmt.Errorf("classification: unexpected output tensor shape %v (expected [1,C] or [C])", out.Shape)
	}

	if len(out.Data) != numClasses {
		return models.Result{}, fmt.Errorf("classification: data length %d does not match num_classes %d", len(out.Data), numClasses)
	}

	probs := softmax(out.Data)

	// Build index slice and sort by probability descending.
	indices := make([]int, numClasses)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(a, b int) bool {
		return probs[indices[a]] > probs[indices[b]]
	})

	k := cfg.MaxDet
	if k <= 0 {
		k = 5
	}
	if k > numClasses {
		k = numClasses
	}

	topK := make([]models.Classification, k)
	for i := 0; i < k; i++ {
		idx := indices[i]
		topK[i] = models.Classification{
			Class: classLabel(cfg, idx),
			Conf:  float64(probs[idx]),
		}
	}

	return models.Result{
		Task:            models.TaskClassification,
		Classifications: topK,
	}, nil
}

// softmax converts logits to probabilities (numerically stable via max subtraction).
func softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}

	// Find max for numerical stability.
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	out := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v) - float64(maxVal))
		out[i] = float32(e)
		sum += e
	}
	if sum > 0 {
		inv := float32(1.0 / sum)
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

func classLabel(cfg models.Config, idx int) string {
	if idx >= 0 && idx < len(cfg.Labels) {
		return cfg.Labels[idx]
	}
	return fmt.Sprintf("class_%d", idx)
}
