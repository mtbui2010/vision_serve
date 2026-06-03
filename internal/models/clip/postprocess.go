package clip

import (
	"fmt"
	"math"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// postprocess decodes the CLIP image encoder output.
//
// Expected output: a single tensor of shape [1, D] float32 where D=512 for ViT-B/32.
// The vector is L2-normalised before returning so that cosine similarity can be computed
// as a plain dot product.
func postprocess(outs []engine.Tensor) (models.Result, error) {
	if len(outs) == 0 {
		return models.Result{}, fmt.Errorf("clip: expected at least 1 output tensor, got 0")
	}

	// The first (and typically only) output is the embedding.
	t := outs[0]

	// Accept [1, D] or [D] shapes.
	var dim int64
	switch len(t.Shape) {
	case 1:
		dim = t.Shape[0]
	case 2:
		if t.Shape[0] != 1 {
			return models.Result{}, fmt.Errorf("clip: unexpected batch size %d (expected 1)", t.Shape[0])
		}
		dim = t.Shape[1]
	default:
		return models.Result{}, fmt.Errorf("clip: unexpected output shape %v (expected [1,D] or [D])", t.Shape)
	}

	if int64(len(t.Data)) < dim {
		return models.Result{}, fmt.Errorf("clip: output data length %d < dimension %d", len(t.Data), dim)
	}

	raw := t.Data[:dim]

	// L2-normalize: v = v / ||v||
	var sumSq float64
	for _, v := range raw {
		sumSq += float64(v) * float64(v)
	}
	norm := math.Sqrt(sumSq)

	normalized := make([]float32, dim)
	if norm > 0 {
		invNorm := float32(1.0 / norm)
		for i, v := range raw {
			normalized[i] = v * invNorm
		}
	} else {
		copy(normalized, raw)
	}

	return models.Result{
		Task:       models.TaskEmbed,
		Embeddings: [][]float32{normalized},
	}, nil
}
