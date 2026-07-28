package explain

import (
	"fmt"
	"math"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// attentionExplainer reads cross-attention weights from transformer decoder
// outputs and produces a spatial saliency map for a given detection query.
//
// Expected tensor shape: [num_layers, batch=1, num_heads, num_queries, spatial_tokens]
// The explainer averages over all layers and heads, then picks the query at
// detectionIdx.  The flat spatial vector is reshaped to a near-square grid and
// upsampled to the original image dimensions.
type attentionExplainer struct {
	outputName    string // ONNX output node name for the attention weight tensor
	spatialStride int    // backbone downsample factor (e.g. 32 for a ViT-32 backbone)
}

// Heatmap implements Explainer.
func (e *attentionExplainer) Heatmap(
	outputs []engine.Tensor,
	outputNames []string,
	meta models.PreprocessMeta,
	detectionIdx int,
	origW, origH int,
) ([]float32, int, int, error) {
	attn, err := findOutput(outputs, outputNames, e.outputName)
	if err != nil {
		return nil, 0, 0, err
	}

	// Expected shape: [num_layers, batch, num_heads, num_queries, spatial_tokens]
	if len(attn.Shape) != 5 {
		return nil, 0, 0, fmt.Errorf(
			"attention: expected 5D tensor [L,B,H,Q,S], got shape %v", attn.Shape)
	}

	numLayers  := int(attn.Shape[0])
	// batch    := int(attn.Shape[1])  // always 1
	numHeads   := int(attn.Shape[2])
	numQueries := int(attn.Shape[3])
	numSpatial := int(attn.Shape[4])

	if detectionIdx < 0 || detectionIdx >= numQueries {
		return nil, 0, 0, fmt.Errorf(
			"attention: detectionIdx %d out of range [0, %d)", detectionIdx, numQueries)
	}

	// Average over all layers and heads → [numSpatial].
	// Tensor layout (row-major): [L, B, H, Q, S] with B=1.
	// Element index: l*(numHeads*numQueries*numSpatial) + h*(numQueries*numSpatial) + q*numSpatial + s
	spatialMap := make([]float32, numSpatial)
	stride_LBH := numHeads * numQueries * numSpatial // stride per (layer, batch=0 implied)
	stride_H   := numQueries * numSpatial
	for l := 0; l < numLayers; l++ {
		for h := 0; h < numHeads; h++ {
			base := l*stride_LBH + h*stride_H + detectionIdx*numSpatial
			for s := 0; s < numSpatial; s++ {
				spatialMap[s] += attn.Data[base+s]
			}
		}
	}
	scale := float32(numLayers * numHeads)
	for i := range spatialMap {
		spatialMap[i] /= scale
	}

	// Reshape flat spatial vector to a 2D grid.
	// Most detection models use square feature maps (e.g. 20×20 for 640 with stride-32).
	// Heuristic: integer sqrt; if not exact, fall back to a row of width=numSpatial.
	sW := int(math.Round(math.Sqrt(float64(numSpatial))))
	sH := numSpatial / sW
	if sW*sH != numSpatial {
		// Non-square: treat as a 1×N strip.  The upsample handles non-square inputs.
		sW = numSpatial
		sH = 1
	}

	spatialMap = Normalize(spatialMap)
	heatmap := upsampleNearest(spatialMap, sW, sH, origW, origH)
	return heatmap, origW, origH, nil
}
