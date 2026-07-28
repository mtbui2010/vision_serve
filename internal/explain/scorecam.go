package explain

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// scoreCamExplainer reads backbone feature maps from the explain session and
// delegates the actual masked-inference loop to ScoreCAMHeatmap, which the
// handler calls directly (it needs the detect runner).
//
// Because Score-CAM requires re-running the detect session N times (once per
// feature channel), the Heatmap method of this type returns an error directing
// callers to use ScoreCAMHeatmap instead — that function takes an explicit
// runner callback so the handler can supply the lifecycle-owned detect session
// without creating a circular import.
type scoreCamExplainer struct {
	outputName  string // ONNX output node name for backbone feature map
	topChannels int    // max channels to evaluate (top-K by L2 norm)
}

// Heatmap implements Explainer.
// Score-CAM requires running the detect session with masked inputs, which needs
// a runner callback the handler supplies. Use ScoreCAMHeatmap for a full result.
func (e *scoreCamExplainer) Heatmap(
	outputs []engine.Tensor,
	outputNames []string,
	meta models.PreprocessMeta,
	detectionIdx int,
	origW, origH int,
) ([]float32, int, int, error) {
	feat, err := findOutput(outputs, outputNames, e.outputName)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(feat.Shape) != 4 {
		return nil, 0, 0, fmt.Errorf(
			"score_cam: expected 4D tensor [1,C,H,W], got %v", feat.Shape)
	}
	// Delegate to the full implementation below, without a score runner.
	// topChannels only — structural saliency, no score reweighting.
	return scoreCAMStructural(feat, e.topChannels, origW, origH)
}

// scoreCAMStructural computes a simplified Score-CAM heatmap using only the
// feature activation magnitudes (no re-inference, no score reweighting).
// It is equivalent to Score-CAM with every channel weight = 1.  The handler
// uses ScoreCAMHeatmap for the full version when it has a detect runner.
func scoreCAMStructural(feat engine.Tensor, topChannels, origW, origH int) ([]float32, int, int, error) {
	numC := int(feat.Shape[1])
	fH   := int(feat.Shape[2])
	fW   := int(feat.Shape[3])

	k := topChannels
	if k <= 0 || k > numC {
		k = numC
	}

	// Rank channels by L2 norm of their activation map.
	type chanEntry struct {
		idx  int
		norm float32
	}
	entries := make([]chanEntry, numC)
	for c := 0; c < numC; c++ {
		base := c * fH * fW
		var sum float32
		for i := 0; i < fH*fW; i++ {
			v := feat.Data[base+i]
			sum += v * v
		}
		entries[c] = chanEntry{c, float32(math.Sqrt(float64(sum)))}
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].norm > entries[j].norm
	})

	heatmap := make([]float32, origW*origH)
	for i := 0; i < k; i++ {
		c    := entries[i].idx
		base := c * fH * fW

		ch := make([]float32, fH*fW)
		copy(ch, feat.Data[base:base+fH*fW])
		ch = Normalize(ch)

		up := upsampleNearest(ch, fW, fH, origW, origH)
		for j := range heatmap {
			heatmap[j] += up[j]
		}
	}
	return Normalize(heatmap), origW, origH, nil
}

// ScoreCAMHeatmap is the full Score-CAM algorithm: for each of the top-K backbone
// feature channels, it upsample the channel activation to image size, applies it as
// a mask to the input image, runs the detect session, and uses the resulting
// detection score to weight the channel's contribution to the final heatmap.
//
// This is called directly by the handler (not through the Explainer interface) because
// it needs the detect runner callback — a function that runs the detect ONNX session
// on a masked image and returns the confidence score of the target detection.
//
// Parameters:
//
//	featTensor    backbone feature tensor [1, C, fH, fW] from the explain session.
//	origImg       the original (unmasked) input image.
//	detectRunner  runs the detect session on a masked image, returns score of detectionIdx.
//	topChannels   number of channels to evaluate (0 = all channels).
//	origW, origH  dimensions of the original image.
func ScoreCAMHeatmap(
	featTensor engine.Tensor,
	origImg image.Image,
	detectRunner func(masked image.Image) (float32, error),
	topChannels int,
	origW, origH int,
) (heatmap []float32, W, H int, err error) {
	if len(featTensor.Shape) != 4 {
		return nil, 0, 0, fmt.Errorf("score_cam: expected 4D tensor [1,C,H,W], got %v", featTensor.Shape)
	}
	numC := int(featTensor.Shape[1])
	fH   := int(featTensor.Shape[2])
	fW   := int(featTensor.Shape[3])

	k := topChannels
	if k <= 0 || k > numC {
		k = numC
	}

	// Rank channels by L2 norm.
	type chanEntry struct {
		idx  int
		norm float32
	}
	entries := make([]chanEntry, numC)
	for c := 0; c < numC; c++ {
		base := c * fH * fW
		var sum float32
		for i := 0; i < fH*fW; i++ {
			v := featTensor.Data[base+i]
			sum += v * v
		}
		entries[c] = chanEntry{c, float32(math.Sqrt(float64(sum)))}
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].norm > entries[j].norm
	})

	result := make([]float32, origW*origH)
	for i := 0; i < k; i++ {
		c    := entries[i].idx
		base := c * fH * fW

		ch := make([]float32, fH*fW)
		copy(ch, featTensor.Data[base:base+fH*fW])
		ch = Normalize(ch)

		// Upsample channel activation to original image size.
		mask := upsampleNearest(ch, fW, fH, origW, origH)

		// Apply mask to original image.
		masked := applyMask(origImg, mask, origW, origH)

		// Run detect session, get score for the target detection.
		score, rerr := detectRunner(masked)
		if rerr != nil {
			continue // skip channels where inference fails
		}
		if score < 0 {
			score = 0
		}

		// Accumulate score-weighted activation.
		for j := range result {
			result[j] += score * mask[j]
		}
	}

	return Normalize(result), origW, origH, nil
}

// applyMask multiplies each pixel of orig by the corresponding mask value [0,1].
// The output image has the same bounds as the mask (origW×origH, 0-origin).
func applyMask(orig image.Image, mask []float32, origW, origH int) image.Image {
	out := image.NewRGBA(image.Rect(0, 0, origW, origH))
	if orig == nil {
		return out
	}
	bounds := orig.Bounds()
	for y := 0; y < origH; y++ {
		for x := 0; x < origW; x++ {
			m := mask[y*origW+x]
			// orig may have non-zero Min (e.g. from a crop).
			r32, g32, b32, a32 := orig.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			out.SetRGBA(x, y, color.RGBA{
				R: uint8(float32(r32>>8) * m),
				G: uint8(float32(g32>>8) * m),
				B: uint8(float32(b32>>8) * m),
				A: uint8(a32 >> 8),
			})
		}
	}
	return out
}
