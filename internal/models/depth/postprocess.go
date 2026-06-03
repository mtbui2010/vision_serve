package depth

import (
	"fmt"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// postprocess decodes a depth estimation output tensor -> normalized Result.
//
// ===================== OUTPUT FORMAT =====================
// Both Depth Anything V2 and MiDaS emit a single output tensor of shape:
//   [1, H, W]   — inverse-depth (disparity) values, unnormalized float32
//
// We min-max normalize to [0, 1] so the values are comparable across inputs and
// models. The caller receives a row-major flat slice of length H*W.
// =========================================================
func postprocess(outs []engine.Tensor, _ models.PreprocessMeta, cfg models.Config) (models.Result, error) {
	if len(outs) == 0 {
		return models.Result{}, fmt.Errorf("depth: no output tensors returned by the model")
	}

	// Accept either [1, H, W] (3-dim) or [H, W] (2-dim) — some ONNX exports drop the batch dim.
	out := outs[0]
	var h, w int
	switch len(out.Shape) {
	case 3:
		if out.Shape[0] != 1 {
			return models.Result{}, fmt.Errorf("depth: expected batch size 1, got shape %v", out.Shape)
		}
		h, w = int(out.Shape[1]), int(out.Shape[2])
	case 2:
		h, w = int(out.Shape[0]), int(out.Shape[1])
	default:
		return models.Result{}, fmt.Errorf("depth: unexpected output tensor shape %v (expected [1,H,W] or [H,W])", out.Shape)
	}

	n := h * w
	if len(out.Data) != n {
		return models.Result{}, fmt.Errorf("depth: data length %d does not match shape %dx%d=%d", len(out.Data), h, w, n)
	}

	// Min-max normalize to [0, 1].
	normalized := minMaxNormalize(out.Data)

	return models.Result{
		Task:        models.TaskDepth,
		DepthMap:    normalized,
		DepthWidth:  w,
		DepthHeight: h,
	}, nil
}

// minMaxNormalize rescales values to [0, 1]. Returns the input slice unchanged
// (all zeros) when the range is zero.
func minMaxNormalize(data []float32) []float32 {
	if len(data) == 0 {
		return data
	}

	mn, mx := data[0], data[0]
	for _, v := range data[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}

	rng := mx - mn
	out := make([]float32, len(data))
	if rng == 0 {
		// Flat depth map — return all zeros.
		return out
	}
	for i, v := range data {
		out[i] = (v - mn) / rng
	}
	return out
}
