// Package explain computes saliency heatmaps for VisionServe models.
//
// Two strategies are supported:
//   - AttentionExplainer: reads cross-attention weights exported from a transformer
//     decoder. One inference call, fast.
//   - ScoreCAMExplainer: samples backbone feature channels and re-runs detection
//     with each masked input. Many inference calls, slow but model-agnostic.
//
// Usage (from a handler):
//
//	exp, err := explain.New(manifest.Explain)
//	heatmap, W, H, err := exp.Heatmap(outputs, outputNames, meta, detIdx, origW, origH)
//	if err := explain.RenderPNG(w, img, heatmap, W, H, 0.5); err != nil { ... }
package explain

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"math"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/registry"
)

// Explainer computes a spatial heatmap from the raw ONNX outputs of an explain session.
//
// The handler orchestrates:
//  1. model.Preprocess(img) → inputTensor, meta
//  2. explainEng.Run([]engine.Tensor{inputTensor}) → outputs, outputNames
//  3. Explainer.Heatmap(outputs, outputNames, meta, detectionIdx, origW, origH)
//
// This keeps the heavy engine sessions in lifecycle.Manager while the decode
// logic lives here.
type Explainer interface {
	// Heatmap extracts and upsample a spatial attention/activation map.
	//
	// outputs: the result of explainEng.Run(), in ONNX output-name order.
	// outputNames: []string returned by explainEng.OutputNames().
	// meta: preprocess metadata (scale/pad) from model.Preprocess.
	// detectionIdx: 0-based index of the detection query to explain.
	// origW, origH: dimensions of the original image (heatmap is in this space).
	//
	// Returns heatmap []float32 of length W*H with values in [0,1], row-major.
	Heatmap(
		outputs []engine.Tensor,
		outputNames []string,
		meta models.PreprocessMeta,
		detectionIdx int,
		origW, origH int,
	) (heatmap []float32, W, H int, err error)
}

// New builds an Explainer from the manifest's ExplainConfig.
func New(cfg *registry.ExplainConfig) (Explainer, error) {
	if cfg == nil {
		return nil, fmt.Errorf("explain: nil ExplainConfig — model does not support /api/explain")
	}
	switch cfg.Type {
	case "attention":
		attnOutput, ok := cfg.Outputs["attention"]
		if !ok {
			return nil, fmt.Errorf("explain: attention type requires outputs.attention in manifest")
		}
		return &attentionExplainer{
			outputName:    attnOutput,
			spatialStride: cfg.EffectiveSpatialStride(),
		}, nil
	case "score_cam":
		featOutput, ok := cfg.Outputs["features"]
		if !ok {
			return nil, fmt.Errorf("explain: score_cam type requires outputs.features in manifest")
		}
		return &scoreCamExplainer{
			outputName:  featOutput,
			topChannels: cfg.EffectiveTopChannels(),
		}, nil
	default:
		return nil, fmt.Errorf("explain: unknown type %q (want attention or score_cam)", cfg.Type)
	}
}

// RenderPNG overlays the heatmap (hW×hH float32 [0,1], row-major) onto orig using
// the JET colormap and writes a PNG to w. alpha in [0,1] controls overlay opacity.
func RenderPNG(w io.Writer, orig image.Image, heatmap []float32, hW, hH int, alpha float32) error {
	bounds := orig.Bounds()
	out := image.NewRGBA(bounds)
	draw.Draw(out, bounds, orig, bounds.Min, draw.Src)

	origW := bounds.Dx()
	origH := bounds.Dy()

	scaleX := float64(hW) / float64(origW)
	scaleY := float64(hH) / float64(origH)

	for py := bounds.Min.Y; py < bounds.Max.Y; py++ {
		for px := bounds.Min.X; px < bounds.Max.X; px++ {
			hx := int(float64(px-bounds.Min.X) * scaleX)
			hy := int(float64(py-bounds.Min.Y) * scaleY)
			if hx >= hW {
				hx = hW - 1
			}
			if hy >= hH {
				hy = hH - 1
			}
			v := heatmap[hy*hW+hx]
			jr, jg, jb := jetColormap(v)
			orig := out.RGBAAt(px, py)
			out.SetRGBA(px, py, color.RGBA{
				R: clampUint8(float32(orig.R)*(1-alpha) + float32(jr)*alpha),
				G: clampUint8(float32(orig.G)*(1-alpha) + float32(jg)*alpha),
				B: clampUint8(float32(orig.B)*(1-alpha) + float32(jb)*alpha),
				A: 255,
			})
		}
	}
	return png.Encode(w, out)
}

// jetColormap maps v ∈ [0,1] to JET colormap RGB [0,255].
// JET: blue → cyan → green → yellow → red as v increases.
func jetColormap(v float32) (r, g, b uint8) {
	x := math.Max(0, math.Min(1, float64(v)))
	var rf, gf, bf float64
	switch {
	case x < 0.125:
		rf = 0
		gf = 0
		bf = 0.5 + x*4
	case x < 0.375:
		rf = 0
		gf = (x - 0.125) * 4
		bf = 1
	case x < 0.625:
		rf = (x - 0.375) * 4
		gf = 1
		bf = 1 - (x-0.375)*4
	case x < 0.875:
		rf = 1
		gf = 1 - (x-0.625)*4
		bf = 0
	default:
		rf = 1 - (x-0.875)*4
		gf = 0
		bf = 0
	}
	return uint8(rf * 255), uint8(gf * 255), uint8(bf * 255)
}

func clampUint8(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// Normalize maps slice values to [0,1]. Returns a zero slice when min==max.
func Normalize(xs []float32) []float32 {
	if len(xs) == 0 {
		return xs
	}
	mn, mx := xs[0], xs[0]
	for _, v := range xs[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	out := make([]float32, len(xs))
	rng := mx - mn
	if rng < 1e-8 {
		return out // all-zero: uniform activation, no meaningful saliency
	}
	for i, v := range xs {
		out[i] = (v - mn) / rng
	}
	return out
}

// findOutput locates a tensor by output name. Returns an error if not found.
func findOutput(outputs []engine.Tensor, names []string, target string) (engine.Tensor, error) {
	for i, n := range names {
		if n == target {
			if i < len(outputs) {
				return outputs[i], nil
			}
			return engine.Tensor{}, fmt.Errorf("explain: output %q at index %d but only %d outputs returned", target, i, len(outputs))
		}
	}
	return engine.Tensor{}, fmt.Errorf("explain: output %q not found — session outputs are %v", target, names)
}

// upsampleNearest nearest-neighbor resizes src (srcW×srcH) to dst (dstW×dstH).
func upsampleNearest(src []float32, srcW, srcH, dstW, dstH int) []float32 {
	dst := make([]float32, dstW*dstH)
	for dy := 0; dy < dstH; dy++ {
		sy := dy * srcH / dstH
		if sy >= srcH {
			sy = srcH - 1
		}
		for dx := 0; dx < dstW; dx++ {
			sx := dx * srcW / dstW
			if sx >= srcW {
				sx = srcW - 1
			}
			dst[dy*dstW+dx] = src[sy*srcW+sx]
		}
	}
	return dst
}
