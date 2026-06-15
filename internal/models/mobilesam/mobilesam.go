// Package mobilesam implements MobileSAM (segmentation, Apache-2.0) for the edge.
//
// MobileSAM = TinyViT image encoder + SAM prompt-encoder/mask-decoder, exported as TWO
// ONNX graphs (roles "encoder" and "decoder"). It needs a PROMPT (box/point), so it is
// a PipelineModel (not a plain Model): it drives encoder → decoder itself via a Runner.
//
// Verified I/O contract (see models/mobile-sam/verify_sam.py + README):
//
//	encoder: input  "input_image" [H,W,3] float32, raw 0..255 (normalize+pad are baked
//	                into the graph) — Go only resizes the long side to 1024.
//	         output "image_embeddings" [1,256,64,64].
//	decoder: image_embeddings, point_coords [1,n,2], point_labels [1,n], mask_input
//	         [1,1,256,256], has_mask_input [1], orig_im_size [2]=(H,W) ; outputs
//	         masks [1,N,H,W] (already upsampled to original size) + iou_predictions.
//	A box [x0,y0,x1,y1] is encoded as 2 points: top-left label 2, bottom-right label 3,
//	in resized-1024 space (= original × scale). Mask threshold = logit > 0.
package mobilesam

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("mobile-sam", New)
}

const (
	roleEncoder = "encoder"
	roleDecoder = "decoder"

	encoderSize = 1024 // SAM long-side input size
)

type mobileSAM struct {
	cfg      models.Config
	gridSize int // AMG grid N (N×N decoder calls); defaults to autoGridSize
}

// New is the factory called by lifecycle after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleEncoder] == "" || cfg.Files[roleDecoder] == "" {
		return nil, fmt.Errorf("mobilesam: manifest must declare files.%s and files.%s", roleEncoder, roleDecoder)
	}
	return &mobileSAM{cfg: cfg, gridSize: autoGridSize}, nil
}

// SetAutoGridSize overrides the Automatic Mask Generator grid: an N×N grid of point
// prompts → N² decoder calls. A smaller N is much faster (fewer decoder runs) but may
// miss small objects; n ≤ 0 resets to the default. Used by the foreground model to
// trade coverage for speed.
func (m *mobileSAM) SetAutoGridSize(n int) {
	if n <= 0 {
		n = autoGridSize
	}
	m.gridSize = n
}

func (m *mobileSAM) Name() string      { return m.cfg.Name }
func (m *mobileSAM) Task() models.Task { return models.TaskSegmentation }

// Roles are the ONNX sessions lifecycle must load (keys into the manifest 'files' map).
func (m *mobileSAM) Roles() []string { return []string{roleEncoder, roleDecoder} }

// PoolSizes requests a pool of 4 decoder sessions so AMG's 256 goroutines have
// 4 concurrent inference slots instead of serializing on a single mutex.
func (m *mobileSAM) PoolSizes() map[string]int { return map[string]int{roleDecoder: 4} }

// Infer runs the encoder once, then either:
//   - Automatic Mask Generator (16×16 grid) when no prompt is provided, or
//   - one decoder run per prompt set (box / point).
//
// It returns the public Result (masks as column-major RLE).
func (m *mobileSAM) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	bms, err := m.inferBitmaps(img, prompt, r)
	if err != nil {
		return models.Result{}, err
	}
	masks := make([]models.Mask, len(bms))
	for i, b := range bms {
		masks[i] = b.ToMask()
	}
	return models.Result{Masks: masks}, nil
}

// InferMasks runs the same encoder→decoder/AMG pipeline as Infer but also returns the
// raw mask bitmaps (original-image resolution). The masks slice is the RLE-encoded
// public form (index-aligned with the bitmaps); in-process consumers such as the grasp
// pipeline use the bitmaps directly and avoid decoding the RLE straight back. The
// encode happens once here (needed for the API response anyway); no decode ever runs.
func (m *mobileSAM) InferMasks(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Mask, []MaskBitmap, error) {
	bms, err := m.inferBitmaps(img, prompt, r)
	if err != nil {
		return nil, nil, err
	}
	masks := make([]models.Mask, len(bms))
	for i, b := range bms {
		masks[i] = b.ToMask()
	}
	return masks, bms, nil
}

// inferBitmaps is the shared core: encoder once, then AMG (no prompt) or one decoder
// run per prompt set, producing raw mask bitmaps at original-image resolution.
func (m *mobileSAM) inferBitmaps(img image.Image, prompt models.Prompt, r models.Runner) ([]MaskBitmap, error) {
	sets, err := promptToPointSets(prompt)
	if err != nil {
		return nil, err
	}

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	// Encoder: image → embedding (always runs once, shared across all decoder calls).
	encIn, scale := encoderInput(img)
	encInName := firstName(r.InputNames(roleEncoder), "input_image")
	encOuts, err := r.Run(roleEncoder, map[string]engine.Tensor{encInName: encIn})
	if err != nil {
		return nil, fmt.Errorf("mobilesam: encoder failed: %w", err)
	}
	if len(encOuts) == 0 {
		return nil, fmt.Errorf("mobilesam: encoder returned no output")
	}
	embedding := encOuts[0]

	decRun := func(inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleDecoder, inputs)
	}
	decOutNames := r.OutputNames(roleDecoder)

	// No prompt → Automatic Mask Generator (N×N grid, N² decoder calls). A per-request
	// prompt.GridSize (> 0) overrides the instance default (thread-safe: read per call).
	if sets == nil {
		grid := m.gridSize
		if prompt.GridSize > 0 {
			grid = prompt.GridSize
		}
		return autoSegment(img, embedding, scale, decRun, decOutNames, grid)
	}

	// Prompted: one decoder run per prompt set.
	zeros := make([]float32, 256*256)
	bitmaps := make([]MaskBitmap, 0, len(sets))
	for _, ps := range sets {
		coords := ps.scaledCoords(scale)
		dec := map[string]engine.Tensor{
			"image_embeddings": embedding,
			"point_coords":     engine.F32(coords, 1, int64(ps.n()), 2),
			"point_labels":     engine.F32(ps.labels, 1, int64(ps.n())),
			"mask_input":       engine.F32(zeros, 1, 1, 256, 256),
			"has_mask_input":   engine.F32([]float32{0}, 1),
			"orig_im_size":     engine.F32([]float32{float32(origH), float32(origW)}, 2),
		}
		outs, err := decRun(dec)
		if err != nil {
			return nil, fmt.Errorf("mobilesam: decoder failed: %w", err)
		}
		maskT, iouT := pickMaskAndIoU(decOutNames, outs)
		if maskT == nil {
			return nil, fmt.Errorf("mobilesam: decoder output has no mask tensor (shapes %v)", shapesOf(outs))
		}
		bm, err := maskToBitmap(maskT, iouT)
		if err != nil {
			return nil, err
		}
		bitmaps = append(bitmaps, bm)
	}

	return bitmaps, nil
}

func firstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}

func shapesOf(ts []engine.Tensor) [][]int64 {
	out := make([][]int64, len(ts))
	for i, t := range ts {
		out[i] = t.Shape
	}
	return out
}
