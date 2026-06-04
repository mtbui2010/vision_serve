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
	cfg models.Config
}

// New is the factory called by lifecycle after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleEncoder] == "" || cfg.Files[roleDecoder] == "" {
		return nil, fmt.Errorf("mobilesam: manifest must declare files.%s and files.%s", roleEncoder, roleDecoder)
	}
	return &mobileSAM{cfg: cfg}, nil
}

func (m *mobileSAM) Name() string      { return m.cfg.Name }
func (m *mobileSAM) Task() models.Task { return models.TaskSegmentation }

// Roles are the ONNX sessions lifecycle must load (keys into the manifest 'files' map).
func (m *mobileSAM) Roles() []string { return []string{roleEncoder, roleDecoder} }

// Infer runs the encoder once, then either:
//   - Automatic Mask Generator (16×16 grid) when no prompt is provided, or
//   - one decoder run per prompt set (box / point).
func (m *mobileSAM) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	sets, err := promptToPointSets(prompt)
	if err != nil {
		return models.Result{}, err
	}

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	// Encoder: image → embedding (always runs once, shared across all decoder calls).
	encIn, scale := encoderInput(img)
	encInName := firstName(r.InputNames(roleEncoder), "input_image")
	encOuts, err := r.Run(roleEncoder, map[string]engine.Tensor{encInName: encIn})
	if err != nil {
		return models.Result{}, fmt.Errorf("mobilesam: encoder failed: %w", err)
	}
	if len(encOuts) == 0 {
		return models.Result{}, fmt.Errorf("mobilesam: encoder returned no output")
	}
	embedding := encOuts[0]

	decRun := func(inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleDecoder, inputs)
	}
	decOutNames := r.OutputNames(roleDecoder)

	// No prompt → Automatic Mask Generator (16×16 grid, ~256 decoder calls).
	if sets == nil {
		masks, err := autoSegment(img, embedding, scale, decRun, decOutNames)
		if err != nil {
			return models.Result{}, err
		}
		return models.Result{Masks: masks}, nil
	}

	// Prompted: one decoder run per prompt set.
	zeros := make([]float32, 256*256)
	masks := make([]models.Mask, 0, len(sets))
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
			return models.Result{}, fmt.Errorf("mobilesam: decoder failed: %w", err)
		}
		maskT, iouT := pickMaskAndIoU(decOutNames, outs)
		if maskT == nil {
			return models.Result{}, fmt.Errorf("mobilesam: decoder output has no mask tensor (shapes %v)", shapesOf(outs))
		}
		mk, err := maskToResult(maskT, iouT, origW, origH)
		if err != nil {
			return models.Result{}, err
		}
		masks = append(masks, mk)
	}

	return models.Result{Masks: masks}, nil
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
