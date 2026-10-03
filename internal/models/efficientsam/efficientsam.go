// Package efficientsam implements EfficientSAM (ViT-Tiny, Apache-2.0) for VisionServe.
//
// EfficientSAM is a lightweight promptable segmenter from "yformer/EfficientSAM".
// Like MobileSAM it uses TWO ONNX sessions (roles "encoder" and "decoder").
//
// I/O contract VERIFIED against models/efficient-sam/*.onnx (shapes via onnxruntime,
// pre/post ops via graph inspection) and against the official ONNX example:
//
//	encoder: input  "batched_images" [1, 3, H, W] float32 in [0,1] (pixel/255) at the
//	                ORIGINAL size — the graph itself resizes to 1024×1024 and applies
//	                ImageNet mean/std (see preprocess.go).
//	         output "image_embeddings" [1, 256, 64, 64].
//
//	decoder: inputs:
//	           "image_embeddings"     [1, 256, 64, 64]
//	           "batched_point_coords" [1, 1, N, 2] float32, ORIGINAL-image pixel coords
//	                                  (the graph rescales x·1024/W, y·1024/H itself)
//	           "batched_point_labels" [1, 1, N]    float32
//	             label 2 = box top-left, 3 = box bottom-right, 1 = fg point, 0 = bg point
//	           "orig_im_size"         [2] int64 = [H, W]
//	         outputs:
//	           "output_masks"     [1, 1, 3, H, W] logits at ORIGINAL resolution (the graph
//	                              upsamples) — thresholded directly at logit >= 0
//	           "iou_predictions"  [1, 1, 3] (NOT sorted — we take the argmax)
//	           a third low-res tensor (unused)
package efficientsam

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/preprocess"
)

// Compile-time checks of the interfaces lifecycle type-asserts at load: a signature drift
// fails the build instead of silently changing how the model is run.
var _ models.PipelineModel = (*efficientSAM)(nil)

func init() {
	// The registered factory refuses a declared preprocess: block (fixed by the export); New
	// itself does not, since composites build this model from manifests whose block belongs to
	// another stage.
	models.Register("efficient-sam", func(cfg models.Config) (models.Base, error) {
		if err := preprocess.FixedByExport("efficient-sam", cfg.PreprocessSpec()); err != nil {
			return nil, err
		}
		return New(cfg)
	})
}

const (
	roleEncoder = "encoder"
	roleDecoder = "decoder"
)

type efficientSAM struct {
	cfg models.Config
}

// New is the factory called by lifecycle after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleEncoder] == "" || cfg.Files[roleDecoder] == "" {
		return nil, fmt.Errorf("efficientsam: manifest must declare files.%s and files.%s", roleEncoder, roleDecoder)
	}
	return &efficientSAM{cfg: cfg}, nil
}

func (m *efficientSAM) Name() string      { return m.cfg.Name }
func (m *efficientSAM) Task() models.Task { return models.TaskSegmentation }

// Roles lists the ONNX sessions lifecycle must load (verified two-session export).
func (m *efficientSAM) Roles() []string { return []string{roleEncoder, roleDecoder} }

// Infer runs the EfficientSAM pipeline:
//  1. Encode the image to an embedding (encoder session, once per image).
//  2. For each box/point-set in the prompt, run the decoder with batched_point_coords +
//     batched_point_labels to get 3 candidate masks; pick the one with the highest IoU.
//  3. Threshold the selected ORIGINAL-resolution mask at logit >= 0; encode as
//     column-major RLE.
func (m *efficientSAM) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	sets, err := promptToPointSets(prompt)
	if err != nil {
		return models.Result{}, err
	}

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	// ── 1. Encoder ──────────────────────────────────────────────────────────────
	// Verified encoder input name: "batched_images" [batch,3,H,W] float32 in [0,1];
	// the graph resizes to 1024² and normalizes internally.
	encIn, err := encoderInput(img)
	if err != nil {
		return models.Result{}, fmt.Errorf("efficientsam: preprocess failed: %w", err)
	}
	encInputName := firstName(r.InputNames(roleEncoder), "batched_images")
	encOuts, err := r.Run(roleEncoder, map[string]engine.Tensor{encInputName: encIn})
	if err != nil {
		return models.Result{}, fmt.Errorf("efficientsam: encoder failed: %w", err)
	}
	if len(encOuts) == 0 {
		return models.Result{}, fmt.Errorf("efficientsam: encoder returned no outputs")
	}
	embedding := encOuts[0]

	// ── 2. Decoder per prompt set ────────────────────────────────────────────────
	// Verified decoder inputs (yunyangx/EfficientSAM efficientsam_ti_decoder.onnx):
	//   "image_embeddings"    [batch,256,64,64]
	//   "batched_point_coords" [1,1,N,2] float32
	//   "batched_point_labels" [1,1,N]   float32 (NOT int64)
	//   "orig_im_size"         [2]       int64  = [origH, origW]
	// Output "output_masks" [1,1,3,H,W] (original res) + "iou_predictions" [1,1,3].
	origImSize := engine.I64([]int64{int64(origH), int64(origW)}, 2)
	outMasks := make([]models.Mask, 0, len(sets))
	for _, ps := range sets {
		coords, labels := ps.batchedTensors()

		dec := map[string]engine.Tensor{
			"image_embeddings":     embedding,
			"batched_point_coords": coords,
			"batched_point_labels": labels,
			"orig_im_size":         origImSize,
		}
		outs, err := r.Run(roleDecoder, dec)
		if err != nil {
			return models.Result{}, fmt.Errorf("efficientsam: decoder failed: %w", err)
		}

		plane, mH, mW, score, pickErr := pickBestMask(r.OutputNames(roleDecoder), outs)
		if pickErr != nil {
			return models.Result{}, fmt.Errorf("efficientsam: picking best mask: %w", pickErr)
		}

		mk, err := maskToResult(plane, mW, mH, score, origW, origH)
		if err != nil {
			return models.Result{}, err
		}
		outMasks = append(outMasks, mk)
	}

	return models.Result{Masks: outMasks}, nil
}

// firstName returns the first name from the slice, or the fallback if empty.
func firstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}
