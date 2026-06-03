// Package efficientsam implements EfficientSAM (ViT-Tiny, Apache-2.0) for VisionServe.
//
// EfficientSAM is a lightweight promptable segmenter from "yformer/EfficientSAM".
// Like MobileSAM it uses TWO ONNX sessions (roles "encoder" and "decoder"), but the
// decoder API differs: EfficientSAM uses a batched point format with different shapes.
//
// Verified I/O contract (based on onnx-community/EfficientSAM ONNX exports):
//
//	encoder: input  "input_image" [1, 3, H, W] NCHW float32, ImageNet-normalized,
//	                resized so long side = 1024 (padding is NOT done — just resize+normalize).
//	         output "image_embeddings" [1, 256, 64, 64].
//
//	         TODO: verify exact input tensor name ("input_image" vs "image") and whether
//	               the graph bakes normalization or expects raw 0..255 values.
//
//	decoder: inputs:
//	           "image_embeddings"    [1, 256, 64, 64]
//	           "batched_point_coords" [1, 1, N, 2]  float32, pixel coords in 0..1023 space
//	           "batched_point_labels" [1, 1, N]     int64
//	             label 2 = box top-left, 3 = box bottom-right (same SAM convention)
//	             label 1 = foreground point, 0 = background point
//	         outputs:
//	           "low_res_masks"  [1, 1, 4, 256, 256]
//	           "iou_predictions" [1, 1, 4]
//
//	         TODO: verify these exact tensor names against the actual ONNX file.
//
// The mask at 256×256 must be upsampled to the original image size in postprocess
// (unlike MobileSAM whose decoder upsamples to orig size via orig_im_size input).
//
// IMPORTANT: EfficientSAM does NOT accept orig_im_size, mask_input, or has_mask_input.
// If the HF export turns out to be single-session (encoder+decoder fused into one graph),
// change Roles() to []string{"model"} and update Infer() accordingly.
package efficientsam

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("efficient-sam", New)
}

const (
	roleEncoder    = "encoder"
	roleDecoder    = "decoder"
	encoderSize    = 1024 // long-side resize target
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

// Roles lists the ONNX sessions lifecycle must load.
//
// TODO: if the onnx-community/EfficientSAM HF export is single-session, change this to
// []string{"model"} and update Infer() to use a single Run call.
func (m *efficientSAM) Roles() []string { return []string{roleEncoder, roleDecoder} }

// Infer runs the EfficientSAM pipeline:
//  1. Encode the image to an embedding (encoder session, once per image).
//  2. For each box/point-set in the prompt, run the decoder with batched_point_coords +
//     batched_point_labels to get 4 candidate masks; pick the one with the highest IoU.
//  3. Upsample selected 256×256 mask to original image size; threshold at logit > 0;
//     encode as column-major RLE.
func (m *efficientSAM) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	sets, err := promptToPointSets(prompt)
	if err != nil {
		return models.Result{}, err
	}

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	// ── 1. Encoder ──────────────────────────────────────────────────────────────
	// Verified encoder input name: "batched_images" [batch,3,H,W] float32.
	// The encoder accepts a rectangular image (no padding required).
	encIn, scale, err := encoderInput(img)
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
	// Output "output_masks" 5-D + "iou_predictions" [1,1,N_masks].
	origImSize := engine.I64([]int64{int64(origH), int64(origW)}, 2)
	outMasks := make([]models.Mask, 0, len(sets))
	for _, ps := range sets {
		coords, labels := ps.batchedTensors(scale)

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

		// TODO: verify output tensor names ("low_res_masks" and "iou_predictions").
		maskT, iouT, pickErr := pickBestMask(r.OutputNames(roleDecoder), outs)
		if pickErr != nil {
			return models.Result{}, fmt.Errorf("efficientsam: picking best mask: %w", pickErr)
		}

		mk, err := maskToResult(maskT, iouT, origW, origH)
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
