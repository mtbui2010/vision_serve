// Package nanosam implements NanoSAM (segmentation, Apache-2.0 license) for the edge.
//
// NanoSAM is NVIDIA's edge-optimized SAM variant designed for Jetson.
// It reuses the standard SAM prompt-encoder/mask-decoder but replaces TinyViT
// with a lightweight ResNet-18 image encoder. Like MobileSAM it is a two-session
// PipelineModel (roles "encoder" and "decoder") and needs a box/point prompt.
//
// Key difference from MobileSAM (see preprocess.go):
//
//	MobileSAM encoder: HWC float32 raw 0..255 — normalization baked into graph.
//	NanoSAM encoder:   NCHW float32 ImageNet-normalized — Go must normalize before ORT.
//
// Tensor names (verified from nanosam/tools/export_image_encoder_onnx.py and
// nanosam/tools/export_sam_mask_decoder_onnx.py in github.com/NVIDIA-AI-IOT/nanosam):
//
//	Encoder input:   "image"            [1, 3, 1024, 1024] NCHW ImageNet-normalized
//	Encoder output:  "image_embeddings" [1, 256, 64, 64]
//
//	Decoder inputs:  "image_embeddings" [1, 256, 64, 64]
//	                 "point_coords"     [1, N, 2]
//	                 "point_labels"     [1, N]
//	                 "mask_input"       [1, 1, 256, 256]
//	                 "has_mask_input"   [1]
//	                 NOTE: NanoSAM decoder does NOT have "orig_im_size" input.
//	Decoder outputs: "iou_predictions"  [1, M]
//	                 "low_res_masks"    [1, M, 256, 256]
//	                 NOTE: output is "low_res_masks" (256×256), not upsampled "masks".
//
// ONNX files (from github.com/NVIDIA-AI-IOT/nanosam, Google Drive links in README):
//
//	resnet18_image_encoder.onnx  — image encoder
//	mobile_sam_mask_decoder.onnx — mask decoder (reuses MobileSAM decoder)
package nanosam

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func init() {
	models.Register("nano-sam", New)
}

const (
	roleEncoder = "encoder"
	roleDecoder = "decoder"
)

type nanoSAM struct {
	cfg models.Config
}

// New is the factory called by lifecycle after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleEncoder] == "" || cfg.Files[roleDecoder] == "" {
		return nil, fmt.Errorf("nanosam: manifest must declare files.%s and files.%s", roleEncoder, roleDecoder)
	}
	return &nanoSAM{cfg: cfg}, nil
}

func (m *nanoSAM) Name() string      { return m.cfg.Name }
func (m *nanoSAM) Task() models.Task { return models.TaskSegmentation }

// Roles are the ONNX session keys lifecycle must load (keys in the manifest 'files' map).
func (m *nanoSAM) Roles() []string { return []string{roleEncoder, roleDecoder} }

// Infer runs the NanoSAM pipeline: encode image → decode once per prompt box/point set.
//
// Encoder step: produces an [1,256,64,64] image embedding via NCHW ImageNet-normalized input.
// Decoder step: runs once per box (or once for the combined point set) → one mask each.
func (m *nanoSAM) Infer(img image.Image, prompt models.Prompt, runner models.Runner) (models.Result, error) {
	if prompt.Empty() {
		return models.Result{}, fmt.Errorf("nanosam: a box or point prompt is required — " +
			"NanoSAM segments around a prompt, e.g. `run nano-sam img.jpg --box x,y,w,h`")
	}

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	// 1) Encoder: image → embedding.
	// scale maps original-image coordinates into the 1024 space the decoder expects.
	encIn, scale := encoderInput(img)

	// Encoder input tensor is named "image" — confirmed from export_image_encoder_onnx.py
	// in github.com/NVIDIA-AI-IOT/nanosam (input_names=["image"]).
	// Fall back to first reported name in case a custom export was used.
	encInputName := firstNameFrom(runner.InputNames(roleEncoder), "image")
	encOuts, err := runner.Run(roleEncoder, map[string]engine.Tensor{encInputName: encIn})
	if err != nil {
		return models.Result{}, fmt.Errorf("nanosam: encoder failed: %w", err)
	}
	if len(encOuts) == 0 {
		return models.Result{}, fmt.Errorf("nanosam: encoder returned no output")
	}
	embedding := encOuts[0]

	decOutNames := runner.OutputNames(roleDecoder)

	// 2) Decoder per box prompt → mask (or once for all points together).
	var masks []models.Mask

	if len(prompt.Boxes) > 0 {
		// One decoder run per box.
		for _, box := range prompt.Boxes {
			singleBoxPrompt := models.Prompt{Boxes: [][4]float64{box}}
			coords, labels, encErr := encodePrompt(singleBoxPrompt, scale)
			if encErr != nil {
				return models.Result{}, encErr
			}
			outs, runErr := runDecoder(runner, embedding, coords, labels, origH, origW)
			if runErr != nil {
				return models.Result{}, runErr
			}
			mk, pickErr := pickBestMask(outs, decOutNames, origW, origH)
			if pickErr != nil {
				return models.Result{}, pickErr
			}
			masks = append(masks, mk)
		}
	} else {
		// Points-only: single decoder run.
		coords, labels, encErr := encodePrompt(prompt, scale)
		if encErr != nil {
			return models.Result{}, encErr
		}
		outs, runErr := runDecoder(runner, embedding, coords, labels, origH, origW)
		if runErr != nil {
			return models.Result{}, runErr
		}
		mk, pickErr := pickBestMask(outs, decOutNames, origW, origH)
		if pickErr != nil {
			return models.Result{}, pickErr
		}
		masks = append(masks, mk)
	}

	return models.Result{Masks: masks}, nil
}

// firstNameFrom returns the first element of names if non-empty, otherwise fallback.
func firstNameFrom(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}
