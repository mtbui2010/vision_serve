// Package sam2 implements SAM2-Tiny (segmentation, Apache-2.0) for VisionServe.
//
// SAM2 (Segment Anything Model 2) by Meta AI — Apache-2.0 license.
// Source: https://github.com/facebookresearch/segment-anything-2
//
// SAM2 is a prompted, two-session model (a PipelineModel, not a plain Model).
// It requires a PROMPT (a box or a point). Two ONNX graphs declared in the manifest.
// Verified I/O (SharpAI/sam2-hiera-tiny-onnx, inspected with onnxruntime):
//
//	encoder: input  "image" [1,3,1024,1024] float32, NCHW, ImageNet-normalized; the image
//	         is SQUASHED to 1024×1024 (upstream SAM2Transforms, no letterbox).
//	         outputs "high_res_feats_0" [1,32,256,256], "high_res_feats_1" [1,64,128,128],
//	                 "image_embed" [1,256,64,64]  — multi-scale features.
//	decoder: image_embed, high_res_feats_0, high_res_feats_1,
//	         point_coords [1,N,2] float32 (1024-space, per-axis scaled),
//	         point_labels [1,N] float32 (1=fg, 0=bg, 2=tl-box, 3=br-box),
//	         mask_input [1,1,256,256] float32, has_mask_input [1] float32.
//	         outputs: masks [1,3,256,256] float32 LOW-RES logits (clamped to ±32),
//	                  iou_predictions [1,3] float32.
//
// The best of the 3 candidates (by iou_predictions) is bilinearly upsampled to the
// ORIGINAL image size (upstream postprocess_masks) before thresholding at 0, so the RLE
// and BBox are in original-image coordinates.
//
// Unlike MobileSAM the encoder outputs THREE tensors (multi-scale); all three must be
// forwarded to the decoder.
package sam2

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/preprocess"
)

func init() {
	// The registered factory refuses a declared preprocess: block (fixed by the export); New
	// itself does not, since composites build this model from manifests whose block belongs to
	// another stage.
	models.Register("sam2", func(cfg models.Config) (models.Base, error) {
		if err := preprocess.FixedByExport("sam2", cfg.PreprocessSpec()); err != nil {
			return nil, err
		}
		return New(cfg)
	})
}

const (
	roleEncoder = "encoder"
	roleDecoder = "decoder"

	encoderSize = 1024 // SAM2 encoder input size (long side)
)

type sam2Model struct {
	cfg models.Config
}

// New is the factory called by lifecycle after parsing the manifest.
func New(cfg models.Config) (models.Base, error) {
	if cfg.Files[roleEncoder] == "" || cfg.Files[roleDecoder] == "" {
		return nil, fmt.Errorf("sam2: manifest must declare files.%s and files.%s", roleEncoder, roleDecoder)
	}
	return &sam2Model{cfg: cfg}, nil
}

func (m *sam2Model) Name() string      { return m.cfg.Name }
func (m *sam2Model) Task() models.Task { return models.TaskSegmentation }

// Roles are the ONNX sessions lifecycle must load (keys into the manifest 'files' map).
func (m *sam2Model) Roles() []string { return []string{roleEncoder, roleDecoder} }

// Infer runs the SAM2 pipeline:
//  1. Encode image → image_embed + high_res_feats_0 + high_res_feats_1.
//  2. Decode (encoder outputs + point prompt) → mask per prompt set.
func (m *sam2Model) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	sets, err := promptToPointSets(prompt)
	if err != nil {
		return models.Result{}, err
	}

	// 1) Build encoder input: NCHW float32 [1,3,1024,1024], squashed + SAM2-normalized.
	encTensor, scaleX, scaleY, err := encoderInput(img)
	if err != nil {
		return models.Result{}, fmt.Errorf("sam2: encoder preprocess failed: %w", err)
	}

	// TODO: verify encoder input name against your ONNX export.
	encInputName := firstName(r.InputNames(roleEncoder), "image")

	encOuts, err := r.Run(roleEncoder, map[string]engine.Tensor{encInputName: encTensor})
	if err != nil {
		return models.Result{}, fmt.Errorf("sam2: encoder failed: %w", err)
	}

	// SAM2 encoder emits 3 tensors: image_embed, high_res_feats_0, high_res_feats_1.
	imageEmbed, highResFeats0, highResFeats1, err := pickEncoderOutputs(r.OutputNames(roleEncoder), encOuts)
	if err != nil {
		return models.Result{}, fmt.Errorf("sam2: encoder outputs: %w", err)
	}

	// 2) Decoder per prompt set → one mask each.
	masks := make([]models.Mask, 0, len(sets))
	for _, ps := range sets {
		coords, labels := ps.scaledPrompt(scaleX, scaleY)
		maskTensor, iouTensor, err := runDecoder(r, imageEmbed, highResFeats0, highResFeats1, coords, labels)
		if err != nil {
			return models.Result{}, fmt.Errorf("sam2: decoder failed: %w", err)
		}
		mk, err := pickBestMask(maskTensor, iouTensor, img.Bounds().Dx(), img.Bounds().Dy())
		if err != nil {
			return models.Result{}, fmt.Errorf("sam2: postprocess failed: %w", err)
		}
		masks = append(masks, mk)
	}

	return models.Result{Masks: masks}, nil
}

// firstName returns the first name in a slice, or the fallback when empty.
func firstName(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}
