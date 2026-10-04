package pipeline

import (
	"image"

	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"
	"visionserve/internal/vision/mask"
)

// SAM is the MobileSAM Segmenter stage: one box-prompted mask per detection, through the
// encoder/decoder sessions under Encoder/Decoder (the encoder runs once per request).
type SAM struct{ Encoder, Decoder string }

// Segment implements Segmenter.
func (s SAM) Segment(c Call, dets []models.Detection) ([]models.Mask, error) {
	return mobilesam.SegmentDetections(c.Img, dets, c.Runner, s.Encoder, s.Decoder)
}

// MaskInferer is a segmentation model that hands back its masks' bitmaps (MobileSAM).
type MaskInferer interface {
	InferMasks(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Mask, []mobilesam.MaskBitmap, error)
}

// SAMBitmaps is the BitmapSegmenter stage over a MaskInferer: one RLE encode for the API, the
// bitmap for the planner, zero decode.
type SAMBitmaps struct{ Model MaskInferer }

// SegmentBitmaps implements BitmapSegmenter: box prompts when boxes is non-empty, the automatic
// mask generator when it is nil (the request's grid_size, else the model's default grid).
func (s SAMBitmaps) SegmentBitmaps(c Call, boxes [][4]float64) ([]models.Mask, []mask.Bitmap, error) {
	masks, bms, err := s.Model.InferMasks(c.Img, segPrompt(c, boxes), c.Runner)
	if err != nil {
		return nil, nil, err
	}
	bitmaps := make([]mask.Bitmap, len(bms))
	for i := range bms {
		bitmaps[i] = mask.Bitmap{Data: bms[i].Data, W: bms[i].W, H: bms[i].H}
	}
	return masks, bitmaps, nil
}

// MaskEacher is a MaskInferer that can hand each bitmap to a callback as soon as it is final
// instead of returning all of them (MobileSAM's InferMasksEach): fn may run concurrently and
// must not keep b.Data; its results come back in InferMasks' order.
type MaskEacher interface {
	InferMasksEach(img image.Image, prompt models.Prompt, r models.Runner, fn func(b mobilesam.MaskBitmap) any) ([]any, error)
}

// SegmentEach implements EachBitmapSegmenter when the model is a MaskEacher (ok=false when it
// is not): each mask is encoded and handed to fn with its bitmap as soon as it is final.
func (s SAMBitmaps) SegmentEach(c Call, boxes [][4]float64, fn func(m models.Mask, b mask.Bitmap) any) (out []any, ok bool, err error) {
	me, ok := s.Model.(MaskEacher)
	if !ok {
		return nil, false, nil
	}
	out, err = me.InferMasksEach(c.Img, segPrompt(c, boxes), c.Runner, func(b mobilesam.MaskBitmap) any {
		return fn(b.ToMask(), mask.Bitmap{Data: b.Data, W: b.W, H: b.H})
	})
	return out, true, err
}

// segPrompt is the segmenter's prompt: the boxes to cut out, or none for automatic masks, which
// then use the request's grid_size (0 = the model's default grid). Only these two fields: the
// request's text, points and thresholds belong to the pipeline, not to the segmenter.
func segPrompt(c Call, boxes [][4]float64) models.Prompt {
	return models.Prompt{Boxes: boxes, GridSize: c.Prompt.GridSize}
}
