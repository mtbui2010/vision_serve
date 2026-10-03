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
// mask generator (the model's default grid) when it is nil.
func (s SAMBitmaps) SegmentBitmaps(c Call, boxes [][4]float64) ([]models.Mask, []mask.Bitmap, error) {
	masks, bms, err := s.Model.InferMasks(c.Img, models.Prompt{Boxes: boxes}, c.Runner)
	if err != nil {
		return nil, nil, err
	}
	bitmaps := make([]mask.Bitmap, len(bms))
	for i := range bms {
		bitmaps[i] = mask.Bitmap{Data: bms[i].Data, W: bms[i].W, H: bms[i].H}
	}
	return masks, bitmaps, nil
}
