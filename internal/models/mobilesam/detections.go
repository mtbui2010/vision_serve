package mobilesam

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/pkg/api"
)

// SegmentDetections is the detect-then-segment step every text-prompted pipeline ends with
// (grounded-sam, the rfdetr-gdino router with masks): one box-prompted MobileSAM mask per
// detection via Segment, through the encoder/decoder sessions under encRole/decRole, each mask
// stamped with its detection's BBox and Conf so a client can pair them without the detections
// slice. Masks are index-aligned with dets.
func SegmentDetections(img image.Image, dets []api.Detection, r models.Runner, encRole, decRole string) ([]api.Mask, error) {
	boxes := make([][4]float64, len(dets))
	for i, d := range dets {
		boxes[i] = d.BBox
	}
	encRun := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(encRole, in) }
	decRun := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(decRole, in) }
	masks, err := Segment(img, boxes, encRun, decRun, firstName(r.InputNames(encRole), "input_image"), r.OutputNames(decRole))
	if err != nil {
		return nil, err
	}
	for i := range masks {
		if i < len(dets) {
			masks[i].BBox = dets[i].BBox
			masks[i].Conf = dets[i].Conf
		}
	}
	return masks, nil
}
