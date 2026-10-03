package pipeline

import "visionserve/internal/models"

// Grounded is detect-then-segment (Grounded-SAM): the detector's boxes for the words, then one
// mask per box, index-aligned and carrying the detection's BBox/Conf.
type Grounded struct {
	Detector  Detector
	Segmenter Segmenter
}

// Infer runs the composition. No detection means no segmentation pass at all.
func (g Grounded) Infer(c Call, words []string) (models.Result, error) {
	dets, err := g.Detector.Detect(c, words)
	if err != nil {
		return models.Result{}, err
	}
	if len(dets) == 0 {
		return models.Result{Detections: dets}, nil
	}
	masks, err := g.Segmenter.Segment(c, dets)
	if err != nil {
		return models.Result{}, err
	}
	return models.Result{Detections: dets, Masks: masks}, nil
}
