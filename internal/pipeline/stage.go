// Package pipeline builds the composite open-vocabulary models out of small stages with small
// interfaces, so a model is a CONFIGURATION — which detector answers which words, whether a
// rescorer re-ranks them, whether a segmenter masks them, whether a planner grasps them — instead
// of a hand-written Infer:
//
//	Detector(img, words)          → []Detection   RF-DETR (closed set), GroundingDINO (open)
//	Rescorer(img, dets, words)    → []Detection   SigLIP crop namer
//	Segmenter(img, dets)          → []Mask        MobileSAM, one mask per detection
//	GraspPlanner(mask)            → []Grasp       analytic parallel-jaw search
//
// The registered models stay where they were (internal/models/hybrid, groundedsam, grasp, …)
// and keep their architecture names, manifests and Roles(); each one's factory wires stages from
// this package into one of the compositions below (Router, Grounded, Grasp). Replacing a
// component — SigLIP by CLIP, MobileSAM by another box-prompted segmenter — is replacing one
// stage.
//
// Stages own no ONNX session. Every session stays owned by lifecycle.Manager and is reached by
// role through the request's models.Runner (CLAUDE.md), which is why it travels in Call.
package pipeline

import (
	"image"

	"visionserve/internal/models"
	"visionserve/internal/vision/mask"
	"visionserve/pkg/api"
)

// Call is one request as every stage sees it: the image, the request's options (thresholds,
// crop_temp, …, read by the stage they concern) and the Runner that reaches the sessions.
type Call struct {
	Img    image.Image
	Prompt models.Prompt
	Runner models.Runner
}

// Detector finds the objects named by words. Boxes are in ORIGINAL image coordinates.
// What an empty word list means is the detector's own contract (a closed-set detector answers
// with everything it knows; GroundingDINO has nothing to look for).
type Detector interface {
	Detect(c Call, words []string) ([]models.Detection, error)
}

// Rescorer re-weights and possibly renames detections against the words they were asked about,
// and may drop the ones it rejects. Order is preserved.
type Rescorer interface {
	Rescore(c Call, dets []models.Detection, words []string) ([]models.Detection, error)
}

// Segmenter returns one mask per detection, index-aligned, each carrying its detection's BBox and
// Conf so a client can pair them without the detections slice.
type Segmenter interface {
	Segment(c Call, dets []models.Detection) ([]models.Mask, error)
}

// BitmapSegmenter is the segmenter a grasp planner needs: the API masks AND their raw bitmaps at
// original resolution, index-aligned, so the planner reads pixels without decoding the RLE back.
// boxes == nil segments the whole image (automatic masks).
type BitmapSegmenter interface {
	SegmentBitmaps(c Call, boxes [][4]float64) ([]models.Mask, []mask.Bitmap, error)
}

// EachBitmapSegmenter is an optional BitmapSegmenter that streams: SegmentEach hands each mask
// and its bitmap to fn as soon as the mask is final, and returns fn's results in SegmentBitmaps'
// order, so the caller never holds every full-resolution bitmap at once. fn may run
// concurrently and must not keep the bitmap's Data. ok=false means this segmenter cannot stream
// (nothing ran); the caller then uses SegmentBitmaps.
type EachBitmapSegmenter interface {
	SegmentEach(c Call, boxes [][4]float64, fn func(m models.Mask, b mask.Bitmap) any) (out []any, ok bool, err error)
}

// GraspPlanner turns one object mask into grasps (original-image pixels), best first.
type GraspPlanner interface {
	Plan(m mask.Bitmap, p models.Prompt) []api.Grasp
}
