package pipeline

import (
	graspcore "visionserve/internal/grasp"
	"visionserve/internal/models"
	"visionserve/internal/vision/mask"
	"visionserve/pkg/api"
)

// AnalyticGrasp is the GraspPlanner stage: the pure-Go antipodal mask2grasp search
// (internal/grasp), no session and no weights.
type AnalyticGrasp struct {
	// GripMin/GripMax are the manifest's jaw-opening bounds in pixels (0 = the core default); a
	// request's gripper_min/gripper_max override them.
	GripMin, GripMax float64
	// MaxPerMask caps the grasps returned for EACH mask, best first. It is deliberately not the
	// manifest's max_detections, which is the DETECTOR's cap (300 on grasp-rfdetr): reused as the
	// grasp cap it let one star-shaped mask return thousands of grasps.
	MaxPerMask int
}

// Params resolves the search parameters for one request: core defaults, overridden by the
// manifest, overridden again by the request.
func (a AnalyticGrasp) Params(p models.Prompt) graspcore.Params {
	gp := graspcore.DefaultParams()
	if a.GripMin > 0 {
		gp.Dmin = a.GripMin
	}
	if a.GripMax > 0 {
		gp.Dmax = a.GripMax
	}
	if p.GripperMin > 0 {
		gp.Dmin = p.GripperMin
	}
	if p.GripperMax > 0 {
		gp.Dmax = p.GripperMax
	}
	gp.MaxGrasps = a.MaxPerMask
	return gp
}

// Plan implements GraspPlanner.
func (a AnalyticGrasp) Plan(m mask.Bitmap, p models.Prompt) []api.Grasp {
	return graspcore.FromMask(m, a.Params(p))
}

// Grasp composes the planar grasp pipeline: [Detector] → BitmapSegmenter → GraspPlanner.
//
//   - With a Detector: detect → size-filter → one mask per box → grasps per mask, each grasp and
//     mask carrying its detection's class/conf (class-aware grasps).
//   - Without one, a request carrying boxes segments exactly those boxes (the "pick the target
//     client-side, then grasp just it" flow); otherwise the whole image is auto-masked, size-
//     filtered and every mask grasped (class-agnostic).
type Grasp struct {
	Detector Detector // nil: class-agnostic
	// Words gives the detector its words for a request (nil: the detector gets none, e.g. a
	// closed-set detector answering with everything it knows).
	Words     func(p models.Prompt) ([]string, error)
	Segmenter BitmapSegmenter
	Planner   GraspPlanner
}

// Infer runs the composition.
func (g Grasp) Infer(c Call) (models.Result, error) {
	w, h := c.Img.Bounds().Dx(), c.Img.Bounds().Dy()
	res := models.Result{Task: models.TaskGrasp}

	if g.Detector != nil {
		var words []string
		if g.Words != nil {
			var err error
			if words, err = g.Words(c.Prompt); err != nil {
				return models.Result{}, err
			}
		}
		dets, err := g.Detector.Detect(c, words)
		if err != nil {
			return models.Result{}, err
		}
		dets = filterDetections(dets, c.Prompt, w, h)
		if len(dets) == 0 {
			return res, nil
		}
		boxes := make([][4]float64, len(dets))
		for i, d := range dets {
			boxes[i] = d.BBox
		}
		masks, bitmaps, err := g.Segmenter.SegmentBitmaps(c, boxes)
		if err != nil {
			return models.Result{}, err
		}
		for i := range bitmaps {
			gs := g.Planner.Plan(bitmaps[i], c.Prompt)
			if i < len(dets) {
				for j := range gs {
					gs[j].Class = dets[i].Class
					gs[j].Conf = dets[i].Conf
				}
				if i < len(masks) {
					masks[i].BBox = dets[i].BBox
					masks[i].Conf = dets[i].Conf
				}
			}
			res.Grasps = append(res.Grasps, gs...)
		}
		res.Detections = dets
		res.Masks = masks
		return res, nil
	}

	boxes := c.Prompt.Boxes // nil: automatic masks over the whole image
	masks, bitmaps, err := g.Segmenter.SegmentBitmaps(c, boxes)
	if err != nil {
		return models.Result{}, err
	}
	if len(boxes) == 0 {
		masks, bitmaps = filterMasksBitmaps(masks, bitmaps, c.Prompt, w, h)
	}
	for i := range bitmaps {
		res.Grasps = append(res.Grasps, g.Planner.Plan(bitmaps[i], c.Prompt)...)
	}
	res.Masks = masks
	return res, nil
}

func filterDetections(dets []models.Detection, p models.Prompt, w, h int) []models.Detection {
	if p.MinSize <= 0 && p.MaxSize <= 0 {
		return dets
	}
	return api.FilterBySizePct(api.Result{Detections: dets}, p.MinSize, p.MaxSize, w, h).Detections
}

// filterMasksBitmaps applies the same bbox-area size filter as api.FilterBySizePct, keeping the
// masks and their index-aligned bitmaps in lockstep.
func filterMasksBitmaps(masks []models.Mask, bitmaps []mask.Bitmap, p models.Prompt, w, h int) ([]models.Mask, []mask.Bitmap) {
	if p.MinSize <= 0 && p.MaxSize <= 0 {
		return masks, bitmaps
	}
	area := float64(w * h)
	var minAbs, maxAbs float64
	if p.MinSize > 0 {
		minAbs = p.MinSize / 100.0 * area
	}
	if p.MaxSize > 0 {
		maxAbs = p.MaxSize / 100.0 * area
	}
	outM := make([]models.Mask, 0, len(masks))
	outB := make([]mask.Bitmap, 0, len(bitmaps))
	for i := range masks {
		a := masks[i].BBox[2] * masks[i].BBox[3]
		if minAbs > 0 && a < minAbs {
			continue
		}
		if maxAbs > 0 && a > maxAbs {
			continue
		}
		outM = append(outM, masks[i])
		if i < len(bitmaps) {
			outB = append(outB, bitmaps[i])
		}
	}
	return outM, outB
}
