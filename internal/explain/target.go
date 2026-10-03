package explain

import (
	"fmt"
	"math"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/detr"
	"visionserve/internal/vision/geom"
)

// BoxDecode is how a DETR-style detector's box output maps to pixels: boxes normalized to the
// InputW×InputH model input, in Format (geom.FormatCXCYWH when empty) — the manifest's
// input.width/height and postprocess.box_format, which its decoder uses too.
type BoxDecode struct {
	InputW, InputH int
	Format         string
}

// QueryForDetection returns the object query of a DETR-style session that produced det.
//
// A detection's position in /api/predict is NOT its query index: the decoder drops the queries
// under conf_threshold and sorts the rest by confidence. The attention explainer indexes
// cross_attn_weights by QUERY, so it needs the query back. The query is found by its box: every
// query's box in outputs (the [1,Q,4] tensor, picked by the decoders' own rule
// detr.SplitOutputs) is decoded exactly as the decoder decodes it — normalized → input pixels
// → original pixels via meta → clamped — and the query whose box reproduces det.BBox is the
// one. Pipelines that rescore the detector's queries (textalign) keep the detector's boxes
// bit-identical, so this holds for them as well.
//
// A box that no query reproduces (within 1 px plus 2% of the box size, the slack for two
// sessions of one graph not agreeing to the last bit) is an error rather than a guess: a guess
// is a heatmap of some other object, labelled as this one.
func QueryForDetection(outputs []engine.Tensor, meta models.PreprocessMeta, dec BoxDecode, det models.Detection) (int, error) {
	o, err := detr.SplitOutputs(outputs, 0, 0)
	if err != nil {
		return -1, fmt.Errorf("explain: %w", err)
	}
	boxes := o.Boxes
	if len(boxes.Shape) != 3 || boxes.Dim(-1) != 4 {
		return -1, fmt.Errorf("explain: boxes output has shape %v, want [1,Q,4]", boxes.Shape)
	}
	q := int(boxes.Dim(1))
	if len(boxes.Data) < q*4 {
		return -1, fmt.Errorf("explain: boxes output has %d values for %d queries", len(boxes.Data), q)
	}
	if dec.InputW <= 0 || dec.InputH <= 0 {
		return -1, fmt.Errorf("explain: invalid model input size %dx%d", dec.InputW, dec.InputH)
	}
	format := dec.Format
	if format == "" {
		format = geom.FormatCXCYWH
	}
	toOrig := meta.Affine()

	best, bestDist := -1, math.Inf(1)
	for i := 0; i < q; i++ {
		b := boxes.Data[i*4 : i*4+4]
		in := geom.NormToInput(float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3]), format, dec.InputW, dec.InputH)
		orig := geom.Clamp(toOrig.BoxToOrig(in), meta.OrigWidth, meta.OrigHeight)
		var d float64
		for k := range orig {
			d = math.Max(d, math.Abs(orig[k]-det.BBox[k]))
		}
		if d < bestDist {
			best, bestDist = i, d
		}
	}
	tol := 1 + 0.02*math.Max(det.BBox[2], det.BBox[3])
	if best < 0 || bestDist > tol {
		return -1, fmt.Errorf("explain: no query of the explain session has the box %v of the %q detection (closest is %.1f px off)",
			det.BBox, det.Class, bestDist)
	}
	return best, nil
}

// SameObjectScore is the Score-CAM score of target on a re-run of the detector (on a masked
// image): the highest confidence among dets of target's class whose box overlaps target's at
// IoU >= 0.5, or 0 when the object is no longer detected. The N-th detection of a re-run is
// not the N-th of the original run (masking changes which queries pass and their order), so
// the object is followed by class and position, not by index.
func SameObjectScore(dets []models.Detection, target models.Detection) float32 {
	var best float64
	for _, d := range dets {
		if d.Class == target.Class && d.Conf > best && geom.IoU(d.BBox, target.BBox) >= 0.5 {
			best = d.Conf
		}
	}
	return float32(best)
}
