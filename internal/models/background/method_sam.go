package background

import (
	"image"

	"visionserve/internal/models"
)

// samMaxAreaPct drops absurdly-huge candidate masks: a mask covering ≥ this percent of the
// frame is the whole image, not a support surface.
const samMaxAreaPct = 97.0

// samSeedGrid is a grid of normalized seed points (fractions of W/H) across the LOWER frame,
// where the support surface usually is. Each seed is prompted SEPARATELY so that a seed that
// happens to land on bare surface yields a clean surface mask even when other seeds land on
// clutter (a single combined-points prompt fails on a busy tabletop — SAM tries to include
// the objects the points sit on). Cols × rows = a spread that tolerates a cluttered centre.
var (
	samSeedColsX = []float64{0.30, 0.50, 0.70}
	samSeedRowsY = []float64{0.72, 0.90}
)

// samSeedPoints returns the grid seed points (foreground, label 1) in ORIGINAL pixels.
func samSeedPoints(w, h int) []models.Point {
	fw, fh := float64(w), float64(h)
	pts := make([]models.Point, 0, len(samSeedColsX)*len(samSeedRowsY))
	for _, fy := range samSeedRowsY {
		for _, fx := range samSeedColsX {
			x := clampF(fx*fw, 0, fw-1)
			y := clampF(fy*fh, 0, fh-1)
			pts = append(pts, models.Point{X: x, Y: y, Label: 1})
		}
	}
	return pts
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// backgroundSAM (method=sam) returns the support-surface (background) mask by prompting
// MobileSAM at a GRID of lower-frame seed points — each seed in its OWN prompt so a seed on
// bare surface produces a clean surface mask even on a cluttered tabletop — then keeping the
// LARGEST mask that qualifies as a support surface (large by area, AND border-touching).
// One encoder pass per seed (~tens of ms each). Returns a row-major []bool (len W*H) at
// original resolution, or (nil, nil) when no seed yields a qualifying surface.
func (m *backgroundModel) backgroundSAM(img image.Image, prompt models.Prompt, r models.Runner) ([]bool, error) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	if w <= 0 || h <= 0 {
		return nil, nil
	}
	imgArea := float64(w * h)

	var best []bool
	var bestArea float64
	for _, seed := range samSeedPoints(w, h) {
		_, bitmaps, err := m.seg.InferMasks(img, models.Prompt{Points: []models.Point{seed}}, r)
		if err != nil {
			return nil, err
		}
		for i := range bitmaps {
			bm := bitmaps[i]
			if bm.W != w || bm.H != h || len(bm.Data) != w*h {
				continue
			}
			areaPx := bitmapArea(bm.Data)
			if areaPx/imgArea*100.0 >= samMaxAreaPct {
				continue // whole frame, not a surface
			}
			// Require a support surface that touches the border (a table/floor does), and
			// keep the LARGEST such mask across all seeds.
			if !touchesBorder(bm.Data, w, h) || !isBackgroundMask(areaPx, imgArea, true) {
				continue
			}
			if areaPx > bestArea {
				bestArea = areaPx
				best = append(best[:0:0], bm.Data...) // copy (bitmaps may be reused)
			}
		}
	}
	if best == nil || !anySet(best) {
		return nil, nil
	}
	return best, nil
}
