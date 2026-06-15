package background

import (
	"image"

	"visionserve/internal/models"
)

// backgroundAutomask runs MobileSAM's Automatic Mask Generator over the whole image and
// UNIONS the masks that qualify as support surfaces (large by area, or border-touching) —
// the inverse selection of the old foreground union. Slow (N² decoder calls); kept as the
// method-of-record / fallback. The grid is the foreground default (8×8) unless the request
// overrides it via grid_size; bg_max_area tunes the area threshold.
func (m *backgroundModel) backgroundAutomask(img image.Image, prompt models.Prompt, r models.Runner) ([]bool, error) {
	_, bitmaps, err := m.seg.InferMasks(img, models.Prompt{GridSize: prompt.GridSize}, r)
	if err != nil {
		return nil, err
	}
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	imgArea := float64(w * h)
	bgMaxPct, _ := m.bgThresholds(prompt)

	union := make([]bool, w*h)
	any := false
	for i := range bitmaps {
		bm := bitmaps[i]
		if bm.W != w || bm.H != h || len(bm.Data) != w*h {
			continue
		}
		areaPct := bitmapArea(bm.Data) / imgArea * 100.0
		isBg := areaPct >= bgMaxPct || (areaPct >= borderMinAreaPct && touchesBorder(bm.Data, w, h))
		if !isBg {
			continue
		}
		orInto(union, bm.Data)
		any = true
	}
	if !any {
		return nil, nil
	}
	return union, nil
}
