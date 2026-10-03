package background

import (
	"image"
	"sync"

	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"
	"visionserve/internal/vision/mask"
)

// maskEacher is a segmenter that hands each mask's bitmap to fn as soon as it is final
// (MobileSAM's InferMasksEach); fn may run concurrently and must not keep b.Data.
type maskEacher interface {
	InferMasksEach(img image.Image, prompt models.Prompt, r models.Runner, fn func(b mobilesam.MaskBitmap) any) ([]any, error)
}

// backgroundAutomask runs MobileSAM's Automatic Mask Generator over the whole image and
// UNIONS the masks that qualify as support surfaces (large by area, or border-touching) —
// the inverse selection of the old foreground union. Slow (N² decoder calls); kept as the
// method-of-record / fallback. The grid is the foreground default (8×8) unless the request
// overrides it via grid_size; bg_max_area / fg_min_area tune the area thresholds.
//
// On a segmenter that streams (maskEacher) each mask is tested and ORed in as soon as it is
// final, so the full-resolution bitmaps are never all held at once; OR is order-independent,
// so the union is exactly the one built from the returned bitmaps.
func (m *backgroundModel) backgroundAutomask(img image.Image, prompt models.Prompt, r models.Runner) ([]bool, error) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	imgArea := float64(w * h)
	bgMaxPct, minPct := m.bgThresholds(prompt)
	qualifies := func(bm mobilesam.MaskBitmap) bool {
		if bm.W != w || bm.H != h || len(bm.Data) != w*h {
			return false
		}
		return isBackgroundMask(bitmapArea(bm.Data), imgArea, touchesBorder(bm.Data, w, h), bgMaxPct, minPct)
	}
	segPrompt := models.Prompt{GridSize: prompt.GridSize}

	if me, ok := m.seg.(maskEacher); ok {
		var mu sync.Mutex
		var union []bool // nil until a mask qualifies
		_, err := me.InferMasksEach(img, segPrompt, r, func(bm mobilesam.MaskBitmap) any {
			if qualifies(bm) {
				mu.Lock()
				if union == nil {
					union = make([]bool, w*h)
				}
				mask.Or(union, bm.Data)
				mu.Unlock()
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		return union, nil
	}

	_, bitmaps, err := m.seg.InferMasks(img, segPrompt, r)
	if err != nil {
		return nil, err
	}
	union := make([]bool, w*h)
	any := false
	for i := range bitmaps {
		if !qualifies(bitmaps[i]) {
			continue
		}
		mask.Or(union, bitmaps[i].Data)
		any = true
	}
	if !any {
		return nil, nil
	}
	return union, nil
}
