package background

import (
	"image"
	"testing"

	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"
)

// fakeSeg hands back fixed bitmaps for every prompt, and counts calls.
type fakeSeg struct {
	bms   []mobilesam.MaskBitmap
	calls int
}

func (f *fakeSeg) InferMasks(image.Image, models.Prompt, models.Runner) ([]models.Mask, []mobilesam.MaskBitmap, error) {
	f.calls++
	return make([]models.Mask, len(f.bms)), f.bms, nil
}

// rectMask is a w×h bitmap with the rectangle [x0,x1)×[y0,y1) set.
func rectMask(w, h, x0, y0, x1, y1 int) mobilesam.MaskBitmap {
	d := make([]bool, w*h)
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			d[y*w+x] = true
		}
	}
	return mobilesam.MaskBitmap{Data: d, W: w, H: h}
}

// B10c: fg_min_area was parsed and then ignored by every method, although the client docs say it
// drops masks smaller than that percentage on the sam / automask methods. An 8 % border strip
// qualifies as a surface by default; with fg_min_area=10 it must not.
func TestFgMinAreaHonouredBySAMAndAutomask(t *testing.T) {
	const w, h = 100, 100
	strip := rectMask(w, h, 0, 92, 100, 100) // 8 % of the frame, touching the bottom edge
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for _, method := range []string{methodSAM, methodAutomask} {
		m := &backgroundModel{seg: &fakeSeg{bms: []mobilesam.MaskBitmap{strip}}, hasSAM: true}
		run := m.backgroundSAM
		if method == methodAutomask {
			run = m.backgroundAutomask
		}
		got, err := run(img, models.Prompt{}, nil)
		if err != nil || got == nil {
			t.Fatalf("%s default: got %v, err %v; want the 8%% strip as the surface", method, got != nil, err)
		}
		got, err = run(img, models.Prompt{FgMinArea: 10}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got != nil {
			t.Errorf("%s: fg_min_area=10 kept an 8%% mask", method)
		}
	}
}

// bg_max_area keeps working on automask (it always did) and now also reaches the sam method's
// classifier through the same helper. An interior 30 % blob is background only when
// bg_max_area <= 30.
func TestBgMaxAreaHonouredByAutomask(t *testing.T) {
	const w, h = 100, 100
	blob := rectMask(w, h, 20, 20, 80, 70) // 30 %, not touching the border
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	m := &backgroundModel{seg: &fakeSeg{bms: []mobilesam.MaskBitmap{blob}}, hasSAM: true}
	if got, _ := m.backgroundAutomask(img, models.Prompt{}, nil); got != nil {
		t.Error("default bg_max_area (50) classified a 30% interior blob as background")
	}
	if got, _ := m.backgroundAutomask(img, models.Prompt{BgMaxArea: 25}, nil); got == nil {
		t.Error("bg_max_area=25 did not classify a 30% blob as background")
	}
}

func TestIsBackgroundMaskDefaultsUnchanged(t *testing.T) {
	bgMax, minPct := (&backgroundModel{}).bgThresholds(models.Prompt{})
	for _, c := range []struct {
		pct    float64
		border bool
		want   bool
	}{{60, false, true}, {50, false, true}, {49, false, false}, {6, true, true}, {4, true, false}, {0.5, true, false}} {
		if got := isBackgroundMask(c.pct, 100, c.border, bgMax, minPct); got != c.want {
			t.Errorf("area %.1f%% border=%v: got %v, want %v", c.pct, c.border, got, c.want)
		}
	}
}
