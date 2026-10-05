package preprocess

import (
	"math"
	"strings"
	"testing"
)

// NaN or Inf in mean / std / pad only ever produce NaN (or zeroed) tensor values, so Validate
// refuses them — declared or legacy. A NaN pixel pad used to pass the [0,255] range check
// (NaN < 0 and NaN > 255 are both false).
func TestValidateRejectsNonFinite(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	base := Spec{Resize: Letterbox, Width: 8, Height: 8,
		Mean: []float32{0.5, 0.5, 0.5}, Std: []float32{0.2, 0.2, 0.2}}
	cases := map[string]func(*Spec){
		"letterbox pad NaN":     func(s *Spec) { s.PadValue = nan },
		"top_left_pad pad NaN":  func(s *Spec) { s.Resize = TopLeftPad; s.PadValue = nan },
		"long_side_pad pad Inf": func(s *Spec) { s.Resize = LongSidePad; s.PadValue = inf },
		"mean NaN":              func(s *Spec) { s.Mean = []float32{nan, 0.5, 0.5} },
		"std Inf":               func(s *Spec) { s.Std = []float32{inf, 0.2, 0.2} },
		"legacy mean NaN":       func(s *Spec) { s.Legacy = true; s.Mean = []float32{nan, 0.5, 0.5} },
		"crop_pct NaN":          func(s *Spec) { s.Resize = CenterCrop; s.PadValue = 0; s.CropPct = nan },
	}
	for name, mutate := range cases {
		s := base
		s.Mean, s.Std = append([]float32(nil), base.Mean...), append([]float32(nil), base.Std...)
		mutate(&s)
		err := s.Validate()
		if err == nil || !strings.Contains(err.Error(), "finite") {
			t.Errorf("%s: Validate() = %v, want a finite-number error", name, err)
		}
	}
	if err := base.Validate(); err != nil {
		t.Fatalf("the finite base spec must stay valid: %v", err)
	}
}

// crop_pct is timm's: the short side is resized to floor(size / crop_pct) and the centred size
// window kept — torchvision's Resize(256) + CenterCrop(224) at 0.875, CLIP's whole short side at
// 0 (or 1). Only center_crop takes it, and only in (0, 1].
func TestCenterCropSizeIsTimmAndTorchvision(t *testing.T) {
	for _, c := range []struct {
		w, h, W, H         int
		pct                float32
		rw, rh, offX, offY int
	}{
		{500, 375, 224, 224, 0.875, 341, 256, 58, 16}, // a typical ImageNet val photo
		{375, 500, 224, 224, 0.875, 256, 341, 16, 58},
		{500, 375, 224, 224, 0, 298, 224, 37, 0},         // CLIP: CoverSize
		{500, 375, 224, 224, 1, 298, 224, 37, 0},         // crop_pct 1 = CoverSize
		{640, 480, 224, 224, 0.9655172, 309, 232, 42, 4}, // 224/232 written as a float: still 232
	} {
		rw, rh, ox, oy := CenterCropSize(c.w, c.h, c.W, c.H, c.pct)
		if rw != c.rw || rh != c.rh || ox != c.offX || oy != c.offY {
			t.Errorf("%dx%d -> %dx%d crop_pct %g: got %dx%d off (%d,%d), want %dx%d off (%d,%d)",
				c.w, c.h, c.W, c.H, c.pct, rw, rh, ox, oy, c.rw, c.rh, c.offX, c.offY)
		}
	}
	for _, s := range []Spec{
		{Resize: Squash, Width: 224, Height: 224, CropPct: 0.875},
		{Resize: CenterCrop, Width: 224, Height: 224, CropPct: 1.5},
		{Resize: CenterCrop, Width: 224, Height: 224, CropPct: -0.5},
	} {
		if err := s.Validate(); err == nil || !strings.Contains(err.Error(), "crop_pct") {
			t.Errorf("%+v: Validate() = %v, want a crop_pct error", s, err)
		}
	}
}
