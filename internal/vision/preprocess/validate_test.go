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
