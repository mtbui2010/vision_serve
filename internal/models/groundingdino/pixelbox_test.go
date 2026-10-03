package groundingdino

import (
	"math/rand"
	"testing"

	"visionserve/internal/models"
)

// refClampSpan is the per-axis clamp postprocess used before it moved onto geom.Clamp, verbatim.
func refClampSpan(x, w, max float64) (float64, float64) {
	if x < 0 {
		w += x
		x = 0
	}
	if x > max {
		x = max
	}
	if x+w > max {
		w = max - x
	}
	if w < 0 {
		w = 0
	}
	return x, w
}

// The box a detection reports must not move by one ulp: pixelBox is held to the old arithmetic
// on random boxes, on boxes hanging off every edge, and on corners past the far edge (the case
// geom.Clamp alone would get wrong).
func TestPixelBoxMatchesReference(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	vals := []float32{-0.5, -0.01, 0, 1e-7, 0.25, 0.5, 0.999999, 1, 1.0000001, 1.5, 2}
	check := func(bb []float32, w, h int) {
		cx, cy, bw, bh := float64(bb[0]), float64(bb[1]), float64(bb[2]), float64(bb[3])
		x, ww := refClampSpan((cx-bw/2)*float64(w), bw*float64(w), float64(w))
		y, hh := refClampSpan((cy-bh/2)*float64(h), bh*float64(h), float64(h))
		if got, want := pixelBox(bb, w, h), [4]float64{x, y, ww, hh}; got != want {
			t.Fatalf("pixelBox(%v, %dx%d) = %v, reference %v", bb, w, h, got, want)
		}
	}
	for _, sz := range [][2]int{{640, 480}, {848, 153}, {1, 1}} {
		for _, a := range vals {
			for _, b := range vals {
				check([]float32{a, b, b, a}, sz[0], sz[1])
			}
		}
		for i := 0; i < 50000; i++ {
			check([]float32{r.Float32()*1.4 - 0.2, r.Float32()*1.4 - 0.2, r.Float32() * 1.2, r.Float32() * 1.2}, sz[0], sz[1])
		}
	}
}

// One precedence for every GroundingDINO pipeline: request > manifest > built-in default.
func TestThresholdsPrecedence(t *testing.T) {
	cases := []struct {
		mb, mt    float64
		p         models.Prompt
		box, text float64
	}{
		{0, 0, models.Prompt{}, DefaultBoxThresh, DefaultTextThresh},
		{0.35, 0.2, models.Prompt{}, 0.35, 0.2},
		{0.35, 0.2, models.Prompt{BoxThresh: 0.1}, 0.1, 0.2},
		{0.35, 0, models.Prompt{TextThresh: 0.4}, 0.35, 0.4},
		{-1, -1, models.Prompt{BoxThresh: -3}, DefaultBoxThresh, DefaultTextThresh},
	}
	for _, c := range cases {
		if b, x := Thresholds(c.mb, c.mt, c.p); b != c.box || x != c.text {
			t.Errorf("Thresholds(%v, %v, %+v) = %v, %v; want %v, %v", c.mb, c.mt, c.p, b, x, c.box, c.text)
		}
	}
}

func TestVocabPath(t *testing.T) {
	if got := VocabPath("/m/grounding-dino/model-fixedmask.onnx", "/m/grounded-sam"); got != "/m/grounding-dino/vocab.txt" {
		t.Errorf("VocabPath = %q", got)
	}
	if got := VocabPath("", "/m/grounded-sam"); got != "/m/grounded-sam/vocab.txt" {
		t.Errorf("VocabPath without weights = %q", got)
	}
}
