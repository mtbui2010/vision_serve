package explain

import (
	"strings"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// Letterboxed 100x50 original in a 100x100 input: scale 1, pad_y 25. Query 1's box runs past
// the right edge, so the detection's box is CLAMPED; the match must still find it.
func TestQueryForDetection(t *testing.T) {
	boxes := engine.F32([]float32{
		0.20, 0.50, 0.20, 0.20, // query 0: x 10..30, y 15..35 in the original
		0.95, 0.50, 0.20, 0.20, // query 1: x 85..105 -> clamped to 85..100
		0.50, 0.50, 0.40, 0.40, // query 2
	}, 1, 3, 4)
	logits := engine.F32(make([]float32, 3*2), 1, 3, 2)
	outs := []engine.Tensor{logits, boxes} // boxes found by shape, not position
	meta := models.PreprocessMeta{OrigWidth: 100, OrigHeight: 50, ScaleX: 1, ScaleY: 1, PadY: 25}
	dec := BoxDecode{InputW: 100, InputH: 100}

	for _, c := range []struct {
		bbox [4]float64
		want int
	}{
		{[4]float64{10, 15, 20, 20}, 0},
		{[4]float64{85, 15, 15, 20}, 1},
		{[4]float64{30, 5, 40, 40}, 2},
		{[4]float64{30.4, 5.3, 40, 40}, 2}, // a sub-pixel disagreement between two sessions
	} {
		got, err := QueryForDetection(outs, meta, dec, models.Detection{BBox: c.bbox, Class: "x"})
		if err != nil || got != c.want {
			t.Errorf("bbox %v: query %d, %v; want %d", c.bbox, got, err, c.want)
		}
	}

	// No query has this box: an error, never the nearest unrelated query.
	if _, err := QueryForDetection(outs, meta, dec, models.Detection{BBox: [4]float64{60, 30, 10, 10}, Class: "x"}); err == nil ||
		!strings.Contains(err.Error(), "no query") {
		t.Fatalf("unmatched box: err = %v, want a no-query error", err)
	}
}

// On a masked re-run the target is followed by class and box: the first detection of the re-run
// is another object, and a same-class detection elsewhere is not the target either.
func TestSameObjectScore(t *testing.T) {
	target := models.Detection{BBox: [4]float64{10, 10, 20, 20}, Class: "cat", Conf: 0.9}
	rerun := []models.Detection{
		{BBox: [4]float64{60, 60, 20, 20}, Class: "dog", Conf: 0.95}, // now first, other object
		{BBox: [4]float64{60, 10, 20, 20}, Class: "cat", Conf: 0.8},  // same class, elsewhere
		{BBox: [4]float64{11, 11, 20, 20}, Class: "cat", Conf: 0.4},  // the target, weaker
	}
	if got := SameObjectScore(rerun, target); got != float32(0.4) {
		t.Fatalf("score = %v, want 0.4 (the target's own confidence)", got)
	}
	if got := SameObjectScore(rerun[:2], target); got != 0 {
		t.Fatalf("target gone: score = %v, want 0", got)
	}
}
