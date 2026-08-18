package textalign

import (
	"math"
	"testing"
)

func TestClaimThreshold(t *testing.T) {
	cases := []struct {
		name string
		in   float64
		want float32
	}{
		{"unset falls back to the default", 0, dualClaimThresh},
		{"negative is not a probability", -0.5, dualClaimThresh},
		{"1 would mean +inf", 1, dualClaimThresh},
		{"above 1 is not a probability", 2, dualClaimThresh},
		{"0.5 is the origin", 0.5, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := claimThreshold(c.in); math.Abs(float64(got-c.want)) > 1e-6 {
				t.Errorf("claimThreshold(%v) = %v, want %v", c.in, got, c.want)
			}
		})
	}
	// Monotone: a stricter probability must demand a higher logit, or the knob is backwards.
	if claimThreshold(0.1) >= claimThreshold(0.9) {
		t.Errorf("claimThreshold is not increasing: %v at p=0.1 vs %v at p=0.9",
			claimThreshold(0.1), claimThreshold(0.9))
	}
}

// The regression that made `dual` stop doing open-vocabulary entirely. When the prompt contains
// the whole base vocabulary, the closed head's best REQUESTED score and its best score OVERALL are
// the same number by construction. The old relative rule (bestC >= obj - margin) was therefore
// always true, head B was never consulted, and |V|=78 produced statistics byte-identical to
// |V|=22. An absolute threshold has to leave head B reachable in exactly that situation.
func TestDualLogitsOpenHeadStillReachableOnSupersetPrompt(t *testing.T) {
	// requested = [cup(closed), towel(closed), zebra(open)]; the two closed words are the
	// detector's ENTIRE real vocabulary, so obj is always one of them.
	closedOfCol := []int{0, 1, -1}
	openCol := []bool{false, false, true}
	const q, c = 2, 3

	// q0: a real cup — the closed head is confident (logit 4).
	// q1: background — every class logit is low, and they are close together, which is exactly
	//     what defeated the relative rule.
	cls := [][]float32{
		{4.0, 3.6},
		{-3.0, -3.2},
	}
	obj := []float32{4.0, -3.0}
	names := []float32{
		0, 0, 1, // head B: zebra
		0, 0, 9, // head B: zebra, confidently
	}

	out, err := dualLogits(names, obj, clsOf(cls), closedOfCol, openCol, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	cols, _ := won(t, out, q, c)
	if cols[0] != 0 {
		t.Errorf("q0 named column %d, want 0 — the closed head is confident here", cols[0])
	}
	if cols[1] != 2 {
		t.Errorf("q1 named column %d, want 2 (head B) — with the whole base vocabulary in the "+
			"prompt the open head must still be reachable", cols[1])
	}
}

// Raising the threshold must hand more queries to the open head, never fewer. This is the
// property the sweep depends on; if it does not hold the knob cannot be tuned.
func TestDualLogitsThresholdIsMonotone(t *testing.T) {
	closedOfCol := []int{0, -1}
	openCol := []bool{false, true}
	const q, c = 1, 2
	cls := [][]float32{{1.0, 0}}
	names := []float32{0, 5}

	lo, err := dualLogits(names, []float32{1.0}, clsOf(cls), closedOfCol, openCol, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	hi, err := dualLogits(names, []float32{1.0}, clsOf(cls), closedOfCol, openCol, 2.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	loCols, _ := won(t, lo, q, c)
	hiCols, _ := won(t, hi, q, c)
	if loCols[0] != 0 {
		t.Errorf("at threshold 0.0 the closed head (logit 1.0) should claim, got column %d", loCols[0])
	}
	if hiCols[0] != 1 {
		t.Errorf("at threshold 2.0 the closed head (logit 1.0) should decline, got column %d", hiCols[0])
	}
}
