package textalign

import (
	"math"
	"reflect"
	"testing"

	"visionserve/internal/engine"
)

// The router owns no detector. Given the detector's label set and the words the caller asked
// for, it decides which head may name which column — and nothing else.
func TestRouteVocab(t *testing.T) {
	labels := []string{"cup", "banana", "N/A", "towel"}
	cases := []struct {
		name        string
		classes     []string
		closedOfCol []int
		openCol     []bool
	}{
		{
			name:        "every word is the detector's own",
			classes:     []string{"cup", "towel"},
			closedOfCol: []int{0, 3},
			openCol:     []bool{false, false},
		},
		{
			name:        "no word is the detector's own",
			classes:     []string{"zebra", "power strip"},
			closedOfCol: []int{-1, -1},
			openCol:     []bool{true, true},
		},
		{
			name:        "mixed: each head gets its own columns",
			classes:     []string{"zebra", "cup", "power strip", "towel"},
			closedOfCol: []int{-1, 0, -1, 3},
			openCol:     []bool{true, false, true, false},
		},
		{
			name:        "case and padding do not decide ownership",
			classes:     []string{"  CUP  "},
			closedOfCol: []int{0},
			openCol:     []bool{false},
		},
		{
			// "N/A" is the background column. Letting it claim a word would hand every
			// background query a name.
			name:        "the background label never owns a word",
			classes:     []string{"n/a"},
			closedOfCol: []int{-1},
			openCol:     []bool{true},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			closedOfCol, openCol := routeVocab(labels, c.classes)
			if !reflect.DeepEqual(closedOfCol, c.closedOfCol) {
				t.Errorf("closedOfCol = %v, want %v", closedOfCol, c.closedOfCol)
			}
			if !reflect.DeepEqual(openCol, c.openCol) {
				t.Errorf("openCol = %v, want %v", openCol, c.openCol)
			}
		})
	}
}

// won returns the column each query was given a real score in, and that score. Everything else
// must sit at the floor, because postprocess takes a per-query argmax.
func won(t *testing.T, logits []float32, q, c int) ([]int, []float32) {
	t.Helper()
	cols := make([]int, q)
	vals := make([]float32, q)
	for i := 0; i < q; i++ {
		cols[i], vals[i] = -1, gatedLogitFloor
		for k := 0; k < c; k++ {
			v := logits[i*c+k]
			if v != gatedLogitFloor {
				if cols[i] >= 0 {
					t.Fatalf("query %d has two non-floor columns (%d and %d)", i, cols[i], k)
				}
				cols[i], vals[i] = k, v
			}
		}
	}
	return cols, vals
}

// clsOf turns per-query detector logit rows into the accessor dualLogits takes.
func clsOf(rows [][]float32) func(int) []float32 {
	return func(i int) []float32 {
		if i < 0 || i >= len(rows) {
			return nil
		}
		return rows[i]
	}
}

// detector labels: [cup, banana, N/A, towel]; requested: [zebra, cup, power strip, towel]
var (
	dualClosedOfCol = []int{-1, 0, -1, 3}
	dualOpenCol     = []bool{true, false, true, false}
)

func TestDualLogits(t *testing.T) {
	const q, c = 3, 4
	// Head B likes "cup" (col 1) for q0 — it must not be allowed to use it; the closed head owns it.
	names := []float32{
		9, 8, 1, 0, // q0: best OPEN column is zebra(0)
		0, 0, 7, 0, // q1: power strip(2)
		1, 2, 3, 4, // q2: best open is power strip(2)
	}
	cls := [][]float32{
		{5.5, 0, 9, 0},    // q0: cup 5.5 is the best REAL class (N/A excluded upstream)
		{-20, -8, 0, -20}, // q1: both requested closed words sit far below its own best (banana -8)
		{0, 0, 0, 3.0},    // q2: towel 3.0
	}
	obj := []float32{5.5, -8, 3.0}

	out, err := dualLogits(names, obj, clsOf(cls), dualClosedOfCol, dualOpenCol, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	cols, vals := won(t, out, q, c)
	if want := []int{1, 2, 3}; !reflect.DeepEqual(cols, want) {
		t.Errorf("winning columns = %v, want %v (closed, head B, closed)", cols, want)
	}
	// The load-bearing property: whichever head named the query, the SCORE is the detector's
	// objectness. If this stops holding, postprocess is thresholding two different scales.
	if !reflect.DeepEqual(vals, obj) {
		t.Errorf("scores = %v, want the detector objectness %v", vals, obj)
	}
}

// The regression this design got wrong once, reproduced from the real failure: the tabletop
// vocabulary carries both "coffee" and "coffee can". A request for "coffee can" must not be lost
// because the detector marginally preferred "coffee", which the caller did not ask for. Before
// the fix this query was named "power strip" — and dropped entirely when no open word was asked
// for.
func TestDualLogitsNearSynonymInLabelSet(t *testing.T) {
	// detector labels [coffee, coffee can, N/A]; requested [coffee can, power strip]
	closedOfCol := []int{1, -1}
	openCol := []bool{false, true}
	const q, c = 1, 2
	// "coffee" (3.0) edges out "coffee can" (2.6); objectness is the global max, 3.0.
	cls := [][]float32{{3.0, 2.6, 0}}
	names := []float32{0, 9} // head B would happily call it a power strip

	out, err := dualLogits(names, []float32{3.0}, clsOf(cls), closedOfCol, openCol, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	cols, _ := won(t, out, q, c)
	if cols[0] != 0 {
		t.Errorf("named column %d, want 0 (\"coffee can\") — a near-synonym in the label set must "+
			"not hand the query to head B", cols[0])
	}
}

// Symmetric to the above: when the closed head's score for every requested word is below the
// claim threshold, it is not recognising any of them, and head B should name it.
func TestDualLogitsBelowThresholdGoesToHeadB(t *testing.T) {
	closedOfCol := []int{1, -1}
	openCol := []bool{false, true}
	const q, c = 1, 2
	cls := [][]float32{{8.0, -5.0, 0}} // requested "coffee can" scores -5 against a best of 8
	names := []float32{0, 9}

	out, err := dualLogits(names, []float32{8.0}, clsOf(cls), closedOfCol, openCol, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	cols, _ := won(t, out, q, c)
	if cols[0] != 1 {
		t.Errorf("named column %d, want 1 (head B) — the closed head recognised none of the "+
			"requested words", cols[0])
	}
}

// While the closed head CLAIMS a query, head B may not re-name it — head B is 16 points worse on
// those words. (Once the closed head declines, that exclusivity lapses; see the test below.)
func TestDualLogitsHeadBCannotClaimClosedWords(t *testing.T) {
	const q, c = 1, 2
	closedOfCol := []int{0, -1} // requested [cup(closed), zebra(open)]; detector [cup, banana]
	openCol := []bool{false, true}
	// The detector IS confident that this is a cup (5, its own best), so the closed head claims
	// the query and head B's enthusiasm for the same column is irrelevant.
	cls := [][]float32{{5, -20}}
	out, err := dualLogits([]float32{99, -99}, []float32{5}, clsOf(cls), closedOfCol, openCol, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	cols, _ := won(t, out, q, c)
	if cols[0] != 0 {
		t.Errorf("named column %d, want 0 — the closed head claimed this query", cols[0])
	}
}

// The regression that motivated making exclusivity conditional. The closed head declines (it
// committed to a label the caller did not ask for), so head B must be free to use ANY requested
// column, including one the closed head nominally owns. Confining it here answered "power strip"
// for a coffee can on a real image.
func TestDualLogitsHeadBFreeOnceClosedHeadDeclines(t *testing.T) {
	const q, c = 1, 2
	closedOfCol := []int{0, -1} // requested [coffee can(closed), power strip(open)]
	openCol := []bool{false, true}
	// detector: "coffee" (label 1, not requested) 3.0 beats "coffee can" (label 0) at -5.
	cls := [][]float32{{-5, 3.0}}
	// head B is sure it is a coffee can (col 0), mildly against power strip (col 1).
	out, err := dualLogits([]float32{8, 1}, []float32{3.0}, clsOf(cls), closedOfCol, openCol, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	cols, _ := won(t, out, q, c)
	if cols[0] != 0 {
		t.Errorf("named column %d, want 0 — once the closed head declines, head B may name a "+
			"word the closed head nominally owns", cols[0])
	}
}

// With no closed words requested at all, dual must reduce exactly to gated — the equivalence that
// lets the two modes be compared on the same footing.
func TestDualReducesToGatedWhenNothingIsClosed(t *testing.T) {
	const q, c = 3, 3
	names := []float32{1, 5, 2, 7, 0, 0, 0, 0, 3}
	obj := []float32{0.5, -1, 2}
	dual, err := dualLogits(names, obj, clsOf(nil), []int{-1, -1, -1}, []bool{true, true, true}, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogits: %v", err)
	}
	gated, err := gatedLogits(names, obj, q, c)
	if err != nil {
		t.Fatalf("gatedLogits: %v", err)
	}
	if !reflect.DeepEqual(dual, gated) {
		t.Errorf("dual and gated disagree with no closed words:\n dual  %v\n gated %v", dual, gated)
	}
}

func TestObjectnessArgmax(t *testing.T) {
	// 2 queries × 3 classes, the last of which is background.
	cls := engine.F32([]float32{
		0.1, 0.9, 5.0,
		-4, -9, 0,
	}, 1, 2, 3)
	real := []bool{true, true, false}
	vals, args, err := objectnessArgmax(cls, real, 2)
	if err != nil {
		t.Fatalf("objectnessArgmax: %v", err)
	}
	if args[0] != 1 || math.Abs(float64(vals[0]-0.9)) > 1e-6 {
		t.Errorf("q0 = (%v, %v), want (1, 0.9) — the background column must not win", args[0], vals[0])
	}
	if args[1] != 0 || math.Abs(float64(vals[1]-(-4))) > 1e-6 {
		t.Errorf("q1 = (%v, %v), want (0, -4)", args[1], vals[1])
	}
}

func TestObjectnessArgmaxNoRealClass(t *testing.T) {
	cls := engine.F32([]float32{7, 8}, 1, 1, 2)
	vals, args, err := objectnessArgmax(cls, []bool{false, false}, 1)
	if err != nil {
		t.Fatalf("objectnessArgmax: %v", err)
	}
	if args[0] != -1 || vals[0] != gatedLogitFloor {
		t.Errorf("got (%v, %v), want (-1, %v)", args[0], vals[0], gatedLogitFloor)
	}
}
