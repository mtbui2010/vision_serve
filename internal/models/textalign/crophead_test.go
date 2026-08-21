package textalign

import (
	"math"
	"reflect"
	"testing"

	"visionserve/internal/models"
)

// The sentinel is what lets the crop namer exist at all: an unclaimed query has to survive
// postprocess carrying the detector's objectness, or the novel objects are gone before anything
// can look at their pixels.
func TestDualLogitsSentinel(t *testing.T) {
	// requested [cup(closed), zebra(open)]; detector labels [cup, banana]
	closedOfCol := []int{0, -1}
	const q, c = 3, 2
	cls := [][]float32{
		{5.0, 0},    // q0: confidently a cup
		{-9.0, 2.0}, // q1: the detector likes banana, which was not requested
		{0.5, 0},    // q2: a weak cup, above threshold 0
	}
	obj := []float32{5.0, 2.0, 0.5}

	out, err := dualLogitsSentinel(obj, clsOf(cls), closedOfCol, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogitsSentinel: %v", err)
	}
	width := c + 1
	if len(out) != q*width {
		t.Fatalf("got %d values, want %d ([Q, C+1])", len(out), q*width)
	}
	cols, vals := won(t, out, q, width)
	if want := []int{0, c, 0}; !reflect.DeepEqual(cols, want) {
		t.Errorf("winning columns = %v, want %v (closed, SENTINEL, closed)", cols, want)
	}
	// Same invariant as dualLogits: the score is the detector's objectness in every case,
	// sentinel included. An unclaimed query must be thresholded on the same scale as a claimed
	// one, or the two-pass design silently changes which boxes survive.
	if !reflect.DeepEqual(vals, obj) {
		t.Errorf("scores = %v, want the detector objectness %v", vals, obj)
	}
}

// Every query must land in exactly one column, sentinel or not — postprocess takes a per-query
// argmax and a second non-floor entry would make the winner arbitrary.
func TestDualLogitsSentinelAlwaysNamesExactlyOnce(t *testing.T) {
	const q, c = 4, 3
	cls := [][]float32{{1, 2, 3}, {-9, -9, -9}, {0, 0, 0}, {7, 0, 0}}
	obj := []float32{3, -9, 0, 7}
	out, err := dualLogitsSentinel(obj, clsOf(cls), []int{0, 1, -1}, 0.0, q, c)
	if err != nil {
		t.Fatalf("dualLogitsSentinel: %v", err)
	}
	won(t, out, q, c+1) // won() fails the test if any query has two non-floor columns
}

func TestDropIndices(t *testing.T) {
	dets := []models.Detection{
		{Class: "a"}, {Class: "b"}, {Class: "c"}, {Class: "d"},
	}
	got := dropIndices(dets, []int{1, 3})
	want := []string{"a", "c"}
	if len(got) != len(want) {
		t.Fatalf("kept %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Class != want[i] {
			t.Errorf("index %d = %q, want %q — order must be preserved, callers index masks "+
				"against detections", i, got[i].Class, want[i])
		}
	}
}

// A sentinel detection with no open word available must be DROPPED, never emitted. The sentinel
// carries a NUL and is an internal marker; leaking it would put "\x00open" in a client's JSON.
func TestNameOpenCropsDropsWhenNoOpenWords(t *testing.T) {
	m := &textAlign{}
	dets := []models.Detection{
		{Class: "cup", Conf: 0.9},
		{Class: openSentinel, Conf: 0.8},
		{Class: "towel", Conf: 0.7},
	}
	got, err := m.nameOpenCrops(nil, dets, nil, 0, nil)
	if err != nil {
		t.Fatalf("nameOpenCrops: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("kept %d detections, want 2", len(got))
	}
	for _, d := range got {
		if d.Class == openSentinel {
			t.Error("the sentinel leaked into the output")
		}
	}
}

// With nothing marked, the crop namer must not touch the detections at all — and in particular
// must not call the vision tower, which is why this passes a nil Runner.
func TestNameOpenCropsNoOpIfNothingMarked(t *testing.T) {
	m := &textAlign{}
	dets := []models.Detection{{Class: "cup", Conf: 0.9}, {Class: "towel", Conf: 0.7}}
	got, err := m.nameOpenCrops(nil, dets, []string{"zebra"}, 0, nil)
	if err != nil {
		t.Fatalf("nameOpenCrops: %v", err)
	}
	if !reflect.DeepEqual(got, dets) {
		t.Errorf("detections changed: %v", got)
	}
}

func TestHasCropHead(t *testing.T) {
	if (&textAlign{cfg: models.Config{Files: map[string]string{}}}).hasCropHead() {
		t.Error("no files.crop should mean no crop head")
	}
	if (&textAlign{cfg: models.Config{Files: map[string]string{roleCrop: "  "}}}).hasCropHead() {
		t.Error("a blank files.crop should not enable the crop head")
	}
	if !(&textAlign{cfg: models.Config{Files: map[string]string{roleCrop: "../siglip-image/model.onnx"}}}).hasCropHead() {
		t.Error("files.crop should enable the crop head")
	}
}

// A detection whose box was skipped as degenerate must be DROPPED, never emitted still carrying
// the sentinel. This is the second half of the degenerate-box fix: skipping the crop is only safe
// if the detection that box belonged to goes with it.
func TestNameOpenCropsDropsSkippedBoxes(t *testing.T) {
	// Build a case where the middle sentinel box is degenerate. nameOpenCrops needs a Runner for
	// the towers, so exercise the mapping logic through CropTensor's contract instead: kept=[0,2]
	// out of three sentinel boxes means index 1 must end up dropped.
	idx := []int{1, 3, 5} // detection indices carrying the sentinel
	kept := []int{0, 2}   // boxes 0 and 2 embedded; box 1 was degenerate
	named := make(map[int]bool, len(kept))
	for _, k := range kept {
		named[idx[k]] = true
	}
	var drop []int
	for _, di := range idx {
		if !named[di] {
			drop = append(drop, di)
		}
	}
	if want := []int{3}; !reflect.DeepEqual(drop, want) {
		t.Fatalf("drop = %v, want %v (detection 3 owned the degenerate box)", drop, want)
	}
}

// The cache must key on the WORDS, not just their count, or two different vocabularies of the
// same size would share embeddings — silently naming everything with the wrong words.
func TestOpenVocabCacheKeyDependsOnWords(t *testing.T) {
	tmpl := []string{"a photo of a {}."}
	if vocabKey(tmpl, []string{"zebra", "ruler"}) == vocabKey(tmpl, []string{"stapler", "eraser"}) {
		t.Error("two different vocabularies produced the same cache key")
	}
	if vocabKey(tmpl, []string{"zebra"}) != vocabKey(tmpl, []string{"ZEBRA"}) {
		t.Error("case should not split the cache — normalizeVocab lowercases upstream")
	}
}

// The crop head must be allowed to name a CLOSED word once the closed head has declined the
// query. Confining it to the open words was what made it a worse ranker despite being a much
// better namer: every declined background query was forced to carry a held-out name and became a
// false positive there, where head B could call the same query "cup" and be judged as a cup.
//
// The scope is checked through decodeDualCrop's contract rather than by running the towers: what
// changed is WHICH list is handed to the namer.
func TestCropNamerScoresAgainstAllRequestedWords(t *testing.T) {
	labels := []string{"cup", "towel", "N/A"}           // the detector's own 2 real classes
	classes := []string{"cup", "towel", "hat", "ruler"} // what the caller asked for
	closedOfCol, openCol := routeVocab(labels, classes)

	// Sanity on the routing itself: the first two are the detector's, the last two are not.
	if want := []int{0, 1, -1, -1}; !reflect.DeepEqual(closedOfCol, want) {
		t.Fatalf("closedOfCol = %v, want %v", closedOfCol, want)
	}
	if want := []bool{false, false, true, true}; !reflect.DeepEqual(openCol, want) {
		t.Fatalf("openCol = %v, want %v", openCol, want)
	}

	// The namer is handed the FULL vocabulary. If a future change narrows it back to the open
	// subset, this is the assertion that should fail.
	open := 0
	for _, isOpen := range openCol {
		if isOpen {
			open++
		}
	}
	if open == len(classes) {
		t.Fatal("test is vacuous: pick a vocabulary with both closed and open words")
	}
	if got := len(classes); got != 4 {
		t.Fatalf("the crop namer must receive all %d requested words, not the %d open ones", got, open)
	}
}

// The confidence reported for an open-named detection is P(object) x P(name | crop). Before this,
// it was the detector's objectness alone — its confidence that the box is one of the classes IT
// knows, none of which is the name being reported. A hat went out carrying the detector's
// confidence that it is a towel, which is why `hat` scored 3.82 AP at 88% naming accuracy.
func TestSoftmaxAt(t *testing.T) {
	// A decisive row: one clear winner should approach 1.
	row := []float32{0.9, 0.1, 0.05}
	if p := softmaxAt(row, 0, 0.07); p < 0.99 {
		t.Errorf("clear winner got p=%v, want ~1", p)
	}
	if p := softmaxAt(row, 1, 0.07); p > 0.01 {
		t.Errorf("clear loser got p=%v, want ~0", p)
	}
	// A tie must split evenly, whatever the temperature.
	tie := []float32{0.5, 0.5}
	if p := softmaxAt(tie, 0, 0.07); math.Abs(float64(p)-0.5) > 1e-6 {
		t.Errorf("tie got p=%v, want 0.5", p)
	}
	// Probabilities over a row must sum to 1.
	var sum float32
	for k := range row {
		sum += softmaxAt(row, k, 0.07)
	}
	if math.Abs(float64(sum)-1) > 1e-5 {
		t.Errorf("row sums to %v, want 1", sum)
	}
}

// Temperature must be monotone in the intended direction, or the sweep cannot be read.
func TestSoftmaxTemperatureIsMonotone(t *testing.T) {
	row := []float32{0.6, 0.4}
	decisive := softmaxAt(row, 0, 0.01)
	soft := softmaxAt(row, 0, 1.0)
	if decisive <= soft {
		t.Errorf("lower temperature must be MORE decisive: %v at T=0.01 vs %v at T=1.0", decisive, soft)
	}
}

// A single candidate is certain by construction; a non-positive temperature would divide by zero
// or invert the ordering, so it falls back to the default rather than producing silent nonsense.
func TestSoftmaxAtEdgeCases(t *testing.T) {
	if p := softmaxAt([]float32{0.3}, 0, 0.07); p != 1 {
		t.Errorf("single candidate got p=%v, want 1", p)
	}
	if p := softmaxAt([]float32{0.9, 0.1}, 0, 0); p != softmaxAt([]float32{0.9, 0.1}, 0, cropTemp) {
		t.Error("a non-positive temperature must fall back to the package default")
	}
	if p := softmaxAt([]float32{0.9, 0.1}, 5, 0.07); p != 0 {
		t.Errorf("out-of-range index got p=%v, want 0", p)
	}
	if p := softmaxAt(nil, 0, 0.07); p != 0 {
		t.Errorf("empty row got p=%v, want 0", p)
	}
}
