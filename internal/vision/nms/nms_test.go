package nms

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"visionserve/internal/vision/geom"
	"visionserve/pkg/api"
)

func TestSuppressesOverlapSameClass(t *testing.T) {
	dets := []api.Detection{
		{BBox: [4]float64{0, 0, 100, 100}, Class: "cat", Conf: 0.9},
		{BBox: [4]float64{10, 10, 100, 100}, Class: "cat", Conf: 0.8}, // high IoU -> suppressed
	}
	keep := Detections(dets, Options{IoU: 0.5})
	if len(keep) != 1 || keep[0].Conf != 0.9 {
		t.Fatalf("want 1 box kept with conf 0.9, got %+v", keep)
	}
}

func TestKeepsDifferentClassUnlessAgnostic(t *testing.T) {
	dets := []api.Detection{
		{BBox: [4]float64{0, 0, 100, 100}, Class: "cat", Conf: 0.9},
		{BBox: [4]float64{0, 0, 100, 100}, Class: "dog", Conf: 0.8},
	}
	if keep := Detections(dets, Options{IoU: 0.5}); len(keep) != 2 {
		t.Fatalf("class-aware: want 2 boxes kept for different classes, got %d", len(keep))
	}
	if keep := Detections(dets, Options{IoU: 0.5, ClassAgnostic: true}); len(keep) != 1 {
		t.Fatalf("class-agnostic: want 1 box kept, got %d", len(keep))
	}
}

// A tight box inside a large one: plain IoU is low (0.04) so only Containment suppresses.
func TestContainment(t *testing.T) {
	dets := []api.Detection{
		{BBox: [4]float64{40, 40, 20, 20}, Class: "o", Conf: 0.9},
		{BBox: [4]float64{0, 0, 100, 100}, Class: "o", Conf: 0.8},
	}
	if keep := Detections(dets, Options{IoU: 0.5}); len(keep) != 2 {
		t.Fatalf("IoU only: want 2 kept, got %d", len(keep))
	}
	if keep := Detections(dets, Options{IoU: 0.5, Containment: true}); len(keep) != 1 || keep[0].Conf != 0.9 {
		t.Fatalf("containment: want the tight box only, got %+v", keep)
	}
}

func TestEmptyAndSortedOutput(t *testing.T) {
	if got := Detections(nil, Options{IoU: 0.5}); len(got) != 0 {
		t.Fatalf("nil input: got %v", got)
	}
	dets := []api.Detection{
		{BBox: [4]float64{0, 0, 10, 10}, Class: "a", Conf: 0.2},
		{BBox: [4]float64{50, 50, 10, 10}, Class: "a", Conf: 0.7},
		{BBox: [4]float64{90, 90, 10, 10}, Class: "a", Conf: 0.7},
	}
	keep := Detections(dets, Options{IoU: 0.5})
	if len(keep) != 3 || keep[0] != dets[1] || keep[1] != dets[2] || keep[2] != dets[0] {
		t.Fatalf("want descending conf with stable ties, got %+v", keep)
	}
}

// benchDets builds n random boxes (fixed seed) on a 640x640 canvas, roughly the
// SCRFD-10GF proposal count (16 800 at 640x640) when n = 16800. Mostly small, scattered
// boxes, so few suppress each other — the worst case for an all-pairs inner loop.
func benchDets(n int) []api.Detection {
	r := rand.New(rand.NewSource(1))
	dets := make([]api.Detection, n)
	for i := range dets {
		w, h := 8+r.Float64()*40, 8+r.Float64()*40
		dets[i] = api.Detection{
			BBox:  [4]float64{r.Float64() * (640 - w), r.Float64() * (640 - h), w, h},
			Class: "face",
			Conf:  r.Float64(),
		}
	}
	return dets
}

func BenchmarkNMS16800(b *testing.B) {
	dets := benchDets(16800)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Detections(dets, Options{IoU: 0.4})
	}
}

// naive is the textbook all-pairs implementation, kept as the oracle. Containment uses the
// OWLv2 formula: max(IoU, inter/min(area)) > thr.
func naive(dets []api.Detection, o Options) []api.Detection {
	idx := make([]int, len(dets))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return dets[idx[a]].Conf > dets[idx[b]].Conf })
	suppressed := make([]bool, len(dets))
	var keep []api.Detection
	for _, i := range idx {
		if suppressed[i] {
			continue
		}
		keep = append(keep, dets[i])
		for _, j := range idx {
			if j == i || suppressed[j] {
				continue
			}
			if !o.ClassAgnostic && dets[j].Class != dets[i].Class {
				continue
			}
			ov := geom.IoU(dets[i].BBox, dets[j].BBox)
			if o.Containment {
				a, b := dets[i].BBox, dets[j].BBox
				iw := math.Min(a[0]+a[2], b[0]+b[2]) - math.Max(a[0], b[0])
				ih := math.Min(a[1]+a[3], b[1]+b[3]) - math.Max(a[1], b[1])
				if m := math.Min(a[2]*a[3], b[2]*b[3]); iw > 0 && ih > 0 && m > 0 {
					ov = math.Max(ov, iw*ih/m)
				}
			}
			if ov > o.IoU {
				suppressed[j] = true
			}
		}
	}
	if o.TopK > 0 && len(keep) > o.TopK {
		keep = keep[:o.TopK]
	}
	return keep
}

// TestMatchesNaive: the triangular scan (n < gridMin) and the grid-indexed scan
// (n >= gridMin) must keep exactly the boxes the all-pairs version keeps, in the same
// order — on dense overlapping data with two classes, plus a giant box, a zero-area box
// and boxes with negative coordinates — for every option combination.
func TestMatchesNaive(t *testing.T) {
	for _, n := range []int{300, 3000} {
		r := rand.New(rand.NewSource(int64(n)))
		dets := make([]api.Detection, n)
		for i := range dets {
			w, h := 10+r.Float64()*60, 10+r.Float64()*60
			cls := "a"
			if r.Intn(2) == 0 {
				cls = "b"
			}
			dets[i] = api.Detection{BBox: [4]float64{r.Float64()*400 - 50, r.Float64()*300 - 50, w, h}, Class: cls, Conf: float64(r.Intn(1000)) / 1000}
		}
		dets[3] = api.Detection{BBox: [4]float64{-100, -100, 600, 500}, Class: "a", Conf: 0.4}
		dets[4] = api.Detection{BBox: [4]float64{20, 20, 0, 0}, Class: "b", Conf: 0.99}
		for _, thr := range []float64{0.3, 0.5, 0.7} {
			for _, o := range []Options{
				{IoU: thr},
				{IoU: thr, ClassAgnostic: true},
				{IoU: thr, Containment: true},
				{IoU: thr, ClassAgnostic: true, Containment: true, TopK: 17},
				{IoU: thr, TopK: 1},
			} {
				got, want := Detections(dets, o), naive(dets, o)
				if len(got) != len(want) {
					t.Fatalf("n=%d %+v: kept %d, naive kept %d", n, o, len(got), len(want))
				}
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("n=%d %+v: box %d differs: %+v vs %+v", n, o, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestMatchesNaiveBench: the benchmark workload (scattered, grid path) agrees too.
func TestMatchesNaiveBench(t *testing.T) {
	dets := benchDets(4000)
	o := Options{IoU: 0.4}
	got, want := Detections(dets, o), naive(dets, o)
	if len(got) != len(want) {
		t.Fatalf("kept %d, naive kept %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("box %d differs", i)
		}
	}
}

// An infinite coordinate among >= gridMin boxes (a model emitting Inf) used to build a grid with
// zero cells and panic with "index out of range [-1]"; it now takes the triangular scan, which
// keeps what the all-pairs version keeps.
func TestNonFiniteBoxesOnTheGridPath(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	base := make([]api.Detection, 600)
	for i := range base {
		base[i] = api.Detection{BBox: [4]float64{r.Float64() * 400, r.Float64() * 300, 10 + r.Float64()*40, 10 + r.Float64()*40},
			Class: "a", Conf: float64(r.Intn(1000)) / 1000}
	}
	for name, bad := range map[string][4]float64{
		"+Inf width": {10, 10, math.Inf(1), 20},
		"-Inf x":     {math.Inf(-1), 10, 20, 20},
		"+Inf y":     {10, math.Inf(1), 20, 20},
	} {
		dets := append([]api.Detection(nil), base...)
		dets[5].BBox = bad
		for _, o := range []Options{{IoU: 0.5}, {IoU: 0.5, Containment: true}} {
			got, want := Detections(dets, o), naive(dets, o)
			if len(got) != len(want) {
				t.Fatalf("%s %+v: kept %d, naive kept %d", name, o, len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("%s %+v: box %d differs: %+v vs %+v", name, o, i, got[i], want[i])
				}
			}
		}
	}
}
