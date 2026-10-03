package imageproc

import (
	"math/rand"
	"sort"
	"testing"

	"visionserve/pkg/api"
)

func TestNMSSuppressesOverlapSameClass(t *testing.T) {
	dets := []api.Detection{
		{BBox: [4]float64{0, 0, 100, 100}, Class: "cat", Conf: 0.9},
		{BBox: [4]float64{10, 10, 100, 100}, Class: "cat", Conf: 0.8}, // high IoU -> suppressed
	}
	keep := NMS(dets, 0.5)
	if len(keep) != 1 || keep[0].Conf != 0.9 {
		t.Fatalf("want 1 box kept with conf 0.9, got %+v", keep)
	}
}

func TestNMSKeepsDifferentClass(t *testing.T) {
	dets := []api.Detection{
		{BBox: [4]float64{0, 0, 100, 100}, Class: "cat", Conf: 0.9},
		{BBox: [4]float64{0, 0, 100, 100}, Class: "dog", Conf: 0.8}, // different class -> kept
	}
	if keep := NMS(dets, 0.5); len(keep) != 2 {
		t.Fatalf("want 2 boxes kept for different classes, got %d", len(keep))
	}
}

func TestIoUNoOverlap(t *testing.T) {
	if v := iou([4]float64{0, 0, 10, 10}, [4]float64{20, 20, 10, 10}); v != 0 {
		t.Fatalf("IoU of disjoint boxes want 0, got %v", v)
	}
}

// nmsBenchDets builds n random boxes (fixed seed) on a 640x640 canvas, roughly the
// SCRFD-10GF proposal count (16 800 at 640x640) when n = 16800. Mostly small, scattered
// boxes, so few suppress each other — the worst case for the old all-pairs inner loop.
func nmsBenchDets(n int) []api.Detection {
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
	dets := nmsBenchDets(16800)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NMS(dets, 0.4)
	}
}

// naiveNMS is the previous all-pairs implementation, kept as the oracle.
func naiveNMS(dets []api.Detection, iouThresh float64) []api.Detection {
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
			if dets[j].Class == dets[i].Class && iou(dets[i].BBox, dets[j].BBox) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

// TestNMSMatchesNaive: the triangular scan (n < nmsGridMin) and the grid-indexed scan
// (n >= nmsGridMin) must keep exactly the boxes the all-pairs version kept, in the same
// order — on dense overlapping data with two classes, plus a giant box, a zero-area box
// and boxes with negative coordinates.
func TestNMSMatchesNaive(t *testing.T) {
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
			got, want := NMS(dets, thr), naiveNMS(dets, thr)
			if len(got) != len(want) {
				t.Fatalf("n=%d thr %v: kept %d, naive kept %d", n, thr, len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("n=%d thr %v: box %d differs: %+v vs %+v", n, thr, i, got[i], want[i])
				}
			}
		}
	}
}

// TestNMSMatchesNaiveBench: the benchmark workload (scattered, grid path) agrees too.
func TestNMSMatchesNaiveBench(t *testing.T) {
	dets := nmsBenchDets(4000)
	got, want := NMS(dets, 0.4), naiveNMS(dets, 0.4)
	if len(got) != len(want) {
		t.Fatalf("kept %d, naive kept %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("box %d differs", i)
		}
	}
}
