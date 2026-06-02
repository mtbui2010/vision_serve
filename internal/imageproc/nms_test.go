package imageproc

import (
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
