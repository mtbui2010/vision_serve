package imageproc

import (
	"testing"

	"visionserve/pkg/api"
)

func TestNMSSuppressesOverlapSameClass(t *testing.T) {
	dets := []api.Detection{
		{BBox: [4]float64{0, 0, 100, 100}, Class: "cat", Conf: 0.9},
		{BBox: [4]float64{10, 10, 100, 100}, Class: "cat", Conf: 0.8}, // IoU cao -> bị suppress
	}
	keep := NMS(dets, 0.5)
	if len(keep) != 1 || keep[0].Conf != 0.9 {
		t.Fatalf("muốn giữ 1 box conf 0.9, nhận %+v", keep)
	}
}

func TestNMSKeepsDifferentClass(t *testing.T) {
	dets := []api.Detection{
		{BBox: [4]float64{0, 0, 100, 100}, Class: "cat", Conf: 0.9},
		{BBox: [4]float64{0, 0, 100, 100}, Class: "dog", Conf: 0.8}, // khác class -> giữ
	}
	if keep := NMS(dets, 0.5); len(keep) != 2 {
		t.Fatalf("muốn giữ 2 box khác class, nhận %d", len(keep))
	}
}

func TestIoUNoOverlap(t *testing.T) {
	if v := iou([4]float64{0, 0, 10, 10}, [4]float64{20, 20, 10, 10}); v != 0 {
		t.Fatalf("IoU rời nhau muốn 0, nhận %v", v)
	}
}
