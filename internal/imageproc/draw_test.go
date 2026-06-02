package imageproc

import (
	"image"
	"image/color"
	"testing"

	"visionserve/pkg/api"
)

func TestDrawDetectionsKeepsBoundsAndDrawsBox(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 200, 150))
	// nền trắng để dễ phát hiện pixel bị vẽ đè.
	for y := 0; y < 150; y++ {
		for x := 0; x < 200; x++ {
			src.Set(x, y, color.RGBA{0xFF, 0xFF, 0xFF, 0xFF})
		}
	}
	dets := []api.Detection{{BBox: [4]float64{20, 30, 60, 40}, Class: "cat", Conf: 0.9}}

	out := DrawDetections(src, dets)
	if out.Bounds() != src.Bounds() {
		t.Fatalf("bounds đổi: %v != %v", out.Bounds(), src.Bounds())
	}
	// nguồn KHÔNG bị sửa (DrawDetections vẽ trên bản sao).
	if src.RGBAAt(20, 30) != (color.RGBA{0xFF, 0xFF, 0xFF, 0xFF}) {
		t.Fatalf("ảnh nguồn bị sửa tại mép box")
	}
	// có ít nhất một pixel trên đường viền box khác trắng.
	changed := false
	for x := 20; x < 80; x++ {
		if out.RGBAAt(x, 30) != (color.RGBA{0xFF, 0xFF, 0xFF, 0xFF}) {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("không thấy viền box được vẽ ở mép trên")
	}
}

func TestDrawDetectionsClampsOutOfBounds(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 64, 64))
	// box tràn ra ngoài biên + box rỗng — không được panic, không vẽ ngoài ảnh.
	dets := []api.Detection{
		{BBox: [4]float64{50, 50, 100, 100}, Class: "x", Conf: 0.5}, // tràn phải/dưới
		{BBox: [4]float64{-10, -10, 5, 5}, Class: "y", Conf: 0.5},   // tràn trái/trên
		{BBox: [4]float64{10, 10, 0, 0}, Class: "z", Conf: 0.5},     // rỗng
	}
	out := DrawDetections(src, dets) // chỉ cần không panic
	if out.Bounds() != src.Bounds() {
		t.Fatalf("bounds đổi")
	}
}

func TestDrawDetectionsEmpty(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 32, 32))
	out := DrawDetections(src, nil)
	if out.Bounds() != src.Bounds() {
		t.Fatalf("bounds đổi với 0 detection")
	}
}
