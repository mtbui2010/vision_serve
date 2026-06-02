package imageproc

import (
	"image"
	"image/color"
	"testing"

	"visionserve/pkg/api"
)

func TestDrawDetectionsKeepsBoundsAndDrawsBox(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 200, 150))
	// white background to make overdrawn pixels easy to detect.
	for y := 0; y < 150; y++ {
		for x := 0; x < 200; x++ {
			src.Set(x, y, color.RGBA{0xFF, 0xFF, 0xFF, 0xFF})
		}
	}
	dets := []api.Detection{{BBox: [4]float64{20, 30, 60, 40}, Class: "cat", Conf: 0.9}}

	out := DrawDetections(src, dets)
	if out.Bounds() != src.Bounds() {
		t.Fatalf("bounds changed: %v != %v", out.Bounds(), src.Bounds())
	}
	// the source is NOT modified (DrawDetections draws on a copy).
	if src.RGBAAt(20, 30) != (color.RGBA{0xFF, 0xFF, 0xFF, 0xFF}) {
		t.Fatalf("source image was modified at the box edge")
	}
	// at least one pixel on the box outline differs from white.
	changed := false
	for x := 20; x < 80; x++ {
		if out.RGBAAt(x, 30) != (color.RGBA{0xFF, 0xFF, 0xFF, 0xFF}) {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("no box outline drawn at the top edge")
	}
}

func TestDrawDetectionsClampsOutOfBounds(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 64, 64))
	// boxes overflowing the bounds + an empty box — must not panic, must not draw outside the image.
	dets := []api.Detection{
		{BBox: [4]float64{50, 50, 100, 100}, Class: "x", Conf: 0.5}, // overflow right/bottom
		{BBox: [4]float64{-10, -10, 5, 5}, Class: "y", Conf: 0.5},   // overflow left/top
		{BBox: [4]float64{10, 10, 0, 0}, Class: "z", Conf: 0.5},     // empty
	}
	out := DrawDetections(src, dets) // just needs to not panic
	if out.Bounds() != src.Bounds() {
		t.Fatalf("bounds changed")
	}
}

func TestDrawDetectionsEmpty(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 32, 32))
	out := DrawDetections(src, nil)
	if out.Bounds() != src.Bounds() {
		t.Fatalf("bounds changed with 0 detections")
	}
}
