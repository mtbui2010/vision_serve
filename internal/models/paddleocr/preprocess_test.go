package paddleocr

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func solid(w, h int) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 255, 255, 255, 255
	}
	img.SetNRGBA(0, 0, color.NRGBA{0, 0, 0, 255})
	return img
}

// detPreprocess produces the det model's real input shape [1,3,H,W] with H,W multiples of
// 32 (the ONNX accepts dynamic H/W; output prob map is [1,1,H,W]).
func TestDetPreprocess_ShapeAndMeta(t *testing.T) {
	cases := []struct {
		w, h         int
		wantH, wantW int64
		wantScale    float64
	}{
		{800, 360, 384, 800, 1},      // never upscale; pad 360 -> 384
		{2000, 1000, 480, 960, 0.48}, // long side -> 960
	}
	for _, c := range cases {
		tns, meta := detPreprocess(solid(c.w, c.h), detMaxSide)
		want := []int64{1, 3, c.wantH, c.wantW}
		for i := range want {
			if tns.Shape[i] != want[i] {
				t.Fatalf("%dx%d: shape %v, want %v", c.w, c.h, tns.Shape, want)
			}
		}
		if math.Abs(meta.ScaleX-c.wantScale) > 1e-9 || meta.PadX != 0 || meta.PadY != 0 {
			t.Fatalf("%dx%d: meta %+v", c.w, c.h, meta)
		}
		// ImageNet normalisation of a white pixel in the R plane, black at (0,0).
		if v := tns.Data[1]; math.Abs(float64(v)-(1-0.485)/0.229) > 1e-3 && c.wantScale == 1 {
			t.Errorf("white R = %v", v)
		}
		if v := tns.Data[0]; math.Abs(float64(v)-(0-0.485)/0.229) > 1e-3 && c.wantScale == 1 {
			t.Errorf("black R = %v", v)
		}
	}
}

// Boxes from the det map map back to ORIGINAL coordinates (BBox rule) and are clamped.
func TestMapBoxToOriginal(t *testing.T) {
	_, meta := detPreprocess(solid(2000, 1000), detMaxSide) // scale 0.48
	got := mapBoxToOriginal([4]float64{48, 96, 480, 48}, meta)
	want := [4]float64{100, 200, 1000, 100}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	got = mapBoxToOriginal([4]float64{900, 470, 100, 40}, meta) // overruns the padded edge
	if got[0]+got[2] > 2000+1e-9 || got[1]+got[3] > 1000+1e-9 {
		t.Fatalf("not clamped to original bounds: %v", got)
	}
}

// recPreprocess yields the rec model's real input [1,3,48,W], normalised to [-1,1].
func TestRecPreprocess_Shape(t *testing.T) {
	tns, w := recPreprocess(solid(400, 200), [4]float64{10, 10, 200, 40})
	if w != 240 || tns.Shape[0] != 1 || tns.Shape[1] != 3 || tns.Shape[2] != 48 || tns.Shape[3] != 240 {
		t.Fatalf("shape %v w=%d, want [1 3 48 240]", tns.Shape, w)
	}
	if v := tns.Data[0]; math.Abs(float64(v)-1) > 1e-6 {
		t.Fatalf("white pixel normalised to %v, want 1", v)
	}
}
