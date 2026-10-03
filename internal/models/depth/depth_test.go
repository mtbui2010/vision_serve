package depth

import (
	"image"
	"image/color"
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func gradient(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x * 255 / w), G: uint8(y * 255 / h), B: 128, A: 255})
		}
	}
	return img
}

var imagenet = struct{ mean, std []float32 }{
	[]float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225},
}

func TestPreprocessSquashUnchanged(t *testing.T) {
	// MiDaS-style manifest: no keep_aspect -> exactly width x height whatever the input aspect.
	cfg := models.Config{Width: 256, Height: 256, Mean: imagenet.mean, Std: imagenet.std}
	in, meta, err := preprocess(gradient(848, 480), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{1, 3, 256, 256}; !equalShape(in.Shape, want) {
		t.Fatalf("shape %v, want %v", in.Shape, want)
	}
	if meta.OrigWidth != 848 || meta.OrigHeight != 480 || meta.PadX != 0 || meta.PadY != 0 {
		t.Fatalf("meta %+v", meta)
	}
	if !near(meta.ScaleX, 256.0/848) || !near(meta.ScaleY, 256.0/480) {
		t.Fatalf("scale %v,%v", meta.ScaleX, meta.ScaleY)
	}
}

func TestPreprocessKeepAspect(t *testing.T) {
	cfg := models.Config{Width: 518, Height: 518, KeepAspect: true, MultipleOf: 14,
		Mean: imagenet.mean, Std: imagenet.std}
	cases := []struct{ w, h, wantW, wantH int }{
		{848, 480, 910, 518}, // the HF example
		{480, 848, 518, 910}, // portrait
		{640, 427, 518, 350}, // short side BELOW 518 (width scale is closer to 1)
		{518, 518, 518, 518},
	}
	for _, c := range cases {
		in, meta, err := preprocess(gradient(c.w, c.h), cfg)
		if err != nil {
			t.Fatal(err)
		}
		if want := []int64{1, 3, int64(c.wantH), int64(c.wantW)}; !equalShape(in.Shape, want) {
			t.Errorf("%dx%d: shape %v, want %v", c.w, c.h, in.Shape, want)
			continue
		}
		if len(in.Data) != 3*c.wantW*c.wantH {
			t.Errorf("%dx%d: %d values for shape %v", c.w, c.h, len(in.Data), in.Shape)
		}
		if meta.OrigWidth != c.w || meta.OrigHeight != c.h || meta.PadX != 0 || meta.PadY != 0 {
			t.Errorf("%dx%d: meta %+v", c.w, c.h, meta)
		}
		// Per-axis scale (the multiple-of rounding makes them differ slightly).
		if !near(meta.ScaleX, float64(c.wantW)/float64(c.w)) || !near(meta.ScaleY, float64(c.wantH)/float64(c.h)) {
			t.Errorf("%dx%d: scale %v,%v", c.w, c.h, meta.ScaleX, meta.ScaleY)
		}
	}
}

// The normalisation is the same on both paths: a flat grey image gives (v/255-mean)/std.
func TestPreprocessKeepAspectNormalizes(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 100, 60))
	for i := range img.Pix {
		img.Pix[i] = 200
	}
	cfg := models.Config{Width: 518, Height: 518, KeepAspect: true, MultipleOf: 14,
		Mean: imagenet.mean, Std: imagenet.std}
	in, _, err := preprocess(img, cfg)
	if err != nil {
		t.Fatal(err)
	}
	hw := int(in.Shape[2] * in.Shape[3])
	for c := 0; c < 3; c++ {
		want := (200.0/255 - float64(imagenet.mean[c])) / float64(imagenet.std[c])
		if got := float64(in.Data[c*hw+hw/2]); math.Abs(got-want) > 1e-3 {
			t.Errorf("channel %d: %v, want %v", c, got, want)
		}
	}
}

// Keep-aspect inputs make NON-square outputs ([1, 518, 910] for an 848x480 photo): the decoder
// must take H and W from the tensor, row-major.
func TestPostprocessNonSquare(t *testing.T) {
	h, w := 3, 5
	data := make([]float32, h*w)
	for i := range data {
		data[i] = float32(i)
	}
	cfg := models.Config{Width: 518, Height: 518}
	res, err := postprocess([]engine.Tensor{{Data: data, Shape: []int64{1, int64(h), int64(w)}}}, models.PreprocessMeta{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if res.DepthWidth != w || res.DepthHeight != h || len(res.DepthMap) != h*w {
		t.Fatalf("got %dx%d (%d values), want %dx%d", res.DepthWidth, res.DepthHeight, len(res.DepthMap), w, h)
	}
	// Row-major: element (y=1, x=0) is index 5 -> 5/14.
	if math.Abs(float64(res.DepthMap[w])-5.0/14) > 1e-6 || res.DepthMap[0] != 0 || res.DepthMap[h*w-1] != 1 {
		t.Fatalf("normalisation/order wrong: %v", res.DepthMap)
	}
	if _, err := postprocess([]engine.Tensor{{Data: data[:14], Shape: []int64{1, 3, 5}}}, models.PreprocessMeta{}, cfg); err == nil {
		t.Fatal("expected an error for a data/shape mismatch")
	}
}

func equalShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
