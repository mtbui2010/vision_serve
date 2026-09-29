package rfdetr

import (
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// Tests DETR decoding + mapping boxes back to ORIGINAL image coords (the trickiest part).
func TestPostprocessDecodeAndMapToOriginal(t *testing.T) {
	m := &rfDETR{cfg: models.Config{
		Name:       "rf-detr",
		Width:      100,
		Height:     100,
		BoxFormat:  "cxcywh",
		ConfThresh: 0.5,
		MaxDet:     300,
		Labels:     []string{"a", "b"},
	}}

	// 1 query, 2 classes. class 1 (raw 5 -> sigmoid≈0.993) beats class 0 (raw -5).
	logits := engine.Tensor{Data: []float32{-5, 5}, Shape: []int64{1, 1, 2}}
	// box cxcywh normalized (0.5,0.5,0.5,0.5) -> input px x=25,y=25,w=50,h=50
	boxes := engine.Tensor{Data: []float32{0.5, 0.5, 0.5, 0.5}, Shape: []int64{1, 1, 4}}

	// meta: letterbox scale 0.5, pad (0,25), original image 200x100
	meta := models.PreprocessMeta{OrigWidth: 200, OrigHeight: 100, ScaleX: 0.5, ScaleY: 0.5, PadX: 0, PadY: 25}

	res, err := m.postprocess([]engine.Tensor{logits, boxes}, meta)
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("want 1 detection, got %d", len(res.Detections))
	}
	d := res.Detections[0]
	if d.Class != "b" {
		t.Fatalf("class = %q, want \"b\"", d.Class)
	}
	if math.Abs(d.Conf-0.9933) > 1e-2 {
		t.Fatalf("conf = %v, want ~0.993", d.Conf)
	}
	// input box x=25,y=25,w=50,h=50 -> orig: ((25-0)/0.5, (25-25)/0.5, 50/0.5, 50/0.5)
	want := [4]float64{50, 0, 100, 100}
	for i := range want {
		if math.Abs(d.BBox[i]-want[i]) > 1e-6 {
			t.Fatalf("bbox = %v, want %v", d.BBox, want)
		}
	}
}

// Squash path end to end (letterbox: false, which is how every RF-DETR checkpoint here was
// trained — BUGS_TO_FIX.md #1). On a strongly non-square image ScaleX != ScaleY, which the
// letterbox path never exercised: the box must come back with each axis divided by ITS OWN
// scale, and a box covering the whole input must cover the whole original image.
func TestPostprocessSquashNonSquareRoundTrip(t *testing.T) {
	m := &rfDETR{cfg: models.Config{
		Width: 64, Height: 64, Letterbox: false,
		BoxFormat: "cxcywh", ConfThresh: 0.5, Labels: []string{"a"},
	}}
	// 400x100 original: ScaleX = 64/400 = 0.16, ScaleY = 64/100 = 0.64.
	_, meta, err := m.preprocess(fillRGBA(400, 100, 1, 2, 3))
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}
	if meta.ScaleX == meta.ScaleY {
		t.Fatalf("test needs ScaleX != ScaleY, got %v", meta.ScaleX)
	}

	logits := engine.Tensor{Data: []float32{5, 5}, Shape: []int64{1, 2, 1}}
	boxes := engine.Tensor{Data: []float32{
		0.5, 0.5, 1, 1, // the whole input
		0.25, 0.75, 0.5, 0.5, // bottom-left quadrant of the input
	}, Shape: []int64{1, 2, 4}}
	res, err := m.postprocess([]engine.Tensor{logits, boxes}, meta)
	if err != nil {
		t.Fatalf("postprocess: %v", err)
	}
	if len(res.Detections) != 2 {
		t.Fatalf("want 2 detections, got %d", len(res.Detections))
	}
	want := map[[4]float64]bool{
		{0, 0, 400, 100}: false,
		{0, 50, 200, 50}: false,
	}
	for _, d := range res.Detections {
		found := false
		for w := range want {
			ok := true
			for i := range w {
				if math.Abs(d.BBox[i]-w[i]) > 1e-6 {
					ok = false
				}
			}
			if ok {
				want[w], found = true, true
			}
		}
		if !found {
			t.Fatalf("unexpected bbox %v (original coords), want one of %v", d.BBox, want)
		}
	}
}

// No output tensor has a last dim == 4 -> must return an error, NOT guess.
func TestPostprocessRejectsUnknownShape(t *testing.T) {
	m := &rfDETR{cfg: models.Config{Width: 100, Height: 100}}
	a := engine.Tensor{Data: make([]float32, 6), Shape: []int64{1, 2, 3}}
	b := engine.Tensor{Data: make([]float32, 6), Shape: []int64{1, 2, 3}}
	if _, err := m.postprocess([]engine.Tensor{a, b}, models.PreprocessMeta{}); err == nil {
		t.Fatal("want an error when the boxes tensor cannot be identified")
	}
}
