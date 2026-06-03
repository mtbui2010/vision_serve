package rtdetr

import (
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// Tests DETR decoding + mapping boxes back to ORIGINAL image coords (the trickiest part).
// RT-DETR uses COCO-80 (indices 0-79, no N/A gap) — label index maps directly to cfg.Labels.
func TestPostprocessDecodeAndMapToOriginal(t *testing.T) {
	m := &rtDETR{cfg: models.Config{
		Name:       "rt-detr",
		Width:      640,
		Height:     640,
		BoxFormat:  "cxcywh",
		ConfThresh: 0.5,
		MaxDet:     300,
		Labels:     []string{"person", "bicycle"},
	}}

	// 1 query, 2 classes. class 1 (raw 5 -> sigmoid≈0.993) beats class 0 (raw -5).
	logits := engine.Tensor{Data: []float32{-5, 5}, Shape: []int64{1, 1, 2}}
	// box cxcywh normalized (0.5,0.5,0.5,0.5) -> input px cx=320,cy=320,w=320,h=320
	// -> x=160,y=160,w=320,h=320
	boxes := engine.Tensor{Data: []float32{0.5, 0.5, 0.5, 0.5}, Shape: []int64{1, 1, 4}}

	// meta: letterbox scale 0.5, pad (0,80), original image 1280x960
	meta := models.PreprocessMeta{
		OrigWidth: 1280, OrigHeight: 960,
		ScaleX: 0.5, ScaleY: 0.5,
		PadX: 0, PadY: 80,
	}

	res, err := m.postprocess([]engine.Tensor{logits, boxes}, meta)
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("want 1 detection, got %d", len(res.Detections))
	}
	d := res.Detections[0]
	if d.Class != "bicycle" {
		t.Fatalf("class = %q, want \"bicycle\"", d.Class)
	}
	if math.Abs(d.Conf-0.9933) > 1e-2 {
		t.Fatalf("conf = %v, want ~0.993", d.Conf)
	}
	// input box x=160,y=160,w=320,h=320 -> orig:
	//   ox = (160-0)/0.5 = 320
	//   oy = (160-80)/0.5 = 160
	//   ow = 320/0.5 = 640
	//   oh = 320/0.5 = 640
	// clamped to 1280x960: ox=320, oy=160, ow=640, oh=640 (all within bounds)
	want := [4]float64{320, 160, 640, 640}
	for i := range want {
		if math.Abs(d.BBox[i]-want[i]) > 1e-6 {
			t.Fatalf("bbox[%d] = %v, want %v (full bbox: %v)", i, d.BBox[i], want[i], d.BBox)
		}
	}
}

// Queries below the confidence threshold must be filtered out.
func TestPostprocessConfidenceFiltering(t *testing.T) {
	m := &rtDETR{cfg: models.Config{
		Width: 640, Height: 640,
		BoxFormat:  "cxcywh",
		ConfThresh: 0.9,
		Labels:     []string{"person"},
	}}

	// 2 queries: query 0 raw=5 -> sigmoid≈0.993 (above 0.9), query 1 raw=1 -> sigmoid≈0.731 (below 0.9)
	logits := engine.Tensor{Data: []float32{5, 1}, Shape: []int64{1, 2, 1}}
	boxes := engine.Tensor{Data: []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, Shape: []int64{1, 2, 4}}

	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 640, ScaleX: 1.0, ScaleY: 1.0}

	res, err := m.postprocess([]engine.Tensor{logits, boxes}, meta)
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("want 1 detection (above threshold), got %d", len(res.Detections))
	}
	if res.Detections[0].Conf < 0.9 {
		t.Fatalf("surviving detection conf %v is below threshold 0.9", res.Detections[0].Conf)
	}
}

// No output tensor has last dim == 4 -> must return an error, NOT guess.
func TestPostprocessRejectsUnknownShape(t *testing.T) {
	m := &rtDETR{cfg: models.Config{Width: 640, Height: 640}}
	a := engine.Tensor{Data: make([]float32, 6), Shape: []int64{1, 2, 3}}
	b := engine.Tensor{Data: make([]float32, 6), Shape: []int64{1, 2, 3}}
	if _, err := m.postprocess([]engine.Tensor{a, b}, models.PreprocessMeta{}); err == nil {
		t.Fatal("want an error when the boxes tensor cannot be identified")
	}
}

// Wrong number of output tensors -> must return an error.
func TestPostprocessRejectsWrongOutputCount(t *testing.T) {
	m := &rtDETR{cfg: models.Config{Width: 640, Height: 640}}
	single := engine.Tensor{Data: make([]float32, 4), Shape: []int64{1, 1, 4}}
	if _, err := m.postprocess([]engine.Tensor{single}, models.PreprocessMeta{}); err == nil {
		t.Fatal("want an error when only 1 output tensor is provided")
	}
}

// MaxDet cap: only the top-N detections by confidence should be returned.
func TestPostprocessMaxDetCap(t *testing.T) {
	m := &rtDETR{cfg: models.Config{
		Width: 100, Height: 100,
		BoxFormat:  "cxcywh",
		ConfThresh: 0.0,
		MaxDet:     2,
		Labels:     []string{"x"},
	}}

	// 4 queries, 1 class, raw scores 1,2,3,4 (all pass threshold 0)
	logits := engine.Tensor{Data: []float32{1, 2, 3, 4}, Shape: []int64{1, 4, 1}}
	boxes := engine.Tensor{
		Data:  []float32{0.5, 0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2},
		Shape: []int64{1, 4, 4},
	}
	meta := models.PreprocessMeta{OrigWidth: 100, OrigHeight: 100, ScaleX: 1.0, ScaleY: 1.0}

	res, err := m.postprocess([]engine.Tensor{logits, boxes}, meta)
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(res.Detections) != 2 {
		t.Fatalf("want 2 detections (MaxDet=2), got %d", len(res.Detections))
	}
	// Sorted by confidence descending: highest conf first
	if res.Detections[0].Conf < res.Detections[1].Conf {
		t.Fatal("detections not sorted by confidence descending")
	}
}
