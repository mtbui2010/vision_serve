package rfdetr

import (
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// Kiểm tra decode DETR + map box về toạ độ ảnh GỐC (phần dễ sai nhất).
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

	// 1 query, 2 class. class 1 (raw 5 -> sigmoid≈0.993) thắng class 0 (raw -5).
	logits := engine.Tensor{Data: []float32{-5, 5}, Shape: []int64{1, 1, 2}}
	// box cxcywh normalized (0.5,0.5,0.5,0.5) -> input px x=25,y=25,w=50,h=50
	boxes := engine.Tensor{Data: []float32{0.5, 0.5, 0.5, 0.5}, Shape: []int64{1, 1, 4}}

	// meta: letterbox scale 0.5, pad (0,25), ảnh gốc 200x100
	meta := models.PreprocessMeta{OrigWidth: 200, OrigHeight: 100, ScaleX: 0.5, ScaleY: 0.5, PadX: 0, PadY: 25}

	res, err := m.postprocess([]engine.Tensor{logits, boxes}, meta)
	if err != nil {
		t.Fatalf("postprocess lỗi: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("muốn 1 detection, nhận %d", len(res.Detections))
	}
	d := res.Detections[0]
	if d.Class != "b" {
		t.Fatalf("class = %q, muốn \"b\"", d.Class)
	}
	if math.Abs(d.Conf-0.9933) > 1e-2 {
		t.Fatalf("conf = %v, muốn ~0.993", d.Conf)
	}
	// input box x=25,y=25,w=50,h=50 -> orig: ((25-0)/0.5, (25-25)/0.5, 50/0.5, 50/0.5)
	want := [4]float64{50, 0, 100, 100}
	for i := range want {
		if math.Abs(d.BBox[i]-want[i]) > 1e-6 {
			t.Fatalf("bbox = %v, muốn %v", d.BBox, want)
		}
	}
}

// Output không có tensor nào chiều cuối == 4 -> phải báo lỗi, KHÔNG đoán bừa.
func TestPostprocessRejectsUnknownShape(t *testing.T) {
	m := &rfDETR{cfg: models.Config{Width: 100, Height: 100}}
	a := engine.Tensor{Data: make([]float32, 6), Shape: []int64{1, 2, 3}}
	b := engine.Tensor{Data: make([]float32, 6), Shape: []int64{1, 2, 3}}
	if _, err := m.postprocess([]engine.Tensor{a, b}, models.PreprocessMeta{}); err == nil {
		t.Fatal("muốn lỗi khi không xác định được tensor boxes")
	}
}
