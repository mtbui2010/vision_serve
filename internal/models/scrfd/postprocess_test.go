package scrfd

import (
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// TestDist2BBox verifies the core anchor-center + distance-to-bbox decoding formula.
// We construct a synthetic score+bbox tensor for stride 8 with a single high-confidence
// proposal at grid position (row=1, col=2, anchor=0) and assert the decoded BBox
// (in letterboxed 640-space, before original-coord remapping) is correct.
//
// Anchor center: cx = (2+0.5)*8 = 20, cy = (1+0.5)*8 = 12
// Distances: l=1, t=2, r=3, b=4  (in stride units)
// Expected:
//
//	x1 = 20 - 1*8 = 12
//	y1 = 12 - 2*8 = -4  → clamped to 0 (since inputH=640 and inputW=640; y1<0 so y1=0)
//	x2 = 20 + 3*8 = 44
//	y2 = 12 + 4*8 = 44
//	w  = 44 - 12 = 32
//	h  = 44 - 0  = 44
func TestDist2BBox_StrideDecoding(t *testing.T) {
	const (
		inputW = 640
		inputH = 640
		stride = 8
		n      = 12800 // 80*80*2
	)

	// All scores at -100 (sigmoid ≈ 0) except one.
	scoreData := make([]float32, n)
	for i := range scoreData {
		scoreData[i] = -100 // near-zero probability
	}

	bboxData := make([]float32, n*4)

	// row=1, col=2, anchor=0 → gridW=80, numAnchors=2
	// k = row*(gridW*numAnchors) + col*numAnchors + anchor
	//   = 1*(80*2) + 2*2 + 0 = 160 + 4 = 164
	const targetK = 164
	scoreData[targetK] = 10.0 // sigmoid(10) ≈ 0.9999, well above 0.5 threshold

	// distances: l=1, t=2, r=3, b=4
	bboxData[targetK*4+0] = 1.0 // l
	bboxData[targetK*4+1] = 2.0 // t
	bboxData[targetK*4+2] = 3.0 // r
	bboxData[targetK*4+3] = 4.0 // b

	scoreT := engine.F32(scoreData, 1, int64(n), 1)
	bboxT := engine.F32(bboxData, 1, int64(n), 4)

	sd := scrfdStride{stride: stride, numAnchors: 2, numProposals: n}

	dets, err := decodeStride(&scoreT, &bboxT, sd, inputW, inputH, 0.5)
	if err != nil {
		t.Fatalf("decodeStride error: %v", err)
	}
	if len(dets) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(dets))
	}

	// cx = (2 + 0.5) * 8 = 20
	// cy = (1 + 0.5) * 8 = 12
	// x1 = 20 - 1*8 = 12,  y1 = 12 - 2*8 = -4 → clamped → 0
	// x2 = 20 + 3*8 = 44,  y2 = 12 + 4*8 = 44
	// w = 44-12 = 32,  h = 44-0 = 44
	want := [4]float64{12, 0, 32, 44}
	got := dets[0].BBox
	const eps = 1e-4
	for i := 0; i < 4; i++ {
		if math.Abs(got[i]-want[i]) > eps {
			t.Errorf("BBox[%d]: want %v, got %v", i, want[i], got[i])
		}
	}
	if dets[0].Class != "face" {
		t.Errorf("Class: want \"face\", got %q", dets[0].Class)
	}
	if dets[0].Conf < 0.99 {
		t.Errorf("Conf: expected ~1.0, got %v", dets[0].Conf)
	}
}

// TestDist2BBox_NoClamp checks that a box entirely within bounds is decoded without clamping.
func TestDist2BBox_NoClamp(t *testing.T) {
	const (
		inputW = 640
		inputH = 640
		stride = 16
		n      = 3200
	)
	scoreData := make([]float32, n)
	bboxData := make([]float32, n*4)

	// row=5, col=10, anchor=1 → gridW=40, numAnchors=2
	// k = 5*(40*2) + 10*2 + 1 = 400 + 20 + 1 = 421
	const targetK = 421
	scoreData[targetK] = 8.0

	// cx = (10+0.5)*16 = 168, cy = (5+0.5)*16 = 88
	// l=2, t=2, r=2, b=2 → x1=136, y1=56, x2=200, y2=120
	bboxData[targetK*4+0] = 2.0
	bboxData[targetK*4+1] = 2.0
	bboxData[targetK*4+2] = 2.0
	bboxData[targetK*4+3] = 2.0

	scoreT := engine.F32(scoreData, 1, int64(n), 1)
	bboxT := engine.F32(bboxData, 1, int64(n), 4)

	sd := scrfdStride{stride: stride, numAnchors: 2, numProposals: n}
	dets, err := decodeStride(&scoreT, &bboxT, sd, inputW, inputH, 0.5)
	if err != nil {
		t.Fatalf("decodeStride error: %v", err)
	}
	if len(dets) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(dets))
	}

	// x1=168-2*16=136, y1=88-2*16=56, x2=168+2*16=200, y2=88+2*16=120
	// w=64, h=64
	want := [4]float64{136, 56, 64, 64}
	got := dets[0].BBox
	const eps = 1e-4
	for i := 0; i < 4; i++ {
		if math.Abs(got[i]-want[i]) > eps {
			t.Errorf("BBox[%d]: want %v, got %v", i, want[i], got[i])
		}
	}
}

// TestPostprocess_TensorIdentification verifies that postprocess correctly routes
// tensors to the right stride buckets by shape, and that NMS + original-coord
// remapping don't crash on a minimal synthetic input.
func TestPostprocess_TensorIdentification(t *testing.T) {
	cfg := models.Config{
		Width:      640,
		Height:     640,
		ConfThresh: 0.5,
		MaxDet:     100,
	}
	meta := models.PreprocessMeta{
		OrigWidth: 1280, OrigHeight: 720,
		ScaleX: 0.5, ScaleY: 0.5,
		PadX: 0, PadY: 0,
	}

	// Build minimal tensors: one proposal visible at stride-8, rest below threshold.
	const n8, n16, n32 = 12800, 3200, 800
	s8 := make([]float32, n8)
	b8 := make([]float32, n8*4)
	s16 := make([]float32, n16)
	b16 := make([]float32, n16*4)
	s32 := make([]float32, n32)
	b32 := make([]float32, n32*4)

	// Place one face at stride-8, k=0 (row=0, col=0)
	// cx=4, cy=4; l=0.5, t=0.5, r=0.5, b=0.5 → in 640-space: x1=0, y1=0, x2=8, y2=8
	s8[0] = 8.0
	b8[0] = 0.5
	b8[1] = 0.5
	b8[2] = 0.5
	b8[3] = 0.5

	outs := []engine.Tensor{
		engine.F32(s8, 1, n8, 1),
		engine.F32(b8, 1, n8, 4),
		engine.F32(s16, 1, n16, 1),
		engine.F32(b16, 1, n16, 4),
		engine.F32(s32, 1, n32, 1),
		engine.F32(b32, 1, n32, 4),
	}

	result, err := postprocess(outs, meta, cfg)
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(result.Detections) == 0 {
		t.Fatal("expected at least 1 detection")
	}
	d := result.Detections[0]
	if d.Class != "face" {
		t.Errorf("class: want face, got %q", d.Class)
	}
	// In original coords: x1=0*..., remapped from letterboxed.
	// Just check the struct is valid (non-negative w/h).
	if d.BBox[2] <= 0 || d.BBox[3] <= 0 {
		t.Errorf("invalid BBox: %v", d.BBox)
	}
}

// TestMapToOrig verifies the letterbox inverse mapping.
func TestMapToOrig(t *testing.T) {
	// scale=0.5, padX=80, padY=0 → orig = (input-pad)/scale
	meta := models.PreprocessMeta{
		OrigWidth: 800, OrigHeight: 600,
		ScaleX: 0.5, ScaleY: 0.5,
		PadX: 80, PadY: 20,
	}
	// box in 640-space: x=100, y=40, w=50, h=60
	ox, oy, ow, oh := mapToOrig([4]float64{100, 40, 50, 60}, meta)
	// ox = (100-80)/0.5 = 40, oy = (40-20)/0.5 = 40, ow = 100, oh = 120
	if math.Abs(ox-40) > 1e-4 {
		t.Errorf("ox: want 40, got %v", ox)
	}
	if math.Abs(oy-40) > 1e-4 {
		t.Errorf("oy: want 40, got %v", oy)
	}
	if math.Abs(ow-100) > 1e-4 {
		t.Errorf("ow: want 100, got %v", ow)
	}
	if math.Abs(oh-120) > 1e-4 {
		t.Errorf("oh: want 120, got %v", oh)
	}
}
