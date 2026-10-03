package scrfd

import (
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// realOutputs builds the 9 det_10g.onnx outputs with their REAL shapes for a WxH input:
// 2-D [N,1] scores (already sigmoided), [N,4] boxes, [N,10] kps, in the graph's output
// order (scores, boxes, kps). Verified with onnxruntime: at 640x640 the outputs are
// [12800,1] [3200,1] [800,1] [12800,4] [3200,4] [800,4] [12800,10] [3200,10] [800,10].
func realOutputs(w, h int) (outs []engine.Tensor, scores, boxes [3][]float32) {
	for i, s := range []int{8, 16, 32} {
		n := (h / s) * (w / s) * 2
		scores[i] = make([]float32, n)
		boxes[i] = make([]float32, n*4)
	}
	for i := 0; i < 3; i++ {
		outs = append(outs, engine.F32(scores[i], int64(len(scores[i])), 1))
	}
	for i := 0; i < 3; i++ {
		outs = append(outs, engine.F32(boxes[i], int64(len(scores[i])), 4))
	}
	for i := 0; i < 3; i++ {
		n := len(scores[i])
		outs = append(outs, engine.F32(make([]float32, n*10), int64(n), 10))
	}
	return outs, scores, boxes
}

// TestDist2BBox_StrideDecoding verifies the anchor-centre + distance-to-bbox formula on
// the REAL 2-D [N,C] tensor shapes. InsightFace anchor centres are (col*stride,
// row*stride) — the cell's top-left corner, no +0.5.
//
// stride 8, row=1, col=2, anchor=0 → k = 1*(80*2) + 2*2 + 0 = 164; cx=16, cy=8.
// l=1,t=2,r=3,b=4 → x1=8, y1=-8, x2=40, y2=40 → [8,-8,32,48] (no clamping in input space).
func TestDist2BBox_StrideDecoding(t *testing.T) {
	const n = 12800 // 80*80*2
	scoreData := make([]float32, n)
	bboxData := make([]float32, n*4)
	const targetK = 164
	scoreData[targetK] = 0.9 // a probability: the graph already applied Sigmoid
	copy(bboxData[targetK*4:], []float32{1, 2, 3, 4})

	scoreT := engine.F32(scoreData, n, 1)
	bboxT := engine.F32(bboxData, n, 4)
	sd := scrfdStride{stride: 8, numAnchors: 2, numProposals: n}

	dets, err := decodeStride(&scoreT, &bboxT, sd, 640, 640, 0.5)
	if err != nil {
		t.Fatalf("decodeStride error: %v", err)
	}
	if len(dets) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(dets))
	}
	want := [4]float64{8, -8, 32, 48}
	for i := 0; i < 4; i++ {
		if math.Abs(dets[0].BBox[i]-want[i]) > 1e-4 {
			t.Errorf("BBox[%d]: want %v, got %v", i, want[i], dets[0].BBox[i])
		}
	}
	if dets[0].Class != "face" {
		t.Errorf("Class: want \"face\", got %q", dets[0].Class)
	}
	// The score must be passed through unchanged (no second sigmoid: sigmoid(0.9)=0.71).
	if math.Abs(dets[0].Conf-0.9) > 1e-6 {
		t.Errorf("Conf: want 0.9 (score used as-is), got %v", dets[0].Conf)
	}
}

// TestDecodeStride_NoDoubleSigmoid: a probability just below the threshold must be
// rejected. With the old double sigmoid every score >= 0 became >= 0.5 and passed.
func TestDecodeStride_NoDoubleSigmoid(t *testing.T) {
	const n = 800 // stride 32 at 640x640
	scoreData := make([]float32, n)
	bboxData := make([]float32, n*4)
	scoreData[0] = 0.3
	copy(bboxData[0:], []float32{1, 1, 1, 1})
	scoreT := engine.F32(scoreData, n, 1)
	bboxT := engine.F32(bboxData, n, 4)
	sd := scrfdStride{stride: 32, numAnchors: 2, numProposals: n}
	dets, err := decodeStride(&scoreT, &bboxT, sd, 640, 640, 0.5)
	if err != nil {
		t.Fatalf("decodeStride error: %v", err)
	}
	if len(dets) != 0 {
		t.Fatalf("score 0.3 < 0.5 must be dropped, got %+v", dets)
	}
}

// TestDecodeStride_RejectsLogits: a raw-logit export (score outside [0,1]) is reported
// as an error rather than silently mis-thresholded.
func TestDecodeStride_RejectsLogits(t *testing.T) {
	const n = 800
	scoreData := make([]float32, n)
	scoreData[5] = 4.2
	scoreT := engine.F32(scoreData, n, 1)
	bboxT := engine.F32(make([]float32, n*4), n, 4)
	sd := scrfdStride{stride: 32, numAnchors: 2, numProposals: n}
	if _, err := decodeStride(&scoreT, &bboxT, sd, 640, 640, 0.5); err == nil {
		t.Fatal("expected an error for a non-probability score")
	}
}

// TestDist2BBox_NoClamp checks a box well inside the input (stride 16, anchor 1).
func TestDist2BBox_NoClamp(t *testing.T) {
	const n = 3200
	scoreData := make([]float32, n)
	bboxData := make([]float32, n*4)

	// row=5, col=10, anchor=1 → gridW=40 → k = 5*(40*2) + 10*2 + 1 = 421
	const targetK = 421
	scoreData[targetK] = 0.95
	copy(bboxData[targetK*4:], []float32{2, 2, 2, 2})

	scoreT := engine.F32(scoreData, n, 1)
	bboxT := engine.F32(bboxData, n, 4)
	sd := scrfdStride{stride: 16, numAnchors: 2, numProposals: n}
	dets, err := decodeStride(&scoreT, &bboxT, sd, 640, 640, 0.5)
	if err != nil {
		t.Fatalf("decodeStride error: %v", err)
	}
	if len(dets) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(dets))
	}
	// cx = 10*16 = 160, cy = 5*16 = 80 → x1=128, y1=48, x2=192, y2=112
	want := [4]float64{128, 48, 64, 64}
	for i := 0; i < 4; i++ {
		if math.Abs(dets[0].BBox[i]-want[i]) > 1e-4 {
			t.Errorf("BBox[%d]: want %v, got %v", i, want[i], dets[0].BBox[i])
		}
	}
}

// TestPostprocess_RealShapes runs the full postprocess on the REAL 2-D output layout
// (the old decoder read Dim(1) as N, saw C, and returned 0 faces for every image).
// One face at stride 8, row 10, col 20 (k = 10*160 + 20*2 = 1640): cx=160, cy=80,
// l=t=r=b=2 → input box [144,64,32,32]. Letterbox meta scale 0.5, no pad (1280x1280
// image) → original = [288,128,64,64].
func TestPostprocess_RealShapes(t *testing.T) {
	cfg := models.Config{Width: 640, Height: 640, ConfThresh: 0.5, MaxDet: 100}
	meta := models.PreprocessMeta{OrigWidth: 1280, OrigHeight: 1280, ScaleX: 0.5, ScaleY: 0.5}
	outs, scores, boxes := realOutputs(640, 640)
	if outs[0].Shape[0] != 12800 || outs[3].Shape[1] != 4 || len(outs) != 9 {
		t.Fatalf("test fixture does not match det_10g.onnx shapes: %v", outs[0].Shape)
	}
	const k = 1640
	scores[0][k] = 0.88
	copy(boxes[0][k*4:], []float32{2, 2, 2, 2})
	// A weaker duplicate on the other anchor of the same cell must be removed by NMS.
	scores[0][k+1] = 0.70
	copy(boxes[0][(k+1)*4:], []float32{2, 2, 2.1, 2})

	res, err := postprocess(outs, meta, cfg)
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("want 1 face after NMS, got %d: %+v", len(res.Detections), res.Detections)
	}
	want := [4]float64{288, 128, 64, 64}
	for i := 0; i < 4; i++ {
		if math.Abs(res.Detections[0].BBox[i]-want[i]) > 1e-4 {
			t.Errorf("BBox[%d]: want %v, got %v", i, want[i], res.Detections[0].BBox[i])
		}
	}
	if math.Abs(res.Detections[0].Conf-0.88) > 1e-6 {
		t.Errorf("Conf: want 0.88, got %v", res.Detections[0].Conf)
	}
}

// TestPostprocess_NonSquareInput: proposal counts must be derived from the configured
// input size, not hard-coded for 640x640. At 320x256: stride 8 → 32*40*2 = 2560,
// stride 16 → 16*20*2 = 640, stride 32 → 8*10*2 = 160.
func TestPostprocess_NonSquareInput(t *testing.T) {
	cfg := models.Config{Width: 320, Height: 256, ConfThresh: 0.5}
	meta := models.PreprocessMeta{OrigWidth: 320, OrigHeight: 256, ScaleX: 1, ScaleY: 1}
	outs, scores, boxes := realOutputs(320, 256)
	if len(scores[0]) != 2560 || len(scores[1]) != 640 || len(scores[2]) != 160 {
		t.Fatalf("unexpected proposal counts %d %d %d", len(scores[0]), len(scores[1]), len(scores[2]))
	}
	// stride 32, row 3, col 7, anchor 0 → k = 3*(10*2) + 7*2 = 74; cx=224, cy=96.
	const k = 74
	scores[2][k] = 0.9
	copy(boxes[2][k*4:], []float32{1, 1, 1, 1})
	res, err := postprocess(outs, meta, cfg)
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(res.Detections) != 1 {
		t.Fatalf("want 1 face, got %d", len(res.Detections))
	}
	want := [4]float64{192, 64, 64, 64}
	for i := 0; i < 4; i++ {
		if math.Abs(res.Detections[0].BBox[i]-want[i]) > 1e-4 {
			t.Errorf("BBox[%d]: want %v, got %v", i, want[i], res.Detections[0].BBox[i])
		}
	}
}

// TestPostprocess_BatchedShapesAccepted: an export that keeps the batch dim ([1,N,C])
// decodes the same as the 2-D layout.
func TestPostprocess_BatchedShapesAccepted(t *testing.T) {
	cfg := models.Config{Width: 640, Height: 640, ConfThresh: 0.5}
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 640, ScaleX: 1, ScaleY: 1}
	outs, scores, boxes := realOutputs(640, 640)
	scores[1][421] = 0.95
	copy(boxes[1][421*4:], []float32{2, 2, 2, 2})
	for i := range outs {
		outs[i].Shape = append([]int64{1}, outs[i].Shape...)
	}
	res, err := postprocess(outs, meta, cfg)
	if err != nil {
		t.Fatalf("postprocess error: %v", err)
	}
	if len(res.Detections) != 1 || math.Abs(res.Detections[0].BBox[0]-128) > 1e-4 {
		t.Fatalf("want 1 face at x=128, got %+v", res.Detections)
	}
}

// TestPostprocess_NoMatchingOutputs: outputs that match no stride level are an error,
// not a silent "0 faces".
func TestPostprocess_NoMatchingOutputs(t *testing.T) {
	cfg := models.Config{Width: 640, Height: 640}
	outs := []engine.Tensor{engine.F32(make([]float32, 7), 7, 1)}
	if _, err := postprocess(outs, models.PreprocessMeta{OrigWidth: 1, OrigHeight: 1, ScaleX: 1, ScaleY: 1}, cfg); err == nil {
		t.Fatal("expected an error when no output matches a stride level")
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
