package efficientsam

import (
	"image"
	"image/color"
	"math"
	"strconv"
	"strings"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// Real I/O shapes of models/efficient-sam/*.onnx (inspected with onnxruntime):
//
//	encoder  in  batched_images        [1, 3, H, W]   (graph resizes to 1024² + ImageNet-normalizes)
//	         out image_embeddings      [1, 256, 64, 64]
//	decoder  in  batched_point_coords  [1, 1, N, 2]   original-pixel coords
//	             batched_point_labels  [1, 1, N]      float32
//	             orig_im_size          [2]            int64 [H, W]
//	         out output_masks          [1, 1, 3, H, W] ORIGINAL resolution
//	             iou_predictions       [1, 1, 3]      unsorted
//	             onnx::Shape_1830      [1, 3, 256, 256] (unused)
var realDecoderOutputs = []string{"output_masks", "iou_predictions", "onnx::Shape_1830"}

// decodeRLE decodes the column-major RLE produced by encodeRLEColumnMajor into a
// row-major bool grid.
func decodeRLE(t *testing.T, rle string, h, w int) []bool {
	t.Helper()
	out := make([]bool, h*w)
	pos, v := 0, false
	for _, f := range strings.Fields(rle) {
		c, err := strconv.Atoi(f)
		if err != nil {
			t.Fatalf("bad RLE count %q", f)
		}
		for k := 0; k < c; k++ {
			if pos >= h*w {
				t.Fatalf("RLE overruns %dx%d", w, h)
			}
			x, y := pos/h, pos%h
			out[y*w+x] = v
			pos++
		}
		v = !v
	}
	if pos != h*w {
		t.Fatalf("RLE covers %d pixels, want %d", pos, h*w)
	}
	return out
}

// TestEncoderInput_OriginalSizeUnitRange: the encoder graph resizes and normalizes, so
// Go must feed the ORIGINAL-size image as pixel/255 — no resize, no ImageNet norm.
func TestEncoderInput_OriginalSizeUnitRange(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	img.Set(0, 0, color.NRGBA{255, 0, 0, 255})
	img.Set(2, 1, color.NRGBA{0, 128, 255, 255})

	tns, err := encoderInput(img)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{1, 3, 2, 3}; len(tns.Shape) != 4 || tns.Shape[0] != want[0] || tns.Shape[1] != want[1] || tns.Shape[2] != want[2] || tns.Shape[3] != want[3] {
		t.Fatalf("shape = %v, want %v (original size, NCHW)", tns.Shape, want)
	}
	for i, v := range tns.Data {
		if v < 0 || v > 1 {
			t.Fatalf("data[%d] = %v outside [0,1] — encoder must get raw pixel/255", i, v)
		}
	}
	plane := 6
	check := func(c, y, x int, want float32) {
		t.Helper()
		if got := tns.Data[c*plane+y*3+x]; math.Abs(float64(got-want)) > 1e-6 {
			t.Errorf("c%d (%d,%d) = %v, want %v", c, x, y, got, want)
		}
	}
	check(0, 0, 0, 1)
	check(1, 0, 0, 0)
	check(1, 1, 2, 128.0/255)
	check(2, 1, 2, 1)
}

// TestBatchedTensors_OriginalPixelCoords: the decoder rescales coords by orig_im_size
// itself, so a box must be passed in ORIGINAL pixels as two points labelled 2 and 3.
func TestBatchedTensors_OriginalPixelCoords(t *testing.T) {
	sets, err := promptToPointSets(models.Prompt{Boxes: [][4]float64{{11, 52, 305, 420}}})
	if err != nil || len(sets) != 1 {
		t.Fatalf("promptToPointSets: %v, %d sets", err, len(sets))
	}
	coords, labels := sets[0].batchedTensors()
	if s := coords.Shape; len(s) != 4 || s[0] != 1 || s[1] != 1 || s[2] != 2 || s[3] != 2 {
		t.Fatalf("coords shape %v, want [1 1 2 2]", s)
	}
	if s := labels.Shape; len(s) != 3 || s[2] != 2 {
		t.Fatalf("labels shape %v, want [1 1 2]", s)
	}
	wantC := []float32{11, 52, 316, 472}
	for i, w := range wantC {
		if coords.Data[i] != w {
			t.Errorf("coords[%d] = %v, want %v (unscaled original pixels)", i, coords.Data[i], w)
		}
	}
	if labels.Data[0] != 2 || labels.Data[1] != 3 {
		t.Errorf("labels = %v, want [2 3]", labels.Data)
	}
}

// TestPickBestMask_RealShapes: [1,1,3,H,W] masks at original resolution with UNSORTED
// IoU predictions (as the real export returns, e.g. [0.70 0.96 0.99]) → argmax plane.
func TestPickBestMask_RealShapes(t *testing.T) {
	const H, W = 5, 7
	masks := make([]float32, 3*H*W)
	for c := 0; c < 3; c++ {
		for i := 0; i < H*W; i++ {
			masks[c*H*W+i] = float32(c*100 + i)
		}
	}
	outs := []engine.Tensor{
		engine.F32(masks, 1, 1, 3, H, W),
		engine.F32([]float32{0.70, 0.96, 0.99}, 1, 1, 3),
		engine.F32(make([]float32, 3*256*256), 1, 3, 256, 256),
	}
	plane, mH, mW, score, err := pickBestMask(realDecoderOutputs, outs)
	if err != nil {
		t.Fatal(err)
	}
	if mH != H || mW != W {
		t.Fatalf("dims = %dx%d, want %dx%d", mW, mH, W, H)
	}
	if math.Abs(score-0.99) > 1e-6 {
		t.Errorf("score = %v, want 0.99", score)
	}
	if len(plane) != H*W || plane[0] != 200 {
		t.Errorf("picked plane starts with %v (len %d), want candidate 2", plane[0], len(plane))
	}
}

// TestMaskToResult_OriginalResolution: the plane is already at the original size and
// is thresholded directly at logit >= 0; BBox is the tight [x,y,w,h] in original pixels.
func TestMaskToResult_OriginalResolution(t *testing.T) {
	const H, W = 6, 8
	plane := make([]float32, H*W)
	for i := range plane {
		plane[i] = -5
	}
	// foreground rectangle x∈[2,4], y∈[1,3]; one pixel exactly 0 (>= 0 → foreground).
	for y := 1; y <= 3; y++ {
		for x := 2; x <= 4; x++ {
			plane[y*W+x] = 3
		}
	}
	plane[3*W+4] = 0

	mk, err := maskToResult(plane, W, H, 0.9, W, H)
	if err != nil {
		t.Fatal(err)
	}
	if want := [4]float64{2, 1, 3, 3}; mk.BBox != want {
		t.Errorf("bbox = %v, want %v", mk.BBox, want)
	}
	bin := decodeRLE(t, mk.RLE, H, W)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			want := y >= 1 && y <= 3 && x >= 2 && x <= 4
			if bin[y*W+x] != want {
				t.Fatalf("pixel (%d,%d) = %v, want %v", x, y, bin[y*W+x], want)
			}
		}
	}
	if mk.Conf != 0.9 {
		t.Errorf("conf = %v", mk.Conf)
	}
	if _, err := maskToResult(plane[:5], W, H, 0, W, H); err == nil {
		t.Error("expected error for plane/shape mismatch")
	}
}

// fakeRunner records the inputs and returns real-shaped outputs.
type fakeRunner struct {
	H, W   int
	encIn  engine.Tensor
	decIns map[string]engine.Tensor
}

func (f *fakeRunner) InputNames(role string) []string {
	if role == roleEncoder {
		return []string{"batched_images"}
	}
	return []string{"image_embeddings", "batched_point_coords", "batched_point_labels", "orig_im_size"}
}

func (f *fakeRunner) OutputNames(role string) []string {
	if role == roleEncoder {
		return []string{"image_embeddings"}
	}
	return realDecoderOutputs
}

func (f *fakeRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role == roleEncoder {
		f.encIn = in["batched_images"]
		return []engine.Tensor{engine.F32(make([]float32, 256*64*64), 1, 256, 64, 64)}, nil
	}
	f.decIns = in
	// Candidate 1 (highest IoU) = box region from the prompt, in original pixels.
	c := in["batched_point_coords"].Data
	masks := make([]float32, 3*f.H*f.W)
	for i := range masks {
		masks[i] = -1
	}
	for y := int(c[1]); y < int(c[3]); y++ {
		for x := int(c[0]); x < int(c[2]); x++ {
			masks[f.H*f.W+y*f.W+x] = 1
		}
	}
	return []engine.Tensor{
		engine.F32(masks, 1, 1, 3, int64(f.H), int64(f.W)),
		engine.F32([]float32{0.2, 0.95, 0.5}, 1, 1, 3),
		engine.F32(make([]float32, 3*256*256), 1, 3, 256, 256),
	}, nil
}

// TestInfer_PassesOriginalGeometry: end-to-end with a fake runner on a non-square image —
// encoder gets [1,3,H,W], decoder gets unscaled coords + orig_im_size=[H,W], and the
// mask BBox comes back in original coordinates.
func TestInfer_PassesOriginalGeometry(t *testing.T) {
	const H, W = 48, 64
	m, err := New(models.Config{Name: "efficient-sam", Files: map[string]string{roleEncoder: "e", roleDecoder: "d"}})
	if err != nil {
		t.Fatal(err)
	}
	fr := &fakeRunner{H: H, W: W}
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	res, err := m.(models.PipelineModel).Infer(img, models.Prompt{Boxes: [][4]float64{{10, 5, 30, 20}}}, fr)
	if err != nil {
		t.Fatal(err)
	}
	if s := fr.encIn.Shape; len(s) != 4 || s[2] != H || s[3] != W {
		t.Errorf("encoder input shape %v, want [1 3 %d %d]", s, H, W)
	}
	if got := fr.decIns["orig_im_size"].DataI64; len(got) != 2 || got[0] != H || got[1] != W {
		t.Errorf("orig_im_size = %v, want [%d %d]", got, H, W)
	}
	if got := fr.decIns["batched_point_coords"].Data; got[0] != 10 || got[1] != 5 || got[2] != 40 || got[3] != 25 {
		t.Errorf("coords = %v, want [10 5 40 25]", got)
	}
	if len(res.Masks) != 1 {
		t.Fatalf("masks = %d, want 1", len(res.Masks))
	}
	if want := [4]float64{10, 5, 30, 20}; res.Masks[0].BBox != want {
		t.Errorf("mask bbox = %v, want %v", res.Masks[0].BBox, want)
	}
	if math.Abs(res.Masks[0].Conf-0.95) > 1e-6 {
		t.Errorf("conf = %v, want 0.95", res.Masks[0].Conf)
	}
}
