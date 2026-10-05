package owlvit

import (
	"fmt"
	"image"
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/geom"
)

// Real I/O shapes of models/owlvit-base/owlv2_base_patch16.onnx (inspected with onnxruntime):
//
//	in  query_pixel_values   [1, 3, 960, 960]
//	in  query_image_features [1, 3, 960, 960]
//	out logits               [1, 3600, 1]
//	out pred_boxes           [1, 3600, 4]   cxcywh, normalized to the padded square
const (
	realInput   = 960
	realPatches = 3600
)

// newTestModel builds the model with a low template threshold (0.1), so the decode tests below
// see every planted patch; the shipped default (0.9) is tested in TestInfer_ThresholdSelection.
func newTestModel(t *testing.T, maxDet int) *owlVIT {
	t.Helper()
	m, err := New(models.Config{Width: realInput, Height: realInput, InstanceSimThreshold: 0.1, MaxDet: maxDet, InstancePatchSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	return m.(*owlVIT)
}

func rgbImage(w, h int, pix []uint8) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < w*h; i++ {
		img.Pix[4*i], img.Pix[4*i+1], img.Pix[4*i+2], img.Pix[4*i+3] = pix[3*i], pix[3*i+1], pix[3*i+2], 255
	}
	return img
}

func assertClose(t *testing.T, name string, got []float32, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > tol {
			t.Fatalf("%s[%d] = %.6f, want %.6f (HF Owlv2ImageProcessor)", name, i, got[i], want[i])
		}
	}
}

// Golden values produced by transformers 5.9 Owlv2ImageProcessor(do_normalize=False) on
// the same pixels: downsampling path (pad 5×7 → 7×7, Gaussian anti-alias σ=2/3, bilinear → 3×3).
func TestPadResize_MatchesHFDownsample(t *testing.T) {
	pix := []uint8{106, 152, 249, 131, 184, 200, 0, 21, 253, 147, 202, 107, 249, 169, 138, 149, 119, 166, 224, 148, 172, 93, 167, 142, 248, 26, 81, 218, 150, 66, 2, 191, 60, 129, 179, 90, 69, 104, 110, 97, 157, 106, 152, 251, 62, 7, 248, 171, 33, 123, 79, 176, 37, 20, 94, 49, 149, 127, 206, 244, 28, 119, 54, 0, 192, 233, 18, 116, 191, 165, 184, 184, 199, 165, 253, 174, 33, 113, 1, 85, 117, 119, 202, 235, 124, 228, 99, 219, 144, 255, 80, 160, 144, 242, 18, 75, 183, 96, 223, 141, 165, 158, 48, 189, 33}
	want := []float64{0.568282, 0.397848, 0.469321, 0.255193, 0.68659, 0.309503, 0.089444, 0.107328, 0.062282, 0.507214, 0.613326, 0.534139, 0.663118, 0.430512, 0.52031, 0.093575, 0.035316, 0.087632, 0.561825, 0.359257, 0.523315, 0.717801, 0.632502, 0.591645, 0.102292, 0.066106, 0.06847}
	got, side := padResizeOWLv2(rgbImage(7, 5, pix), 3)
	if side != 7 {
		t.Fatalf("side = %d, want 7", side)
	}
	assertClose(t, "chw", got, want, 1e-5)
}

// Upsampling path (pad 2×4 → 4×4 with black at the BOTTOM, no blur, bilinear → 6×6).
func TestPadResize_MatchesHFUpsample(t *testing.T) {
	pix := []uint8{99, 206, 239, 189, 230, 118, 144, 73, 8, 228, 231, 190, 155, 112, 158, 208, 7, 204, 143, 113, 181, 231, 80, 27}
	want := []float64{0.388235, 0.564706, 0.711765, 0.594118, 0.729412, 0.894118, 0.498039, 0.638235, 0.742484, 0.598693, 0.731373, 0.9, 0.506536, 0.593137, 0.644335, 0.502723, 0.611111, 0.754902, 0.101307, 0.118627, 0.128867, 0.100545, 0.122222, 0.15098, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.807843, 0.854902, 0.799346, 0.388889, 0.596078, 0.905882, 0.623529, 0.544118, 0.448039, 0.381373, 0.487255, 0.609804, 0.366013, 0.194444, 0.08061, 0.311547, 0.315359, 0.261438, 0.073203, 0.038889, 0.016122, 0.062309, 0.063072, 0.052288, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.937255, 0.7, 0.39085, 0.103268, 0.388235, 0.745098, 0.778431, 0.704902, 0.587909, 0.414052, 0.398039, 0.42549, 0.51634, 0.591503, 0.654139, 0.60403, 0.339869, 0.088235, 0.103268, 0.118301, 0.130828, 0.120806, 0.067974, 0.017647, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	got, side := padResizeOWLv2(rgbImage(4, 2, pix), 6)
	if side != 4 {
		t.Fatalf("side = %d, want 4", side)
	}
	assertClose(t, "chw", got, want, 1e-5)
}

// A landscape 640×427 scene is padded (not squashed): content keeps its aspect ratio, the
// bottom of the 960×960 tensor is black, and meta carries one scale 960/640 on both axes.
func TestPreprocess_PadsToSquareNotSquash(t *testing.T) {
	m := newTestModel(t, 10)
	img := image.NewNRGBA(image.Rect(0, 0, 640, 427))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 200, 100, 50, 255
	}
	tensor, meta, err := m.preprocessImage(img)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{1, 3, realInput, realInput}; len(tensor.Shape) != 4 || tensor.Shape[2] != want[2] || tensor.Shape[3] != want[3] {
		t.Fatalf("shape %v, want %v", tensor.Shape, want)
	}
	if meta.ScaleX != 1.5 || meta.ScaleY != 1.5 || meta.PadX != 0 || meta.PadY != 0 {
		t.Fatalf("meta %+v, want scale 1.5 on both axes, no pad offset", meta)
	}
	plane := realInput * realInput
	content := (200.0/255 - float64(clipMean[0])) / float64(clipStd[0])
	black := (0 - float64(clipMean[0])) / float64(clipStd[0])
	// Content ends at 427*1.5 = 640.5 rows; row 300 is content, row 800 is padding.
	if v := float64(tensor.Data[0*plane+300*realInput+500]); math.Abs(v-content) > 1e-3 {
		t.Errorf("content pixel = %v, want %v", v, content)
	}
	if v := float64(tensor.Data[0*plane+800*realInput+500]); math.Abs(v-black) > 1e-3 {
		t.Errorf("pad pixel = %v, want black %v", v, black)
	}
}

// realBoxes returns a [1,3600,4] pred_boxes buffer filled with tiny boxes in the top-left.
func realBoxes() []float32 {
	b := make([]float32, realPatches*4)
	for p := 0; p < realPatches; p++ {
		b[p*4], b[p*4+1], b[p*4+2], b[p*4+3] = 0.01, 0.01, 0.01, 0.01
	}
	return b
}

func setBox(b []float32, p int, cx, cy, w, h float32) {
	b[p*4], b[p*4+1], b[p*4+2], b[p*4+3] = cx, cy, w, h
}

// Regression: maxDet used to be applied BEFORE NMS, so the budget went to near-duplicates of
// the strongest instance and the other instances were lost.
func TestPostprocess_NMSBeforeMaxDet(t *testing.T) {
	m := newTestModel(t, 3)
	scores := make([]float64, realPatches)
	boxes := realBoxes()
	// Instance A: 5 overlapping patches with the highest scores.
	for k := 0; k < 5; k++ {
		setBox(boxes, 100+k, 0.2+0.002*float32(k), 0.2, 0.1, 0.1)
		scores[100+k] = 0.99 - 0.001*float64(k)
	}
	// Instances B and C: lower scores, far apart.
	setBox(boxes, 2000, 0.5, 0.2, 0.1, 0.1)
	scores[2000] = 0.8
	setBox(boxes, 3000, 0.8, 0.2, 0.1, 0.1)
	scores[3000] = 0.7
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 640, ScaleX: 1.5, ScaleY: 1.5}

	res, err := m.postprocess(scores, boxes, realPatches, meta, m.simThreshold)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Detections) != 3 {
		t.Fatalf("got %d detections, want 3 (one per instance): %+v", len(res.Detections), res.Detections)
	}
	for i := 1; i < len(res.Detections); i++ {
		if res.Detections[i].Conf > res.Detections[i-1].Conf {
			t.Fatalf("not sorted by confidence: %+v", res.Detections)
		}
	}
	if res.Detections[1].Conf != 0.8 || res.Detections[2].Conf != 0.7 {
		t.Fatalf("instances B/C missing: %+v", res.Detections)
	}
}

// Boxes are normalized to the PADDED square (side = max(W,H) = 640 for a 640×427 image),
// mapped with that single scale, clamped to the image, and dropped if they fall entirely in
// the padding.
func TestPostprocess_PaddedSquareMappingAndClamp(t *testing.T) {
	m := newTestModel(t, 10)
	scores := make([]float64, realPatches)
	boxes := realBoxes()
	setBox(boxes, 10, 0.5, 0.5, 0.2, 0.2) // → x 256..384, y 256..384
	scores[10] = 0.9
	setBox(boxes, 20, 0.9, 0.6, 0.3, 0.2) // x 480..672 → clamp 640; y 320..448 → clamp 427
	scores[20] = 0.8
	setBox(boxes, 30, 0.5, 0.9, 0.1, 0.05) // y 560..592: entirely in the bottom padding
	scores[30] = 0.7
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 427, ScaleX: 1.5, ScaleY: 1.5}

	res, err := m.postprocess(scores, boxes, realPatches, meta, m.simThreshold)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Detections) != 2 {
		t.Fatalf("got %d detections, want 2: %+v", len(res.Detections), res.Detections)
	}
	want := [][4]float64{{256, 256, 128, 128}, {480, 320, 160, 107}}
	for i, w := range want {
		for k := 0; k < 4; k++ {
			if math.Abs(res.Detections[i].BBox[k]-w[k]) > 1e-3 {
				t.Fatalf("det %d bbox %v, want %v", i, res.Detections[i].BBox, w)
			}
		}
	}
}

func TestPostprocess_ShortBoxesIsError(t *testing.T) {
	m := newTestModel(t, 10)
	_, err := m.postprocess(make([]float64, realPatches), make([]float32, 8), realPatches,
		models.PreprocessMeta{OrigWidth: 10, OrigHeight: 10, ScaleX: 96, ScaleY: 96}, m.simThreshold)
	if err == nil {
		t.Fatal("want error for pred_boxes shorter than P*4, got nil")
	}
}

// fakeRunner returns tensors with the real output names and shapes.
type fakeRunner struct {
	logits [][]float32 // per call
	boxes  []float32
	calls  int
}

func (f *fakeRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	for _, name := range []string{"query_pixel_values", "query_image_features"} {
		s := in[name].Shape
		if len(s) != 4 || s[0] != 1 || s[1] != 3 || s[2] != realInput || s[3] != realInput {
			return nil, fmt.Errorf("bad input shape for %s: %v", name, s)
		}
	}
	l := f.logits[f.calls]
	f.calls++
	return []engine.Tensor{engine.F32(l, 1, realPatches, 1), engine.F32(f.boxes, 1, realPatches, 4)}, nil
}
func (f *fakeRunner) InputNames(string) []string {
	return []string{"query_pixel_values", "query_image_features"}
}
func (f *fakeRunner) OutputNames(string) []string { return []string{"logits", "pred_boxes"} }

// The score threshold of a (template-prompted) request: the request's box_threshold when it
// sets one, else the manifest's instance.sim_threshold, else 0.9 (HF's image-guided example,
// post_process_image_guided_detection(threshold=0.9)). Three far-apart patches score 0.95, 0.6
// and 0.2, so the threshold in force is read off the number of detections.
func TestInfer_ThresholdSelection(t *testing.T) {
	boxes := realBoxes()
	logits := make([]float32, realPatches)
	for i := range logits {
		logits[i] = -20
	}
	for k, p := range []int{100, 2000, 3500} {
		setBox(boxes, p, 0.15+0.35*float32(k), 0.2, 0.1, 0.1)
	}
	logit := func(p float64) float32 { return float32(math.Log(p / (1 - p))) }
	logits[100], logits[2000], logits[3500] = logit(0.95), logit(0.6), logit(0.2)
	tImg := image.NewNRGBA(image.Rect(0, 0, 30, 40))

	for _, c := range []struct {
		name      string
		manifest  float64 // instance.sim_threshold (0 = absent)
		boxThresh float64 // request box_threshold (0 = absent)
		want      int
	}{
		{"default 0.9", 0, 0, 1},
		{"default, request 0.5", 0, 0.5, 2},
		{"default, request 0.1", 0, 0.1, 3},
		{"manifest 0.5", 0.5, 0, 2},
		{"manifest 0.5, request 0.9", 0.5, 0.9, 1},
		{"manifest 0.5, request 0.1", 0.5, 0.1, 3},
	} {
		base, err := New(models.Config{Width: realInput, Height: realInput, MaxDet: 10, InstanceSimThreshold: c.manifest})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		m := base.(*owlVIT)
		r := &fakeRunner{logits: [][]float32{logits}, boxes: boxes}
		p := models.Prompt{TemplateImages: []image.Image{tImg}, BoxThresh: c.boxThresh}
		res, err := m.Infer(image.NewNRGBA(image.Rect(0, 0, 400, 400)), p, r)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if len(res.Detections) != c.want {
			t.Errorf("%s: %d detections, want %d: %+v", c.name, len(res.Detections), c.want, res.Detections)
		}
		wantModel := c.manifest
		if wantModel == 0 {
			wantModel = 0.9
		}
		if m.simThreshold != wantModel {
			t.Errorf("%s: model threshold %v after the request, want %v", c.name, m.simThreshold, wantModel)
		}
	}
}

// postprocess.conf_threshold used to override the template threshold (the shipped manifest's
// 0.1 applied to every request). The graph has no text-query path, so the value could only be
// ignored: a load error instead. An instance.sim_threshold outside [0, 1) is refused too.
func TestNew_RefusesConfThresholdAndBadSimThreshold(t *testing.T) {
	for _, cfg := range []models.Config{
		{Width: realInput, Height: realInput, ConfThresh: 0.1},
		{Width: realInput, Height: realInput, ConfThresh: 0.1, InstanceSimThreshold: 0.9},
		{Width: realInput, Height: realInput, InstanceSimThreshold: 1},
		{Width: realInput, Height: realInput, InstanceSimThreshold: math.NaN()},
	} {
		if _, err := New(cfg); err == nil {
			t.Errorf("New(conf %v, sim %v) = nil error, want a load error", cfg.ConfThresh, cfg.InstanceSimThreshold)
		}
	}
}

// Two templates: per-patch scores are the max over templates; boxes map to original pixels.
func TestInfer_RealShapesTwoTemplates(t *testing.T) {
	m := newTestModel(t, 10)
	boxes := realBoxes()
	setBox(boxes, 5, 0.25, 0.25, 0.1, 0.1)
	setBox(boxes, 7, 0.75, 0.25, 0.1, 0.1)
	l1 := make([]float32, realPatches)
	l2 := make([]float32, realPatches)
	for i := range l1 {
		l1[i], l2[i] = -20, -20
	}
	l1[5] = 3 // template 1 sees patch 5
	l2[7] = 2 // template 2 sees patch 7
	r := &fakeRunner{logits: [][]float32{l1, l2}, boxes: boxes}
	scene := image.NewNRGBA(image.Rect(0, 0, 400, 200))
	tImg := image.NewNRGBA(image.Rect(0, 0, 30, 40))

	res, err := m.Infer(scene, models.Prompt{TemplateImages: []image.Image{tImg, tImg}}, r)
	if err != nil {
		t.Fatal(err)
	}
	if r.calls != 2 || len(res.Detections) != 2 {
		t.Fatalf("calls=%d dets=%+v, want 2 calls and 2 detections", r.calls, res.Detections)
	}
	// side = 400 → patch 5: x 80..120, y 80..120; patch 7: x 280..320, y 80..120 (clamped: y2 < 200).
	want := [][4]float64{{80, 80, 40, 40}, {280, 80, 40, 40}}
	for i, w := range want {
		for k := 0; k < 4; k++ {
			if math.Abs(res.Detections[i].BBox[k]-w[k]) > 1e-3 {
				t.Fatalf("det %d bbox %v, want %v", i, res.Detections[i].BBox, w)
			}
		}
	}
	if math.Abs(res.Detections[0].Conf-geom.Sigmoid(3)) > 1e-9 {
		t.Fatalf("conf %v, want sigmoid(3)", res.Detections[0].Conf)
	}
}
