package nanosam

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

// Shapes per the upstream NanoSAM exports (github.com/NVIDIA-AI-IOT/nanosam,
// tools/export_*_onnx.py). NOT verified against real weight files — none are present in
// models/nano-sam:
//
//	encoder in  image [1,3,1024,1024]          out image_embeddings [1,256,64,64]
//	decoder out iou_predictions [1,M], low_res_masks [1,M,256,256] (M = 4)

func decodeRLE(t *testing.T, rle string, h, w int) []bool {
	t.Helper()
	bin := make([]bool, h*w)
	pos, val := 0, false
	for _, f := range strings.Fields(rle) {
		n, err := strconv.Atoi(f)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < n; i++ {
			if pos >= h*w {
				t.Fatalf("RLE overflows %dx%d", h, w)
			}
			bin[(pos%h)*w+pos/h] = val // column-major
			pos++
		}
		val = !val
	}
	if pos != h*w {
		t.Fatalf("RLE covers %d px, want %d (mask must be at ORIGINAL resolution)", pos, h*w)
	}
	return bin
}

// syntheticLowRes builds decoder outputs with the real [1,4,256,256] / [1,4] shapes. The
// best channel (2) holds a disc of radius 30 centred at low-res (100,60), and +10 garbage
// in the PADDING region beyond (limX, limY) that upscale_mask must crop away.
func syntheticLowRes(limX, limY int) []engine.Tensor {
	const lr = 256
	data := make([]float32, 4*lr*lr)
	for i := range data {
		data[i] = -5
	}
	for y := 0; y < lr; y++ {
		for x := 0; x < lr; x++ {
			v := float32(30 - math.Sqrt(float64((x-100)*(x-100)+(y-60)*(y-60))))
			if x >= limX || y >= limY {
				v = 10
			}
			data[2*lr*lr+y*lr+x] = v
		}
	}
	for y := 0; y < 20; y++ { // a blob in a lower-IoU channel that must not be picked
		for x := 0; x < 20; x++ {
			data[y*lr+x] = 10
		}
	}
	return []engine.Tensor{
		engine.F32([]float32{0.1, 0.2, 0.95, 0.3}, 1, 4),
		engine.F32(data, 1, 4, lr, lr),
	}
}

// pickBestMask must follow upstream predictor.py upscale_mask: crop the 256 grid to the
// image region, bilinear (align_corners=False) to (H, W), threshold > 0. Expected values
// were produced by running upstream upscale_mask (torch 2.10) on the same logits.
func TestPickBestMask_UpscaleMaskMatchesUpstream(t *testing.T) {
	cases := []struct {
		w, h       int
		limX, limY int
		area       int
		bbox       [4]float64
	}{
		{500, 300, 256, 153, 10820, [4]float64{138, 60, 117, 117}}, // landscape
		{300, 500, 153, 256, 10823, [4]float64{138, 60, 118, 117}}, // portrait
	}
	for _, tc := range cases {
		if lx, ly := lowResCrop(tc.w, tc.h, 256, 256); lx != tc.limX || ly != tc.limY {
			t.Fatalf("%dx%d lowResCrop = (%d,%d), want (%d,%d)", tc.w, tc.h, lx, ly, tc.limX, tc.limY)
		}
		outs := syntheticLowRes(tc.limX, tc.limY)
		mk, err := pickBestMask(outs, []string{"iou_predictions", "low_res_masks"}, tc.w, tc.h)
		if err != nil {
			t.Fatal(err)
		}
		if mk.BBox != tc.bbox {
			t.Errorf("%dx%d bbox = %v, want %v (original-image coords)", tc.w, tc.h, mk.BBox, tc.bbox)
		}
		if math.Abs(mk.Conf-0.95) > 1e-6 {
			t.Errorf("conf = %v, want 0.95 (best channel)", mk.Conf)
		}
		bin := decodeRLE(t, mk.RLE, tc.h, tc.w)
		area := 0
		for _, v := range bin {
			if v {
				area++
			}
		}
		if area != tc.area {
			t.Errorf("%dx%d area = %d, want %d (torch upscale_mask)", tc.w, tc.h, area, tc.area)
		}
	}
}

func TestPickBestMask_Errors(t *testing.T) {
	short := []engine.Tensor{{Shape: []int64{1, 4, 256, 256}, Data: make([]float32, 8)}}
	if _, err := pickBestMask(short, []string{"low_res_masks"}, 10, 10); err == nil {
		t.Fatal("expected error for data shorter than shape")
	}
	if _, err := pickBestMask(nil, nil, 10, 10); err == nil {
		t.Fatal("expected error when no mask tensor")
	}
}

func TestEncoderInput_RealShapeAndPadding(t *testing.T) {
	const w, h = 400, 200
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: 255, G: 128, B: 0, A: 255})
		}
	}
	tensor, scale, err := encoderInput(img)
	if err != nil {
		t.Fatal(err)
	}
	want := []int64{1, 3, 1024, 1024}
	for i, d := range want {
		if tensor.Dim(i) != d {
			t.Fatalf("shape %v, want %v", tensor.Shape, want)
		}
	}
	if math.Abs(float64(scale)-1024.0/400) > 1e-6 {
		t.Fatalf("scale = %v, want %v", scale, 1024.0/400)
	}
	const plane = 1024 * 1024
	if got, want := tensor.Data[0], (1-imagenetMean[0])/imagenetStd[0]; math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("R(0,0) = %v, want %v", got, want)
	}
	// Row 600 is below the 512-row content: zero padding (bottom/right).
	if tensor.Data[plane+600*1024+10] != 0 {
		t.Fatal("padding region must be 0")
	}
}

func TestEncodePrompt_BoxScaled(t *testing.T) {
	pc, pl, err := encodePrompt(models.Prompt{Boxes: [][4]float64{{10, 20, 30, 40}}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if pc.Dim(0) != 1 || pc.Dim(1) != 2 || pc.Dim(2) != 2 || pl.Dim(1) != 2 {
		t.Fatalf("shapes %v %v, want [1,2,2] [1,2]", pc.Shape, pl.Shape)
	}
	want := []float32{20, 40, 80, 120}
	for i, v := range want {
		if pc.Data[i] != v {
			t.Fatalf("coords = %v, want %v", pc.Data, want)
		}
	}
	if pl.Data[0] != 2 || pl.Data[1] != 3 {
		t.Fatalf("labels = %v, want [2 3]", pl.Data)
	}
}
