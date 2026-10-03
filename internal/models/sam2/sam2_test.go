package sam2

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

// Real tensor shapes, inspected with onnxruntime on models/sam2/sam2_tiny_{encoder,decoder}.onnx
// (SharpAI/sam2-hiera-tiny-onnx):
//
//	encoder in  image [1,3,1024,1024]
//	decoder out masks [1,3,256,256] (low-res logits, clamped ±32), iou_predictions [1,3]

func uniformImage(w, h int, c color.NRGBA) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

// The encoder input must be the image SQUASHED to 1024×1024 (upstream SAM2Transforms),
// normalized with ImageNet mean/std, with per-axis prompt scale factors — no padding.
func TestEncoderInput_SquashRealShape(t *testing.T) {
	const w, h = 300, 120 // landscape: a letterbox would leave the bottom rows as padding
	c := color.NRGBA{R: 200, G: 100, B: 50, A: 255}
	tensor, sx, sy, err := encoderInput(uniformImage(w, h, c))
	if err != nil {
		t.Fatal(err)
	}
	wantShape := []int64{1, 3, 1024, 1024}
	for i, d := range wantShape {
		if tensor.Dim(i) != d {
			t.Fatalf("shape %v, want %v", tensor.Shape, wantShape)
		}
	}
	if math.Abs(float64(sx)-1024.0/w) > 1e-6 || math.Abs(float64(sy)-1024.0/h) > 1e-6 {
		t.Fatalf("scale = (%v,%v), want per-axis (%v,%v)", sx, sy, 1024.0/w, 1024.0/h)
	}
	raw := [3]float32{200, 100, 50}
	const plane = 1024 * 1024
	for ch := 0; ch < 3; ch++ {
		want := (raw[ch]/255 - sam2Mean[ch]) / sam2Std[ch]
		// (0,0) and the bottom-right corner: both must be image, not padding.
		for _, idx := range []int{0, plane - 1} {
			if got := tensor.Data[ch*plane+idx]; math.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("ch %d idx %d = %v, want %v", ch, idx, got, want)
			}
		}
	}
}

func TestScaledPrompt_PerAxis(t *testing.T) {
	ps := pointSet{coords: []float64{100, 50, 300, 120}, labels: []int64{2, 3}}
	coords, labels := ps.scaledPrompt(1024.0/300, 1024.0/120)
	want := [][2]float32{{100 * 1024.0 / 300, 50 * 1024.0 / 120}, {1024, 1024}}
	for i := range want {
		for k := 0; k < 2; k++ {
			if math.Abs(float64(coords[i][k]-want[i][k])) > 1e-3 {
				t.Fatalf("coords[%d] = %v, want %v", i, coords[i], want[i])
			}
		}
	}
	if labels[0] != 2 || labels[1] != 3 {
		t.Fatalf("labels = %v, want [2 3]", labels)
	}
}

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
			x, y := pos/h, pos%h // column-major
			bin[y*w+x] = val
			pos++
		}
		val = !val
	}
	if pos != h*w {
		t.Fatalf("RLE covers %d px, want %d (mask must be at ORIGINAL resolution)", pos, h*w)
	}
	return bin
}

// The best low-res candidate (by iou_predictions) must be upsampled to the ORIGINAL size
// before thresholding: RLE covers H×W and BBox is in original pixels.
func TestPickBestMask_RealShapesUpsampledToOriginal(t *testing.T) {
	const lr = 256
	const origW, origH = 512, 1024 // ×2 in x, ×4 in y vs the 256 grid
	data := make([]float32, 3*lr*lr)
	for i := range data {
		data[i] = -10
	}
	// channel 1 (best by IoU): low-res rows 64..127, cols 32..95 positive.
	for y := 64; y < 128; y++ {
		for x := 32; x < 96; x++ {
			data[1*lr*lr+y*lr+x] = 10
		}
	}
	// channel 0 (worse IoU): a different blob that must NOT be chosen.
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			data[y*lr+x] = 10
		}
	}
	masks := engine.F32(data, 1, 3, lr, lr)
	iou := engine.F32([]float32{0.5, 0.9, 0.2}, 1, 3)

	mk, err := pickBestMask(masks, iou, origW, origH)
	if err != nil {
		t.Fatal(err)
	}
	// Zero crossings at low-res 31.5/95.5 (x) and 63.5/127.5 (y) map to
	// x ∈ [64,191], y ∈ [256,511] in the original frame.
	want := [4]float64{64, 256, 128, 256}
	if mk.BBox != want {
		t.Fatalf("bbox = %v, want %v (original-image coords)", mk.BBox, want)
	}
	if math.Abs(mk.Conf-0.9) > 1e-6 {
		t.Fatalf("conf = %v, want 0.9", mk.Conf)
	}
	bin := decodeRLE(t, mk.RLE, origH, origW)
	area := 0
	for _, v := range bin {
		if v {
			area++
		}
	}
	// torch F.interpolate on the same logits gives 32764: the 4 corner pixels fall
	// below 0 under bilinear blending.
	if area != 128*256-4 {
		t.Fatalf("area = %d, want %d (torch reference)", area, 128*256-4)
	}
	if !bin[300*origW+100] || bin[300*origW+10] {
		t.Fatal("RLE content does not match the expected rectangle")
	}
}

func TestPickBestMask_RejectsShortData(t *testing.T) {
	bad := engine.Tensor{Shape: []int64{1, 3, 256, 256}, Data: make([]float32, 10)}
	if _, err := pickBestMask(bad, engine.F32([]float32{1, 0, 0}, 1, 3), 10, 10); err == nil {
		t.Fatal("expected error for data shorter than shape")
	}
}

func TestFindMaskAndIoU_RealNames(t *testing.T) {
	outs := []engine.Tensor{
		engine.F32(make([]float32, 3*256*256), 1, 3, 256, 256),
		engine.F32(make([]float32, 3), 1, 3),
	}
	m, i := findMaskAndIoU([]string{"masks", "iou_predictions"}, outs)
	if m != &outs[0] || i != &outs[1] {
		t.Fatal("masks/iou_predictions not matched by name")
	}
}

func TestPromptToPointSets_BoxLabels(t *testing.T) {
	sets, err := promptToPointSets(models.Prompt{Boxes: [][4]float64{{10, 20, 30, 40}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(sets) != 1 || sets[0].labels[0] != 2 || sets[0].labels[1] != 3 ||
		sets[0].coords[2] != 40 || sets[0].coords[3] != 60 {
		t.Fatalf("box → %+v, want tl(10,20) label 2 + br(40,60) label 3", sets)
	}
	if _, err := promptToPointSets(models.Prompt{}); err == nil {
		t.Fatal("empty prompt must error")
	}
}
