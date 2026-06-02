package rfdetr

import (
	"image"
	"image/color"
	"math"
	"testing"

	"visionserve/internal/models"
)

// fillRGBA tạo ảnh test NRGBA kích thước wxh, tô đồng màu (r,g,b).
func fillRGBA(w, h int, r, g, b uint8) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	c := color.NRGBA{R: r, G: g, B: b, A: 255}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

// RF-DETR preprocess với letterbox: tensor ra đúng shape NCHW [1,3,H,W],
// meta giữ kích thước gốc, và ảnh KHÔNG vuông phải sinh padding đúng một chiều.
func TestPreprocessLetterboxNonSquare(t *testing.T) {
	m := &rfDETR{cfg: models.Config{
		Name:      "rf-detr",
		Width:     64,
		Height:    64,
		Letterbox: true,
	}}

	// Ảnh 80x40 (rộng hơn cao) -> letterbox vào 64x64.
	// scale = min(64/80, 64/40) = min(0.8, 1.6) = 0.8
	// newW = 80*0.8 = 64, newH = 40*0.8 = 32
	// padX = (64-64)/2 = 0, padY = (64-32)/2 = 16
	img := fillRGBA(80, 40, 10, 20, 30)

	ten, meta, err := m.preprocess(img)
	if err != nil {
		t.Fatalf("preprocess lỗi: %v", err)
	}

	wantShape := []int64{1, 3, 64, 64}
	if len(ten.Shape) != 4 {
		t.Fatalf("shape len = %d, muốn 4 (NCHW)", len(ten.Shape))
	}
	for i := range wantShape {
		if ten.Shape[i] != wantShape[i] {
			t.Fatalf("shape = %v, muốn %v", ten.Shape, wantShape)
		}
	}
	if got := int64(len(ten.Data)); got != ten.NumElements() {
		t.Fatalf("len(Data)=%d khác NumElements=%d", got, ten.NumElements())
	}

	if meta.OrigWidth != 80 || meta.OrigHeight != 40 {
		t.Fatalf("orig = %dx%d, muốn 80x40", meta.OrigWidth, meta.OrigHeight)
	}
	if math.Abs(meta.ScaleX-0.8) > 1e-9 || math.Abs(meta.ScaleY-0.8) > 1e-9 {
		t.Fatalf("scale = (%v,%v), muốn (0.8,0.8)", meta.ScaleX, meta.ScaleY)
	}
	if meta.PadX != 0 || meta.PadY != 16 {
		t.Fatalf("pad = (%d,%d), muốn (0,16)", meta.PadX, meta.PadY)
	}
	if meta.ScaleX != meta.ScaleY {
		t.Fatalf("letterbox phải giữ tỉ lệ: ScaleX(%v) != ScaleY(%v)", meta.ScaleX, meta.ScaleY)
	}
}

// Ảnh CAO hơn rộng -> padding phải nằm ở chiều X.
func TestPreprocessLetterboxTallImage(t *testing.T) {
	m := &rfDETR{cfg: models.Config{Width: 64, Height: 64, Letterbox: true}}

	// Ảnh 40x80 (cao hơn rộng) -> scale = min(64/40,64/80)=min(1.6,0.8)=0.8
	// newW = 40*0.8 = 32, newH = 80*0.8 = 64
	// padX = (64-32)/2 = 16, padY = (64-64)/2 = 0
	img := fillRGBA(40, 80, 0, 0, 0)
	_, meta, err := m.preprocess(img)
	if err != nil {
		t.Fatalf("preprocess lỗi: %v", err)
	}
	if meta.PadX != 16 || meta.PadY != 0 {
		t.Fatalf("pad = (%d,%d), muốn (16,0)", meta.PadX, meta.PadY)
	}
	if math.Abs(meta.ScaleX-0.8) > 1e-9 {
		t.Fatalf("scale = %v, muốn 0.8", meta.ScaleX)
	}
}

// Kiểm chứng normalize NCHW theo công thức v' = (v/255 - mean[c]) / std[c].
// Lấy một pixel trong vùng ẢNH THẬT (không phải vùng pad) để so giá trị.
func TestPreprocessNormalizeNCHW(t *testing.T) {
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	m := &rfDETR{cfg: models.Config{
		Width:     64,
		Height:    64,
		Letterbox: true,
		Mean:      mean,
		Std:       std,
	}}

	// Ảnh đồng màu (100,150,200). Vùng ảnh thật nằm tại padY..padY+newH.
	// 80x40 -> newH=32, padY=16, padX=0 -> hàng y=16, cột x=0 chắc chắn là ảnh thật.
	r, g, b := uint8(100), uint8(150), uint8(200)
	img := fillRGBA(80, 40, r, g, b)

	ten, _, err := m.preprocess(img)
	if err != nil {
		t.Fatalf("preprocess lỗi: %v", err)
	}

	W, H := 64, 64
	plane := H * W
	// pixel (x=0, y=16) — trong vùng ảnh thật (padX=0, padY=16).
	px, py := 0, 16
	idx := py*W + px

	rGot := ten.Data[idx]
	gGot := ten.Data[plane+idx]
	bGot := ten.Data[2*plane+idx]

	rWant := (float32(r)/255.0 - mean[0]) / std[0]
	gWant := (float32(g)/255.0 - mean[1]) / std[1]
	bWant := (float32(b)/255.0 - mean[2]) / std[2]

	// resize linear trên ảnh đồng màu giữ nguyên màu -> sai số nhỏ.
	const tol = 1e-3
	if math.Abs(float64(rGot-rWant)) > tol {
		t.Fatalf("R = %v, muốn %v", rGot, rWant)
	}
	if math.Abs(float64(gGot-gWant)) > tol {
		t.Fatalf("G = %v, muốn %v", gGot, gWant)
	}
	if math.Abs(float64(bGot-bWant)) > tol {
		t.Fatalf("B = %v, muốn %v", bGot, bWant)
	}

	// Pixel trong vùng PAD (y=0, màu đen pad) -> normalize của 0.
	padIdx := 0*W + 0 // y=0 nằm trên vùng pad (padY=16)
	rPadWant := (0.0/255.0 - mean[0]) / std[0]
	if math.Abs(float64(ten.Data[padIdx]-rPadWant)) > tol {
		t.Fatalf("R vùng pad = %v, muốn %v (normalize của 0)", ten.Data[padIdx], rPadWant)
	}
}

// Khi Letterbox=false: resize cứng về WxH, ScaleX/ScaleY có thể khác nhau, pad=0.
func TestPreprocessNoLetterboxScalePerAxis(t *testing.T) {
	m := &rfDETR{cfg: models.Config{Width: 64, Height: 32, Letterbox: false}}

	// 80x40 -> ScaleX = 64/80 = 0.8, ScaleY = 32/40 = 0.8 (trùng nhau ở case này),
	// dùng tỉ lệ khác để chứng minh per-axis: 100x40 -> ScaleX=64/100=0.64, ScaleY=32/40=0.8
	img := fillRGBA(100, 40, 5, 5, 5)
	ten, meta, err := m.preprocess(img)
	if err != nil {
		t.Fatalf("preprocess lỗi: %v", err)
	}
	if ten.Shape[2] != 32 || ten.Shape[3] != 64 {
		t.Fatalf("shape HxW = %dx%d, muốn 32x64", ten.Shape[2], ten.Shape[3])
	}
	if math.Abs(meta.ScaleX-0.64) > 1e-9 || math.Abs(meta.ScaleY-0.8) > 1e-9 {
		t.Fatalf("scale = (%v,%v), muốn (0.64,0.8)", meta.ScaleX, meta.ScaleY)
	}
	if meta.PadX != 0 || meta.PadY != 0 {
		t.Fatalf("pad = (%d,%d), muốn (0,0) khi không letterbox", meta.PadX, meta.PadY)
	}
	if meta.OrigWidth != 100 || meta.OrigHeight != 40 {
		t.Fatalf("orig = %dx%d, muốn 100x40", meta.OrigWidth, meta.OrigHeight)
	}
}
