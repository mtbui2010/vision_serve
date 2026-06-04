package mobilesam

import (
	"image"
	"image/color"
	"math"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// fillRGBA builds an NRGBA test image of size wxh, filled with a single color.
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

// Encoder input: HWC float32 [newH,newW,3], values 0..255 (no normalization in Go),
// long side resized to 1024, scale = 1024/max(W,H).
func TestEncoderInputShapeAndScale(t *testing.T) {
	// 640x480 -> scale = 1024/640 = 1.6 ; newW=1024, newH=768.
	img := fillRGBA(640, 480, 100, 150, 200)
	ten, scale := encoderInput(img)

	if math.Abs(scale-1.6) > 1e-9 {
		t.Fatalf("scale = %v, want 1.6", scale)
	}
	want := []int64{768, 1024, 3}
	if len(ten.Shape) != 3 || ten.Shape[0] != want[0] || ten.Shape[1] != want[1] || ten.Shape[2] != want[2] {
		t.Fatalf("shape = %v, want %v (HWC)", ten.Shape, want)
	}
	if int64(len(ten.Data)) != ten.NumElements() {
		t.Fatalf("len(Data)=%d != NumElements=%d", len(ten.Data), ten.NumElements())
	}
	// First pixel must be raw 0..255 RGB (no normalization).
	if ten.Data[0] != 100 || ten.Data[1] != 150 || ten.Data[2] != 200 {
		t.Fatalf("first pixel = (%v,%v,%v), want raw (100,150,200)", ten.Data[0], ten.Data[1], ten.Data[2])
	}
}

// A tall image keeps the long side = 1024 on the height axis.
func TestEncoderInputTall(t *testing.T) {
	// 256x512 -> scale = 1024/512 = 2.0 ; newW=512, newH=1024.
	img := fillRGBA(256, 512, 0, 0, 0)
	ten, scale := encoderInput(img)
	if math.Abs(scale-2.0) > 1e-9 {
		t.Fatalf("scale = %v, want 2.0", scale)
	}
	if ten.Shape[0] != 1024 || ten.Shape[1] != 512 {
		t.Fatalf("HxW = %dx%d, want 1024x512", ten.Shape[0], ten.Shape[1])
	}
}

// A box prompt becomes 2 points (top-left label 2, bottom-right label 3); coords are
// scaled into 1024 space.
func TestPromptBoxToPoints(t *testing.T) {
	p := models.Prompt{Boxes: [][4]float64{{10, 20, 100, 80}}} // x,y,w,h
	sets, err := promptToPointSets(p)
	if err != nil {
		t.Fatalf("promptToPointSets error: %v", err)
	}
	if len(sets) != 1 {
		t.Fatalf("got %d sets, want 1", len(sets))
	}
	ps := sets[0]
	if ps.n() != 2 {
		t.Fatalf("n=%d, want 2", ps.n())
	}
	if ps.labels[0] != 2 || ps.labels[1] != 3 {
		t.Fatalf("labels=%v, want [2 3]", ps.labels)
	}
	// box corners in original coords: (10,20) and (110,100).
	want := []float64{10, 20, 110, 100}
	for i, v := range want {
		if ps.coords[i] != v {
			t.Fatalf("coords=%v, want %v", ps.coords, want)
		}
	}
	// scaled by 0.5 → halved.
	sc := ps.scaledCoords(0.5)
	if sc[0] != 5 || sc[1] != 10 || sc[2] != 55 || sc[3] != 50 {
		t.Fatalf("scaledCoords=%v, want [5 10 55 50]", sc)
	}
}

func TestPromptEmptyReturnsNil(t *testing.T) {
	// Empty prompt → nil slice, nil error (triggers Automatic Mask Generator path).
	sets, err := promptToPointSets(models.Prompt{})
	if err != nil {
		t.Fatalf("expected no error for empty prompt, got: %v", err)
	}
	if sets != nil {
		t.Fatalf("expected nil sets for empty prompt, got: %v", sets)
	}
}

func TestPromptTextOnlyErrors(t *testing.T) {
	// Text-only prompt → error (must use grounded-sam for text).
	if _, err := promptToPointSets(models.Prompt{Text: "cat"}); err == nil {
		t.Fatalf("expected error for text-only prompt on mobile-sam")
	}
}

// RLE is column-major, starts with a background run, and its counts sum to H*W.
func TestEncodeRLEColumnMajor(t *testing.T) {
	// 2x2 mask, column-major order. Set pixel (x=0,y=1) and (x=1,y=1) foreground.
	// bin index = y*w+x. w=h=2.
	h, w := 2, 2
	bin := make([]bool, 4)
	bin[1*w+0] = true // (0,1)
	bin[1*w+1] = true // (1,1)
	// Column-major read: col0 -> (0,0)F=0,(0,1)T ; col1 -> (1,0)F,(1,1)T
	// sequence: 0,1,0,1 -> runs starting background: [1,1,1,1]
	rle := encodeRLEColumnMajor(bin, h, w)
	if rle != "1 1 1 1" {
		t.Fatalf("rle=%q, want \"1 1 1 1\"", rle)
	}
	// counts sum to H*W.
	sum := 0
	for _, f := range []int{1, 1, 1, 1} {
		sum += f
	}
	if sum != h*w {
		t.Fatalf("rle counts sum=%d, want %d", sum, h*w)
	}
}

// pickMaskAndIoU distinguishes masks (4-D, large) from low_res_masks (4-D, 256) and iou (2-D).
func TestPickMaskAndIoU(t *testing.T) {
	outs := []engine.Tensor{
		engine.F32(make([]float32, 1*1*4*4), 1, 1, 4, 4),         // masks (large)
		engine.F32(make([]float32, 1*1*256*256), 1, 1, 256, 256), // low_res_masks
		engine.F32([]float32{0.9}, 1, 1),                         // iou
	}
	names := []string{"masks", "low_res_masks", "iou_predictions"}
	mask, iou := pickMaskAndIoU(names, outs)
	if mask == nil || mask.Dim(2) != 4 {
		t.Fatalf("mask pick wrong: %+v", mask)
	}
	if iou == nil || len(iou.Shape) != 2 {
		t.Fatalf("iou pick wrong: %+v", iou)
	}
}
