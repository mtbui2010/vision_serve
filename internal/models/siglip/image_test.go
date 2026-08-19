package siglip

import (
	"image"
	"image/color"
	"math"
	"strings"
	"testing"

	"visionserve/internal/engine"
)

// solid builds a WxH image whose left half is one colour and right half another, so a crop's
// content is checkable from the tensor alone.
func solid(w, h int, left, right color.RGBA) image.Image {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := left
			if x >= w/2 {
				c = right
			}
			im.Set(x, y, c)
		}
	}
	return im
}

func TestCropTensorShapeAndOrder(t *testing.T) {
	black, white := color.RGBA{0, 0, 0, 255}, color.RGBA{255, 255, 255, 255}
	img := solid(100, 60, black, white)

	// box 0 is entirely in the black half, box 1 entirely in the white half.
	boxes := [][4]float64{{0, 0, 40, 60}, {60, 0, 40, 60}}
	tn, err := CropTensor(img, boxes)
	if err != nil {
		t.Fatalf("CropTensor: %v", err)
	}
	want := []int64{2, 3, ImageSize, ImageSize}
	if len(tn.Shape) != 4 || tn.Shape[0] != want[0] || tn.Shape[1] != want[1] ||
		tn.Shape[2] != want[2] || tn.Shape[3] != want[3] {
		t.Fatalf("shape %v, want %v", tn.Shape, want)
	}
	plane := ImageSize * ImageSize
	if len(tn.Data) != 2*3*plane {
		t.Fatalf("data length %d, want %d", len(tn.Data), 2*3*plane)
	}

	// Normalisation is (v/255 - 0.5)/0.5, so black is -1 and white is +1. Checking the CENTRE
	// pixel of each crop pins the batch order: if the two crops were swapped or interleaved,
	// these signs flip.
	centre := (ImageSize/2)*ImageSize + ImageSize/2
	if got := tn.Data[centre]; math.Abs(float64(got+1)) > 1e-5 {
		t.Errorf("crop 0 centre = %v, want -1 (black)", got)
	}
	if got := tn.Data[3*plane+centre]; math.Abs(float64(got-1)) > 1e-5 {
		t.Errorf("crop 1 centre = %v, want +1 (white) — batch order or stride is wrong", got)
	}
}

func TestCropTensorClamps(t *testing.T) {
	img := solid(100, 60, color.RGBA{10, 10, 10, 255}, color.RGBA{200, 200, 200, 255})
	// A box hanging off every edge must still produce a valid crop rather than panicking.
	if _, err := CropTensor(img, [][4]float64{{-20, -10, 200, 200}}); err != nil {
		t.Fatalf("an over-large box should clamp, got: %v", err)
	}
}

// A box outside the image is a caller bug — the coordinates are in the wrong space. Embedding a
// black rectangle instead would produce a plausible vector that gets scored like any other, so
// this must be an error, loudly.
func TestCropTensorRejectsBoxOutsideImage(t *testing.T) {
	img := solid(100, 60, color.RGBA{}, color.RGBA{})
	_, err := CropTensor(img, [][4]float64{{500, 500, 10, 10}})
	if err == nil {
		t.Fatal("a box entirely outside the image must be an error, not a black crop")
	}
	if !strings.Contains(err.Error(), "original-image space") {
		t.Errorf("error should say what is probably wrong, got: %v", err)
	}
}

func TestCropTensorRejectsEmptyBatch(t *testing.T) {
	img := solid(10, 10, color.RGBA{}, color.RGBA{})
	if _, err := CropTensor(img, nil); err == nil {
		t.Fatal("an empty box list must be an error, not an empty tensor")
	}
}

func TestDecodeImageEmbeddingsNormalises(t *testing.T) {
	// Two rows, deliberately un-normalised, as the tower emits them.
	out := engine.F32([]float32{3, 4, 0, 0, 0, 5}, 2, 3)
	embs, err := DecodeImageEmbeddings([]engine.Tensor{out}, 2)
	if err != nil {
		t.Fatalf("DecodeImageEmbeddings: %v", err)
	}
	for i, e := range embs {
		var n float64
		for _, v := range e {
			n += float64(v) * float64(v)
		}
		if math.Abs(math.Sqrt(n)-1) > 1e-6 {
			t.Errorf("row %d has norm %v, want 1 — the tower does not normalise, so we must",
				i, math.Sqrt(n))
		}
	}
}

func TestDecodeImageEmbeddingsBatchMismatch(t *testing.T) {
	out := engine.F32([]float32{1, 0, 0, 1}, 2, 2)
	if _, err := DecodeImageEmbeddings([]engine.Tensor{out}, 3); err == nil {
		t.Fatal("a batch size that disagrees with the crop count must be an error: it means " +
			"crops and boxes are no longer index-aligned")
	}
}

func TestScoreCrops(t *testing.T) {
	crops := [][]float32{{1, 0}, {0, 1}}
	text := [][]float32{{1, 0}, {0, 1}, {0.6, 0.8}}
	s, err := ScoreCrops(crops, text)
	if err != nil {
		t.Fatalf("ScoreCrops: %v", err)
	}
	if len(s) != 6 {
		t.Fatalf("got %d scores, want 6 (2 crops x 3 names)", len(s))
	}
	// Row-major [crop][name]: crop 0 matches name 0, crop 1 matches name 1.
	if math.Abs(float64(s[0]-1)) > 1e-6 || math.Abs(float64(s[1])) > 1e-6 {
		t.Errorf("crop 0 row = %v, want [1 0 0.6]", s[:3])
	}
	if math.Abs(float64(s[4]-1)) > 1e-6 || math.Abs(float64(s[5]-0.8)) > 1e-6 {
		t.Errorf("crop 1 row = %v, want [0 1 0.8]", s[3:])
	}
}

// The two towers must be the same checkpoint. If they are not, the widths differ and every
// score is meaningless — worth an error naming the actual cause.
func TestScoreCropsRejectsMismatchedWidths(t *testing.T) {
	_, err := ScoreCrops([][]float32{{1, 0}}, [][]float32{{1, 0, 0}})
	if err == nil {
		t.Fatal("mismatched embedding widths must be an error")
	}
	if !strings.Contains(err.Error(), "different checkpoints") {
		t.Errorf("error should name the likely cause, got: %v", err)
	}
}
