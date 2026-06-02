package imageproc

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestImageToCHWFloatShapeAndNormalize(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{255, 0, 0, 255}) // R=1.0, G=0, B=0 sau /255

	mean := []float32{0.5, 0.5, 0.5}
	std := []float32{0.5, 0.5, 0.5}
	tt := ImageToCHWFloat(img, mean, std)

	want := []int64{1, 3, 1, 1}
	if len(tt.Shape) != 4 {
		t.Fatalf("shape len = %d", len(tt.Shape))
	}
	for i := range want {
		if tt.Shape[i] != want[i] {
			t.Fatalf("shape = %v, muốn %v", tt.Shape, want)
		}
	}
	// R: (1.0-0.5)/0.5 = 1.0 ; G,B: (0-0.5)/0.5 = -1.0
	if math.Abs(float64(tt.Data[0])-1.0) > 1e-6 {
		t.Fatalf("R plane = %v, muốn 1.0", tt.Data[0])
	}
	if math.Abs(float64(tt.Data[1])+1.0) > 1e-6 || math.Abs(float64(tt.Data[2])+1.0) > 1e-6 {
		t.Fatalf("G/B plane = %v,%v muốn -1.0", tt.Data[1], tt.Data[2])
	}
}
