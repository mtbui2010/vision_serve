package siglip

import (
	"image"
	"math/rand"
	"testing"
)

// refClampBox is clampBox as it was before it moved onto geom.Clamp, kept verbatim: the crop a
// detection gets must not move by a pixel, or every SigLIP cosine downstream moves with it.
func refClampBox(b [4]float64, bounds image.Rectangle) (image.Rectangle, bool) {
	x0 := bounds.Min.X + int(b[0]+0.5)
	y0 := bounds.Min.Y + int(b[1]+0.5)
	x1 := x0 + int(b[2]+0.5)
	y1 := y0 + int(b[3]+0.5)
	if x0 < bounds.Min.X {
		x0 = bounds.Min.X
	}
	if y0 < bounds.Min.Y {
		y0 = bounds.Min.Y
	}
	if x1 > bounds.Max.X {
		x1 = bounds.Max.X
	}
	if y1 > bounds.Max.Y {
		y1 = bounds.Max.Y
	}
	if x1-x0 < 1 || y1-y0 < 1 {
		return image.Rectangle{}, false
	}
	return image.Rect(x0, y0, x1, y1), true
}

func TestClampBoxMatchesReference(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	bounds := []image.Rectangle{image.Rect(0, 0, 640, 480), image.Rect(20, 10, 120, 70), image.Rect(0, 0, 1, 1)}
	edge := []float64{-50.5, -1, -0.6, -0.5, -0.4, 0, 0.2, 0.49, 0.5, 0.51, 1, 99.5, 100, 639.4, 639.5, 640, 641, 900}
	n := 0
	check := func(b [4]float64, bd image.Rectangle) {
		want, ok := refClampBox(b, bd)
		got, err := clampBox(b, bd)
		if ok != (err == nil) || got != want {
			t.Fatalf("box %v in %v: got %v (err %v), reference %v (ok %v)", b, bd, got, err, want, ok)
		}
		n++
	}
	for _, bd := range bounds {
		for _, x := range edge {
			for _, w := range edge {
				check([4]float64{x, x / 2, w, w / 3}, bd)
			}
		}
		for i := 0; i < 20000; i++ {
			check([4]float64{r.Float64()*900 - 200, r.Float64()*700 - 150, r.Float64()*800 - 50, r.Float64()*600 - 50}, bd)
		}
	}
	t.Logf("%d boxes identical", n)
}
