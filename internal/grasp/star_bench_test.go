package grasp

import (
	"math"
	"testing"
)

// filledStar builds a W×H bitmap holding a filled n-pointed star (outer radius ro, inner ri)
// centred at (cx, cy). A star is the worst realistic case for the antipodal search: a long,
// heavily decimated boundary whose many facing edge pairs all pass force closure.
func filledStar(W, H, cx, cy, n int, ro, ri float64) Bitmap {
	b := Bitmap{W: W, H: H, Data: make([]bool, W*H)}
	// Polygon vertices, alternating outer/inner.
	vx := make([]float64, 2*n)
	vy := make([]float64, 2*n)
	for k := 0; k < 2*n; k++ {
		r := ro
		if k%2 == 1 {
			r = ri
		}
		a := float64(k)*math.Pi/float64(n) - math.Pi/2
		vx[k] = float64(cx) + r*math.Cos(a)
		vy[k] = float64(cy) + r*math.Sin(a)
	}
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			// even-odd point-in-polygon
			in := false
			px, py := float64(x)+0.5, float64(y)+0.5
			for i, j := 0, 2*n-1; i < 2*n; j, i = i, i+1 {
				if (vy[i] > py) != (vy[j] > py) && px < (vx[j]-vx[i])*(py-vy[i])/(vy[j]-vy[i])+vx[i] {
					in = !in
				}
			}
			b.Data[y*W+x] = in
		}
	}
	return b
}

func starParams() Params {
	p := DefaultParams()
	p.Dmax = 400
	return p
}

// The uncapped search on one star mask is the B11 case: thousands of grasps for one object.
func TestStarMaskGraspCount(t *testing.T) {
	gs := FromMask(filledStar(400, 400, 200, 200, 5, 180, 70), starParams())
	t.Logf("uncapped: %d grasps for one star mask", len(gs))
	if len(gs) < 1000 {
		t.Skipf("only %d grasps; the star no longer exercises the large-output path", len(gs))
	}
}

func BenchmarkFromMaskStar(b *testing.B) {
	mask := filledStar(400, 400, 200, 200, 5, 180, 70)
	for _, k := range []int{0, 20} {
		p := starParams()
		p.MaxGrasps = k
		b.Run(map[int]string{0: "uncapped", 20: "top20"}[k], func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				FromMask(mask, p)
			}
		})
	}
}
