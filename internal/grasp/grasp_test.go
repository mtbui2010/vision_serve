package grasp

import (
	"math"
	"testing"
)

// filledRect builds a W×H bitmap with a solid axis-aligned rectangle of size
// rw×rh placed at top-left (ox,oy).
func filledRect(W, H, ox, oy, rw, rh int) Bitmap {
	b := Bitmap{W: W, H: H, Data: make([]bool, W*H)}
	for y := oy; y < oy+rh; y++ {
		for x := ox; x < ox+rw; x++ {
			if x >= 0 && y >= 0 && x < W && y < H {
				b.Data[y*W+x] = true
			}
		}
	}
	return b
}

// filledDisk builds a W×H bitmap with a solid disk of radius r centered at (cx,cy).
func filledDisk(W, H, cx, cy, r int) Bitmap {
	b := Bitmap{W: W, H: H, Data: make([]bool, W*H)}
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			dx, dy := x-cx, y-cy
			if dx*dx+dy*dy <= r*r {
				b.Data[y*W+x] = true
			}
		}
	}
	return b
}

// angleModPi returns the absolute difference between two angles modulo pi
// (parallel-jaw orientation is defined mod pi).
func angleModPi(a, b float64) float64 {
	d := math.Mod(a-b, math.Pi)
	if d < 0 {
		d += math.Pi
	}
	// fold into [0, pi/2]
	if d > math.Pi/2 {
		d = math.Pi - d
	}
	return d
}

func TestRectangleGraspClosesShortAxis(t *testing.T) {
	// 40 wide × 20 tall rectangle, centered in a 100×80 frame.
	const W, H = 100, 80
	const rw, rh = 40, 20
	ox, oy := (W-rw)/2, (H-rh)/2
	mask := filledRect(W, H, ox, oy, rw, rh)

	p := DefaultParams()
	// allow both axes (20 and 40) so the search must PREFER the short one.
	p.Dmin = 8
	p.Dmax = 120

	gs := FromMask(mask, p)
	if len(gs) == 0 {
		t.Fatal("expected at least one grasp on a rectangle, got none")
	}

	best := gs[0]
	t.Logf("best grasp: x=%.1f y=%.1f theta=%.3f width=%.1f q=%.3f",
		best.X, best.Y, best.Theta, best.Width, best.Quality)

	// width should be near the SHORT axis (~20 px), decimation tolerance ±8.
	if math.Abs(best.Width-float64(rh)) > 8 {
		t.Errorf("best grasp width = %.1f, expected near short axis %d (±8)", best.Width, rh)
	}

	// theta should close top-to-bottom: finger vector ~vertical => angle ~pi/2 (mod pi).
	if d := angleModPi(best.Theta, math.Pi/2); d > 0.35 {
		t.Errorf("best grasp theta = %.3f, expected ~pi/2 mod pi (off by %.3f rad)", best.Theta, d)
	}

	// center near the rectangle centroid.
	wantCx, wantCy := float64(ox)+float64(rw)/2, float64(oy)+float64(rh)/2
	if math.Hypot(best.X-wantCx, best.Y-wantCy) > 12 {
		t.Errorf("best grasp center (%.1f,%.1f) far from centroid (%.1f,%.1f)",
			best.X, best.Y, wantCx, wantCy)
	}

	// quality must be in [0,1].
	for i, g := range gs {
		if g.Quality < 0 || g.Quality > 1 {
			t.Errorf("grasp %d quality %.3f out of [0,1]", i, g.Quality)
		}
	}
}

func TestDiskGraspSanity(t *testing.T) {
	const W, H = 80, 80
	mask := filledDisk(W, H, 40, 40, 18)

	p := DefaultParams()
	p.Dmin = 8
	p.Dmax = 80

	gs := FromMask(mask, p)
	if len(gs) == 0 {
		t.Fatal("expected at least one valid grasp on a disk, got none")
	}
	for i, g := range gs {
		if g.Width <= p.Dmin || g.Width >= p.Dmax {
			t.Errorf("grasp %d width %.1f outside (Dmin,Dmax)=(%.1f,%.1f)", i, g.Width, p.Dmin, p.Dmax)
		}
		if g.Quality < 0 || g.Quality > 1 {
			t.Errorf("grasp %d quality %.3f out of [0,1]", i, g.Quality)
		}
		// center should be near the disk center (40,40), generous tolerance.
		if math.Hypot(g.X-40, g.Y-40) > 30 {
			t.Errorf("grasp %d center (%.1f,%.1f) implausibly far from disk center", i, g.X, g.Y)
		}
	}
}

func TestSquareReturnsGrasp(t *testing.T) {
	const W, H = 60, 60
	mask := filledRect(W, H, 15, 15, 30, 30)
	p := DefaultParams()
	p.Dmin = 8
	p.Dmax = 80
	gs := FromMask(mask, p)
	if len(gs) == 0 {
		t.Fatal("expected at least one grasp on a square, got none")
	}
	for i, g := range gs {
		if g.Quality < 0 || g.Quality > 1 {
			t.Errorf("grasp %d quality %.3f out of [0,1]", i, g.Quality)
		}
		if g.Width <= p.Dmin || g.Width >= p.Dmax {
			t.Errorf("grasp %d width %.1f outside bounds", i, g.Width)
		}
	}
}

func TestMaxGraspsCap(t *testing.T) {
	const W, H = 100, 80
	mask := filledRect(W, H, 30, 30, 40, 20)
	p := DefaultParams()
	p.Dmin = 8
	p.Dmax = 120
	p.MaxGrasps = 3
	gs := FromMask(mask, p)
	if len(gs) > 3 {
		t.Errorf("MaxGrasps=3 not honored, got %d grasps", len(gs))
	}
	// results must be sorted by quality descending.
	for i := 1; i < len(gs); i++ {
		if gs[i-1].Quality < gs[i].Quality {
			t.Errorf("grasps not sorted by quality desc at %d: %.3f < %.3f", i, gs[i-1].Quality, gs[i].Quality)
		}
	}
}

func TestEmptyMaskNoGrasps(t *testing.T) {
	mask := Bitmap{W: 50, H: 50, Data: make([]bool, 50*50)} // all false
	gs := FromMask(mask, DefaultParams())
	if len(gs) != 0 {
		t.Errorf("empty mask should yield 0 grasps, got %d", len(gs))
	}
}

func TestDegenerateMasks(t *testing.T) {
	// zero-size and malformed bitmaps must not panic and must return no grasps.
	cases := []Bitmap{
		{W: 0, H: 0, Data: nil},
		{W: 10, H: 10, Data: make([]bool, 5)}, // wrong length
		{W: -1, H: 5, Data: nil},
	}
	for i, m := range cases {
		gs := FromMask(m, DefaultParams())
		if len(gs) != 0 {
			t.Errorf("case %d: expected 0 grasps, got %d", i, len(gs))
		}
	}
}

func TestDeterministic(t *testing.T) {
	const W, H = 100, 80
	mask := filledRect(W, H, 30, 30, 40, 20)
	p := DefaultParams()
	p.Dmin = 8
	p.Dmax = 120
	a := FromMask(mask, p)
	b := FromMask(mask, p)
	if len(a) != len(b) {
		t.Fatalf("non-deterministic count: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("non-deterministic grasp at %d: %+v vs %+v", i, a[i], b[i])
		}
	}
}
