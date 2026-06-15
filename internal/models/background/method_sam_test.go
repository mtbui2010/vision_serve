package background

import "testing"

// TestSAMSeedPointsInLowerFrame checks samSeedPoints produces in-bounds foreground points
// concentrated in the lower frame across a range of aspect ratios.
func TestSAMSeedPointsInLowerFrame(t *testing.T) {
	cases := [][2]int{{640, 480}, {1920, 1080}, {100, 100}, {320, 800}}
	for _, c := range cases {
		w, h := c[0], c[1]
		pts := samSeedPoints(w, h)
		if len(pts) == 0 {
			t.Fatalf("w=%d h=%d: expected seed points, got none", w, h)
		}
		fw, fh := float64(w), float64(h)
		for i, p := range pts {
			if p.X < 0 || p.Y < 0 || p.X > fw-1+1e-9 || p.Y > fh-1+1e-9 {
				t.Errorf("w=%d h=%d point %d out of bounds: (%v,%v)", w, h, i, p.X, p.Y)
			}
			if p.Label != 1 {
				t.Errorf("w=%d h=%d point %d: label=%d, want 1 (foreground)", w, h, i, p.Label)
			}
			if p.Y/fh < 0.5-1e-9 {
				t.Errorf("w=%d h=%d point %d: y-frac=%.3f not in lower frame", w, h, i, p.Y/fh)
			}
		}
	}
}
