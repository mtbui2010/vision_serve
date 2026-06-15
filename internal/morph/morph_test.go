package morph

import "testing"

func area(b []bool) int {
	n := 0
	for _, v := range b {
		if v {
			n++
		}
	}
	return n
}

func TestDilateErodeSquare(t *testing.T) {
	w, h := 11, 11
	data := make([]bool, w*h)
	data[5*w+5] = true // single centre pixel

	// dilate by 2 → a 5×5 square (Chebyshev radius 2) = 25 pixels
	d := Dilate(data, w, h, 2)
	if got := area(d); got != 25 {
		t.Fatalf("dilate(1px, r=2) area = %d, want 25 (5x5 square)", got)
	}
	// corners of the 5x5 block set
	for _, p := range []int{(3*w + 3), (3*w + 7), (7*w + 3), (7*w + 7)} {
		if !d[p] {
			t.Errorf("dilated square missing corner at %d", p)
		}
	}

	// erode the dilated 5×5 by 1 → 3×3 = 9
	e := Dilate(d, w, h, -1)
	if got := area(e); got != 9 {
		t.Fatalf("erode(5x5, r=1) area = %d, want 9 (3x3)", got)
	}

	// erode by 0 = no-op; large erode wipes a small blob
	if area(Dilate(d, w, h, 0)) != 25 {
		t.Error("radius 0 should be a no-op")
	}
	if area(Dilate(d, w, h, -5)) != 0 {
		t.Error("eroding a 5x5 by 5 should clear it")
	}
}

func TestDilateBorderErosionKeepsEdge(t *testing.T) {
	// A blob filling the whole image must NOT erode at the image border (out-of-image is
	// "don't care" for erosion), so a full mask eroded by 1 stays full.
	w, h := 6, 6
	full := make([]bool, w*h)
	for i := range full {
		full[i] = true
	}
	if got := area(Dilate(full, w, h, -1)); got != w*h {
		t.Fatalf("eroding a full mask = %d, want %d (border not eroded)", got, w*h)
	}
}
