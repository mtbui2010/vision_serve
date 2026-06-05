package maskgrasp

import (
	"strconv"
	"strings"
	"testing"

	"visionserve/internal/grasp"
	"visionserve/internal/models"
)

// TestRegistered confirms the package init() registered the "mask-grasp" factory
// (so the central blank-import in cmd/visionserve/main.go actually wires it in).
func TestRegistered(t *testing.T) {
	if !models.IsRegistered("mask-grasp") {
		t.Fatalf("mask-grasp not registered; registered = %v", models.Registered())
	}
}

// encodeRLEColumnMajor mirrors the SAM models' encoder (mobilesam/postprocess.go):
// column-major (x outer, y inner) traversal, runs starting with background.
// Kept here so the test asserts decode is the exact inverse of the real encoder.
func encodeRLEColumnMajor(bin []bool, h, w int) string {
	if len(bin) == 0 {
		return ""
	}
	var counts []int
	prev := false
	run := 0
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			v := bin[y*w+x]
			if v == prev {
				run++
			} else {
				counts = append(counts, run)
				prev = v
				run = 1
			}
		}
	}
	counts = append(counts, run)
	parts := make([]string, len(counts))
	for i, c := range counts {
		parts[i] = strconv.Itoa(c)
	}
	return strings.Join(parts, " ")
}

// TestDecodeRoundTrip checks decodeRLEColumnMajor is the exact inverse of the
// SAM encoder for several masks, including a filled rectangle and edge pixels.
func TestDecodeRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		w, h int
		set  func(x, y int) bool
	}{
		{"empty", 8, 6, func(x, y int) bool { return false }},
		{"full", 8, 6, func(x, y int) bool { return true }},
		{"rect", 12, 10, func(x, y int) bool { return x >= 3 && x <= 8 && y >= 2 && y <= 6 }},
		{"corners", 5, 5, func(x, y int) bool { return (x == 0 && y == 0) || (x == 4 && y == 4) }},
		{"single_col", 7, 7, func(x, y int) bool { return x == 3 }},
		{"single_row", 7, 7, func(x, y int) bool { return y == 3 }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			orig := make([]bool, c.w*c.h)
			for y := 0; y < c.h; y++ {
				for x := 0; x < c.w; x++ {
					orig[y*c.w+x] = c.set(x, y)
				}
			}
			rle := encodeRLEColumnMajor(orig, c.h, c.w)
			bm, err := decodeRLEColumnMajor(rle, c.w, c.h)
			if err != nil {
				t.Fatalf("decode error: %v", err)
			}
			if bm.W != c.w || bm.H != c.h || len(bm.Data) != c.w*c.h {
				t.Fatalf("dims mismatch: got %dx%d len=%d", bm.W, bm.H, len(bm.Data))
			}
			for i := range orig {
				if bm.Data[i] != orig[i] {
					t.Fatalf("pixel %d (x=%d y=%d): got %v want %v", i, i%c.w, i/c.w, bm.Data[i], orig[i])
				}
			}
		})
	}
}

// TestDecodeEmptyString returns an all-false mask of the right size, no error.
func TestDecodeEmptyString(t *testing.T) {
	bm, err := decodeRLEColumnMajor("", 4, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bm.W != 4 || bm.H != 3 || len(bm.Data) != 12 {
		t.Fatalf("bad dims: %dx%d len=%d", bm.W, bm.H, len(bm.Data))
	}
	for i, v := range bm.Data {
		if v {
			t.Fatalf("pixel %d should be false", i)
		}
	}
}

// TestDecodeBadInput surfaces a clear error on malformed run lengths.
func TestDecodeBadInput(t *testing.T) {
	if _, err := decodeRLEColumnMajor("3 x 4", 4, 4); err == nil {
		t.Fatal("expected error on non-numeric run length")
	}
	if _, err := decodeRLEColumnMajor("3 -2 4", 4, 4); err == nil {
		t.Fatal("expected error on negative run length")
	}
}

// TestDecodedMaskYieldsGrasp wires the decode into the grasp core: a rectangle
// mask round-tripped through RLE still produces a valid grasp (closing the short
// axis), proving the end-to-end mask→RLE→Bitmap→grasp path is consistent.
func TestDecodedMaskYieldsGrasp(t *testing.T) {
	const w, h = 100, 60
	orig := make([]bool, w*h)
	// 40-wide x 20-tall filled rectangle centered in the image.
	for y := 20; y < 40; y++ {
		for x := 30; x < 70; x++ {
			orig[y*w+x] = true
		}
	}
	rle := encodeRLEColumnMajor(orig, h, w)
	bm, err := decodeRLEColumnMajor(rle, w, h)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}
	gs := grasp.FromMask(bm, grasp.DefaultParams())
	if len(gs) == 0 {
		t.Fatal("expected at least one grasp from the rectangle mask")
	}
	best := gs[0]
	if best.Quality < 0 || best.Quality > 1 {
		t.Fatalf("quality out of range: %v", best.Quality)
	}
	// Center should be near the rectangle centroid (~50, ~30).
	if best.X < 40 || best.X > 60 || best.Y < 20 || best.Y > 40 {
		t.Fatalf("grasp center %v,%v not near rectangle centroid", best.X, best.Y)
	}
}
