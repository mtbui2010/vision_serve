package api

import "testing"

func TestMaskRLERoundTrip(t *testing.T) {
	w, h := 7, 5
	bin := make([]bool, w*h)
	// a filled rectangle [2..4]x[1..3]
	for y := 1; y <= 3; y++ {
		for x := 2; x <= 4; x++ {
			bin[y*w+x] = true
		}
	}
	rle := EncodeMaskRLE(bin, w, h)
	got := DecodeMaskRLE(rle, w, h)
	if len(got) != len(bin) {
		t.Fatalf("len = %d, want %d", len(got), len(bin))
	}
	for i := range bin {
		if got[i] != bin[i] {
			t.Fatalf("pixel %d (x=%d,y=%d): got %v want %v (rle=%q)", i, i%w, i/w, got[i], bin[i], rle)
		}
	}
}

func TestMaskRLEEmptyAndFull(t *testing.T) {
	w, h := 4, 3
	if DecodeMaskRLE("", w, h) == nil {
		t.Fatal("decode empty returned nil")
	}
	full := make([]bool, w*h)
	for i := range full {
		full[i] = true
	}
	if got := DecodeMaskRLE(EncodeMaskRLE(full, w, h), w, h); len(got) != w*h {
		t.Fatalf("full round-trip len = %d", len(got))
	} else {
		for i := range got {
			if !got[i] {
				t.Fatalf("full round-trip lost pixel %d", i)
			}
		}
	}
}
