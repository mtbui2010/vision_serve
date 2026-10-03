package mask

import (
	"math/rand"
	"testing"
)

// ResizeNearest must sample exactly the pixels UpsampleNearest samples, so a bool mask and the
// same mask as a float map resize identically (background's depth/cv methods used a private copy
// of this loop).
func TestResizeNearestMatchesUpsampleNearest(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for _, sz := range [][4]int{{256, 256, 480, 640}, {37, 53, 100, 41}, {1, 1, 9, 4}, {64, 48, 32, 24}, {5, 7, 5, 7}} {
		sh, sw, dh, dw := sz[0], sz[1], sz[2], sz[3]
		b := New(sh, sw)
		f := make([]float32, sh*sw)
		for i := range b.Data {
			if r.Intn(3) == 0 {
				b.Data[i], f[i] = true, 1
			}
		}
		got := ResizeNearest(b, dh, dw)
		want := UpsampleNearest(f, sh, sw, dh, dw)
		if got.W != dw || got.H != dh || len(got.Data) != dw*dh {
			t.Fatalf("%v: got %dx%d (%d px)", sz, got.W, got.H, len(got.Data))
		}
		for i, v := range got.Data {
			if v != (want[i] == 1) {
				t.Fatalf("%v: pixel %d = %v, float resize says %v", sz, i, v, want[i])
			}
		}
	}
}

func TestResizeNearestSameSizeShares(t *testing.T) {
	b := New(3, 4)
	got := ResizeNearest(b, 3, 4)
	if &got.Data[0] != &b.Data[0] {
		t.Error("same-size resize copied the data; callers rely on it being free")
	}
}

func TestOr(t *testing.T) {
	dst := []bool{false, true, false, false}
	Or(dst, []bool{true, false, false, true})
	want := []bool{true, true, false, true}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("Or = %v, want %v", dst, want)
		}
	}
}
