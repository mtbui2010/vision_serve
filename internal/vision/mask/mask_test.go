package mask

import (
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"
)

func randPlane(r *rand.Rand, n int) []float32 {
	d := make([]float32, n)
	for i := range d {
		d[i] = float32(r.NormFloat64())
		if r.Intn(17) == 0 {
			d[i] = 0 // exact zeros separate > from >=
		}
	}
	return d
}

func TestThresholdMatchesBruteForce(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, sz := range [][2]int{{1, 1}, {3, 5}, {17, 9}, {64, 48}} {
		h, w := sz[0], sz[1]
		const ch = 3
		data := randPlane(r, ch*h*w)
		for c := 0; c < ch; c++ {
			off := c * h * w
			for _, ge := range []bool{false, true} {
				var b Bitmap
				var bbox [4]float64
				if ge {
					b, bbox = ThresholdGE(data, off, h, w, 0)
				} else {
					b, bbox = Threshold(data, off, h, w, 0)
				}
				if b.W != w || b.H != h || len(b.Data) != h*w {
					t.Fatalf("bitmap dims %dx%d len %d, want %dx%d", b.W, b.H, len(b.Data), w, h)
				}
				minX, minY, maxX, maxY := w, h, -1, -1
				for y := 0; y < h; y++ {
					for x := 0; x < w; x++ {
						v := data[off+y*w+x]
						want := v > 0 || (ge && v == 0)
						if b.Data[y*w+x] != want {
							t.Fatalf("ge=%v pixel (%d,%d)=%v: got %v", ge, x, y, v, b.Data[y*w+x])
						}
						if want {
							minX, minY, maxX, maxY = min(minX, x), min(minY, y), max(maxX, x), max(maxY, y)
						}
					}
				}
				var wantBox [4]float64
				if maxX >= 0 {
					wantBox = [4]float64{float64(minX), float64(minY), float64(maxX - minX + 1), float64(maxY - minY + 1)}
				}
				if bbox != wantBox || b.BBox() != wantBox {
					t.Fatalf("bbox %v / %v, want %v", bbox, b.BBox(), wantBox)
				}
			}
		}
	}
}

// Probability maps are compared in float64 (PaddleOCR: float64(p) > 0.3), which differs
// from a float32 compare for p == float32(0.3).
func TestThresholdComparesInFloat64(t *testing.T) {
	b, _ := Threshold([]float32{0.3, 0.29, 0.31}, 0, 1, 3, 0.3)
	if !b.Data[0] || b.Data[1] || !b.Data[2] {
		t.Fatalf("got %v, want [true false true] (float32(0.3) > 0.3 in float64)", b.Data)
	}
}

func TestExtentEmptyAndArea(t *testing.T) {
	b, e := ThresholdExtent([]float32{-1, -2, -3, -4}, 0, 2, 2, 0)
	if !e.Empty() || e.Area != 0 || e.XYWH() != [4]float64{} || b.BBox() != [4]float64{} {
		t.Fatalf("empty: %+v", e)
	}
	if e.MinX != 2 || e.MinY != 2 || e.MaxX != -1 || e.MaxY != -1 {
		t.Fatalf("empty extent = %+v, want {2 2 -1 -1 0}", e)
	}
	_, e = ThresholdExtent([]float32{0, 1, 0, 0, 2, 3}, 0, 2, 3, 0)
	if e.Area != 3 || e.MinX != 1 || e.MaxX != 2 || e.MinY != 0 || e.MaxY != 1 {
		t.Fatalf("extent = %+v", e)
	}
}

// refEncode is the pre-refactor model-local encoder (encodeRLEColumnMajor(bin, h, w)),
// kept as the oracle for the single shared implementation.
func refEncode(bin []bool, h, w int) string {
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

func TestEncodeRLEKnownVector(t *testing.T) {
	// 3 wide × 2 high, row-major:
	//   . X X
	//   . . X
	// column-major: col0 = [0,0], col1 = [1,0], col2 = [1,1] -> runs 2 bg, 1 fg, 1 bg, 2 fg.
	b := Bitmap{W: 3, H: 2, Data: []bool{false, true, true, false, false, true}}
	if got := EncodeRLE(b); got != "2 1 1 2" {
		t.Fatalf("EncodeRLE = %q, want %q", got, "2 1 1 2")
	}
	// A set first pixel starts with an explicit 0-length background run.
	if got := EncodeRLE(Bitmap{W: 2, H: 1, Data: []bool{true, false}}); got != "0 1 1" {
		t.Fatalf("EncodeRLE = %q, want %q", got, "0 1 1")
	}
	if EncodeRLE(Bitmap{}) != "" || EncodeRLE(Bitmap{Data: []bool{true}, W: 0, H: 1}) != "" {
		t.Fatal("degenerate bitmaps must encode to \"\"")
	}
}

func TestRLEMatchesReferenceAndRoundTrips(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, sz := range [][2]int{{1, 1}, {1, 7}, {7, 1}, {5, 9}, {48, 64}, {217, 333}} {
		h, w := sz[0], sz[1]
		for trial := 0; trial < 4; trial++ {
			b := New(h, w)
			p := []float64{0, 1, 0.5, 0.05}[trial]
			for i := range b.Data {
				b.Data[i] = r.Float64() < p
			}
			got := EncodeRLE(b)
			if want := refEncode(b.Data, h, w); got != want {
				t.Fatalf("%dx%d p=%v: EncodeRLE differs from the reference", w, h, p)
			}
			dec := DecodeRLE(got, h, w)
			if dec.W != w || dec.H != h {
				t.Fatalf("decoded dims %dx%d", dec.W, dec.H)
			}
			for i := range b.Data {
				if dec.Data[i] != b.Data[i] {
					t.Fatalf("%dx%d: round trip lost pixel %d", w, h, i)
				}
			}
		}
	}
}

func TestDecodeRLEDegenerate(t *testing.T) {
	if b := DecodeRLE("", 2, 3); len(b.Data) != 6 {
		t.Fatalf("empty rle: len %d, want 6 (all background)", len(b.Data))
	}
	if b := DecodeRLE("1 2", 0, 3); len(b.Data) != 0 {
		t.Fatalf("zero height: len %d", len(b.Data))
	}
	// A malformed count stops decoding; earlier runs are kept.
	b := DecodeRLE("1 2 x 3", 3, 2)
	// column-major: (x0,y0)=bg, (x0,y1),(x0,y2)=fg -> row-major indices 2 and 4.
	want := []bool{false, false, true, false, true, false}
	for i := range want {
		if b.Data[i] != want[i] {
			t.Fatalf("malformed: got %v, want %v", b.Data, want)
		}
	}
	// Counts beyond h*w are clipped.
	if b := DecodeRLE("0 100", 2, 2); !b.Data[0] || !b.Data[3] {
		t.Fatalf("overflowing run: %v", b.Data)
	}
}

// BilinearTaps / UpsampleBilinear must reproduce torch F.interpolate(mode="bilinear",
// align_corners=False). Expected values generated with torch 2.10 on the 3×4 grid below.
func TestUpsampleBilinearMatchesTorch(t *testing.T) {
	src := []float32{-3, 1, 2, -1, 0.5, -2, 4, 1, 2, 2, -5, 0}
	const sh, sw = 3, 4
	cases := []struct {
		h, w int
		want [][]float64
	}{
		{7, 9, [][]float64{
			{-3.0, -2.333333, -0.555556, 1.055556, 1.5, 1.944444, 0.833333, -0.5, -1.0},
			{-2.5, -1.988095, -0.623016, 0.666667, 1.428571, 2.190476, 1.119048, -0.214285, -0.714286},
			{-1.0, -0.952381, -0.825397, -0.5, 1.214286, 2.928571, 1.976191, 0.642857, 0.142857},
			{0.5, 0.083333, -1.027778, -1.666667, 1.0, 3.666667, 2.833333, 1.5, 1.0},
			{1.142857, 0.904762, 0.269841, -0.261905, -0.071429, 0.119047, 0.309524, 0.5, 0.571429},
			{1.785714, 1.726191, 1.567461, 1.142857, -1.142857, -3.428572, -2.214286, -0.5, 0.142857},
			{2.0, 2.0, 2.0, 1.611111, -1.5, -4.611111, -3.055556, -0.833334, 0.0},
		}},
		{2, 3, [][]float64{{-1.729167, 1.375, 0}, {1.520833, -0.875, -0.25}}}, // downsample
	}
	for _, tc := range cases {
		up := UpsampleBilinear(src, sw, sh, sw, tc.h, tc.w)
		bin := UpsampleBilinearThreshold(src, sw, sh, sw, tc.h, tc.w, 0)
		for y := 0; y < tc.h; y++ {
			for x := 0; x < tc.w; x++ {
				got, want := float64(up[y*tc.w+x]), tc.want[y][x]
				if math.Abs(got-want) > 1e-4 {
					t.Errorf("%dx%d [%d,%d] = %v, want %v", tc.h, tc.w, y, x, got, want)
				}
				if bin.Data[y*tc.w+x] != (up[y*tc.w+x] > 0) {
					t.Errorf("%dx%d [%d,%d]: fused threshold disagrees with UpsampleBilinear", tc.h, tc.w, y, x)
				}
			}
		}
	}
}

// The fused threshold equals thresholding the float map, on a windowed (strided) source.
func TestUpsampleBilinearThresholdWindow(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	const stride = 32
	src := randPlane(r, stride*32)
	for _, c := range [][4]int{{32, 20, 100, 63}, {11, 32, 40, 120}, {1, 1, 5, 5}, {32, 32, 17, 9}} {
		sh, sw, dh, dw := c[0], c[1], c[2], c[3]
		up := UpsampleBilinear(src, stride, sh, sw, dh, dw)
		bin := UpsampleBilinearThreshold(src, stride, sh, sw, dh, dw, 0)
		if bin.W != dw || bin.H != dh {
			t.Fatalf("dims %dx%d", bin.W, bin.H)
		}
		for i := range up {
			if bin.Data[i] != (up[i] > 0) {
				t.Fatalf("%v: pixel %d differs", c, i)
			}
		}
	}
}

func TestUpsampleNearest(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6} // 2 rows × 3 cols
	got := UpsampleNearest(src, 2, 3, 4, 6)
	want := []float32{
		1, 1, 2, 2, 3, 3,
		1, 1, 2, 2, 3, 3,
		4, 4, 5, 5, 6, 6,
		4, 4, 5, 5, 6, 6,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	if d := UpsampleNearest(src, 2, 3, 1, 2); d[0] != 1 || d[1] != 2 {
		t.Fatalf("downsample: %v", d)
	}
}

func TestMinMaxNormalize(t *testing.T) {
	got := MinMaxNormalize([]float32{3, -1, 2, 7}, 0)
	want := []float32{0.5, 0, 0.375, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	for _, in := range [][]float32{{2, 2, 2}, {5}} {
		for _, v := range MinMaxNormalize(in, 0) {
			if v != 0 {
				t.Fatalf("flat input %v must give zeros", in)
			}
		}
	}
	// A range below minRange is treated as flat; with minRange 0 it is normalised.
	tiny := []float32{1, 1 + 1e-6}
	if z := MinMaxNormalize(tiny, 1e-5); z[1] != 0 {
		t.Fatalf("below minRange: %v", z)
	}
	if z := MinMaxNormalize(tiny, 0); z[1] != 1 {
		t.Fatalf("minRange 0: %v", z)
	}
	if MinMaxNormalize(nil, 0) != nil {
		t.Fatal("nil input must stay nil")
	}
}

// PixelIoU (extent-restricted) must equal brute-force pixel IoU, and IoUMayExceed must
// never reject a pair whose IoU exceeds the threshold.
func TestPixelIoU(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	const h, w = 23, 31
	mk := func() (Bitmap, Extent) {
		d := make([]float32, h*w)
		x0, y0 := r.Intn(w), r.Intn(h)
		x1, y1 := x0+r.Intn(w-x0), y0+r.Intn(h-y0)
		for y := y0; y <= y1; y++ {
			for x := x0; x <= x1; x++ {
				if r.Intn(5) > 0 {
					d[y*w+x] = 1
				}
			}
		}
		return ThresholdExtent(d, 0, h, w, 0)
	}
	for i := 0; i < 500; i++ {
		a, ea := mk()
		b, eb := mk()
		inter, union := 0, 0
		for k := range a.Data {
			if a.Data[k] && b.Data[k] {
				inter++
			}
			if a.Data[k] || b.Data[k] {
				union++
			}
		}
		want := 0.0
		if union > 0 && ea.Area > 0 && eb.Area > 0 {
			want = float64(inter) / float64(union)
		}
		if got := PixelIoU(a, ea, b, eb); got != want {
			t.Fatalf("PixelIoU = %v, brute force %v", got, want)
		}
		for _, thr := range []float64{0, 0.3, 0.7} {
			if want > thr && !IoUMayExceed(ea, eb, thr) {
				t.Fatalf("IoUMayExceed rejected a pair with IoU %v > %v", want, thr)
			}
		}
	}
	if PixelIoU(New(2, 2), Extent{Area: 1}, New(2, 3), Extent{Area: 1}) != 0 {
		t.Fatal("different sizes must give 0")
	}
}

func BenchmarkThreshold1080p(b *testing.B) {
	d := randPlane(rand.New(rand.NewSource(5)), 1080*1920)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Threshold(d, 0, 1080, 1920, 0)
	}
}

func BenchmarkEncodeRLE1080p(b *testing.B) {
	bm, _ := Threshold(randPlane(rand.New(rand.NewSource(6)), 1080*1920), 0, 1080, 1920, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EncodeRLE(bm)
	}
}

func BenchmarkUpsampleBilinearThreshold(b *testing.B) {
	src := randPlane(rand.New(rand.NewSource(7)), 256*256)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		UpsampleBilinearThreshold(src, 256, 192, 256, 1080, 1440, 0)
	}
}
