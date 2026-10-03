package paddleocr

import (
	"math"
	"testing"

	"visionserve/internal/engine"
)

// Real PP-OCRv4 shapes (inspected from models/paddle-ocr/*.onnx with onnxruntime):
//
//	det  x [1,3,64,320]  -> sigmoid_0.tmp_0 [1,1,64,320]
//	rec  x [1,3,48,320]  -> softmax_11.tmp_0 [1,40,6625]
//	rec  x [1,3,48,100]  -> softmax_11.tmp_0 [1,12,6625]
//
// ppocr_keys_v1.txt holds 6623 keys, so 6625 = blank + 6623 keys + " " (use_space_char).
const (
	realRecClasses = 6625
	realRecT       = 40
	realKeys       = 6623
)

// probRect builds a det output tensor [1,1,h,w] with prob 0.9 inside the pixel rectangle
// [x0..x1] × [y0..y1] (inclusive) and 0.05 elsewhere.
func probRect(h, w int, rects ...[4]int) engine.Tensor {
	data := make([]float32, h*w)
	for i := range data {
		data[i] = 0.05
	}
	for _, r := range rects {
		for y := r[1]; y <= r[3]; y++ {
			for x := r[0]; x <= r[2]; x++ {
				data[y*w+x] = 0.9
			}
		}
	}
	return engine.F32(data, 1, 1, int64(h), int64(w))
}

func approx(t *testing.T, name string, got, want, eps float64) {
	t.Helper()
	if math.Abs(got-want) > eps {
		t.Errorf("%s: got %.4f, want %.4f", name, got, want)
	}
}

// B8: the DB unclip pushes every side out by d = area*ratio/perimeter (PaddleOCR
// DBPostProcess.unclip via pyclipper), measured on pixel centres like cv2.minAreaRect.
// The old code scaled the box ×1.5 about its centre: a 200×16 shrunk kernel became
// 300×24 instead of ~221×36 (the reference), i.e. far too wide and too short.
func TestExtractBBoxes_UnclipExpandsEachSideByDBDistance(t *testing.T) {
	const h, w = 64, 320
	pm := probRect(h, w, [4]int{40, 20, 239, 35}) // centres span 199 × 15
	boxes, err := extractBBoxes(pm.Data, int(pm.Shape[2]), int(pm.Shape[3]), defaultDetThresh, defaultUnclipRatio)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("want 1 box, got %d: %v", len(boxes), boxes)
	}
	d := 199.0 * 15.0 * 1.5 / (2 * (199.0 + 15.0)) // ≈ 10.46
	b := boxes[0]
	approx(t, "x", b[0], 40-d, 1e-9)
	approx(t, "y", b[1], 20-d, 1e-9)
	approx(t, "w", b[2], 199+2*d, 1e-9)
	approx(t, "h", b[3], 15+2*d, 1e-9)
	// Explicit regression guard against the centre-scale formula (h would be 24).
	if b[3] < 30 {
		t.Errorf("box height %.2f: unclip must grow a thin line vertically by 2d (~21px)", b[3])
	}
}

func TestExtractBBoxes_UnclipClampsToMap(t *testing.T) {
	const h, w = 64, 320
	pm := probRect(h, w, [4]int{0, 0, 99, 9})
	boxes, err := extractBBoxes(pm.Data, h, w, defaultDetThresh, defaultUnclipRatio)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("want 1 box, got %d", len(boxes))
	}
	b := boxes[0]
	if b[0] != 0 || b[1] != 0 {
		t.Errorf("box must be clamped at the origin, got %v", b)
	}
	d := unclipDistance(99, 9, defaultUnclipRatio)
	approx(t, "w", b[2], 99+d, 1e-9)
	approx(t, "h", b[3], 9+d, 1e-9)
}

// B10: BFS must visit each pixel once. A map that is entirely text (worst case: one
// component covering 960×960 = 921 600 px) must yield exactly one full-map box, and
// separate blobs must stay separate.
func TestExtractBBoxes_LargeBlobAndSeparateComponents(t *testing.T) {
	const h, w = 960, 960
	pm := probRect(h, w, [4]int{0, 0, w - 1, h - 1})
	boxes, err := extractBBoxes(pm.Data, h, w, defaultDetThresh, defaultUnclipRatio)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 || boxes[0] != [4]float64{0, 0, w, h} {
		t.Fatalf("full map: want one box [0 0 %d %d], got %v", w, h, boxes)
	}

	pm = probRect(64, 320, [4]int{10, 10, 60, 20}, [4]int{100, 10, 200, 20}, [4]int{5, 50, 6, 51}) // last one is noise (<16 px)
	boxes, err = extractBBoxes(pm.Data, 64, 320, defaultDetThresh, defaultUnclipRatio)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 2 {
		t.Fatalf("want 2 components (noise dropped), got %d: %v", len(boxes), boxes)
	}
}

func BenchmarkExtractBBoxesFullMap960(b *testing.B) {
	pm := probRect(960, 960, [4]int{0, 0, 959, 959})
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := extractBBoxes(pm.Data, 960, 960, defaultDetThresh, defaultUnclipRatio); err != nil {
			b.Fatal(err)
		}
	}
}

// Guard: a prob map whose length disagrees with h*w used to index out of range (panic).
func TestExtractBBoxes_LengthMismatchIsError(t *testing.T) {
	pm := probRect(64, 320)
	cases := []struct {
		name string
		data []float32
		h, w int
	}{
		{"longer than h*w", pm.Data, 32, 320},
		{"shorter than h*w", pm.Data[:100], 64, 320},
		{"zero size", pm.Data, 0, 320},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := extractBBoxes(c.data, c.h, c.w, defaultDetThresh, defaultUnclipRatio); err == nil {
				t.Fatal("want error, got nil")
			}
		})
	}
}

// recLogits builds a rec output [1,T,C] whose argmax per step follows seq (class indices).
func recLogits(seq []int, T, C int) engine.Tensor {
	data := make([]float32, T*C)
	for t := 0; t < T; t++ {
		cls := 0
		if t < len(seq) {
			cls = seq[t]
		}
		data[t*C+cls] = 0.9
	}
	return engine.F32(data, 1, int64(T), int64(C))
}

// B9: with the real dictionary (6623 keys) the rec head has 6625 classes; the last one is
// the space PaddleOCR appends (use_space_char=True). It used to be dropped.
func TestCTCDecode_SpaceCharRealCharset(t *testing.T) {
	charset, err := loadCharset("../../../models/paddle-ocr")
	if err != nil {
		t.Fatal(err)
	}
	if len(charset) != realKeys {
		t.Fatalf("ppocr_keys_v1.txt: want %d keys, got %d", realKeys, len(charset))
	}
	idx := func(s string) int {
		for i, c := range charset {
			if c == s {
				return i + 1 // +1 for blank at 0
			}
		}
		t.Fatalf("char %q not in charset", s)
		return 0
	}
	space := realRecClasses - 1
	// "Hi  ab" with CTC structure: duplicates collapsed, blank separates repeats.
	seq := []int{idx("H"), idx("H"), 0, idx("i"), space, space, 0, space, idx("a"), 0, idx("b"), idx("b")}
	lt := recLogits(seq, realRecT, realRecClasses)
	text, conf := ctcDecode(lt.Data, int(lt.Shape[1]), int(lt.Shape[2]), charset)
	if want := "Hi  ab"; text != want {
		t.Fatalf("decode: got %q, want %q", text, want)
	}
	approx(t, "conf", conf, 0.9, 1e-6)
}

// A head WITHOUT the space class (C == len(keys)+1) must not invent spaces.
func TestCTCDecode_NoSpaceClass(t *testing.T) {
	charset := []string{"a", "b"}
	lt := recLogits([]int{1, 0, 2}, 12, 3)
	if text, _ := ctcDecode(lt.Data, 12, 3, charset); text != "ab" {
		t.Fatalf("got %q, want \"ab\"", text)
	}
	// C == len+2: index 3 is the space.
	lt = recLogits([]int{1, 3, 2}, 12, 4)
	if text, _ := ctcDecode(lt.Data, 12, 4, charset); text != "a b" {
		t.Fatalf("got %q, want \"a b\"", text)
	}
}
