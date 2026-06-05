package classification

import (
	"bytes"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"visionserve/internal/models"
)

// These benchmarks quantify VisionServe's per-request CPU cost on the host, WITHOUT touching
// the server core, to explain the W1/W4 finding (GPU inference is <1 ms on A6000, yet the
// served C=1 latency is tens of ms). They attribute that cost to the two pure-Go stages:
//
//	BenchmarkJPEGDecode  — image.Decode of the JPEG bytes (runs in the HTTP handler, OUTSIDE
//	                       the server's reported duration_ms).
//	BenchmarkPreprocess  — resize-to-224 + ImageNet-normalize -> NCHW tensor (the bulk of the
//	                       server's duration_ms; ORT Run is timed separately and is <1 ms).
//	BenchmarkDecodePlusPreprocess — the full pure-Go path a request pays before/around inference.
//
// Run:  go test ./internal/models/classification -bench 'Decode|Preprocess' -benchmem -run x
// Report ns/op in the paper's latency-decomposition table alongside the measured ORT time.
func loadSampleJPEGBytes(b *testing.B) []byte {
	b.Helper()
	// repo-root-relative from internal/models/classification
	path := filepath.Join("..", "..", "..", "test", "testdata", "sample.jpg")
	raw, err := os.ReadFile(path)
	if err != nil {
		b.Skipf("sample image not found (%v) — skipping preprocess benchmark", err)
	}
	return raw
}

func benchCfg() models.Config {
	return models.Config{
		Width: 224, Height: 224,
		Mean: []float32{0.485, 0.456, 0.406},
		Std:  []float32{0.229, 0.224, 0.225},
	}
}

func BenchmarkJPEGDecode(b *testing.B) {
	raw := loadSampleJPEGBytes(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := image.Decode(bytes.NewReader(raw)); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPNGDecode measures decoding the SAME image as PNG instead of JPEG. This matters
// because the Python client (clients/python) PNG-encodes a numpy.ndarray / PIL image before
// sending — so an "ndarray input" still arrives as an encoded image the server must decode,
// and PNG decode (zlib inflate) is typically as slow as or slower than JPEG decode for photos.
func BenchmarkPNGDecode(b *testing.B) {
	raw := loadSampleJPEGBytes(b)
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		b.Fatal(err)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		b.Fatal(err)
	}
	pngBytes := buf.Bytes()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := image.Decode(bytes.NewReader(pngBytes)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocess(b *testing.B) {
	raw := loadSampleJPEGBytes(b)
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		b.Fatal(err)
	}
	cfg := benchCfg()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := preprocess(img, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodePlusPreprocess(b *testing.B) {
	raw := loadSampleJPEGBytes(b)
	cfg := benchCfg()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		img, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			b.Fatal(err)
		}
		if _, _, err := preprocess(img, cfg); err != nil {
			b.Fatal(err)
		}
	}
}
