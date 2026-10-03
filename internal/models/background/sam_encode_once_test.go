package background

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// samRunner fakes the two MobileSAM sessions and counts calls per role.
type samRunner struct{ enc, dec int }

func (s *samRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	switch role {
	case roleEncoder:
		s.enc++
		return []engine.Tensor{engine.F32(make([]float32, 256*64*64), 1, 256, 64, 64)}, nil
	case roleDecoder:
		s.dec++
		hw := in["orig_im_size"].Data
		h, w := int(hw[0]), int(hw[1])
		mask := make([]float32, h*w)
		for y := h * 3 / 4; y < h; y++ { // bottom quarter: a border-touching surface
			for x := 0; x < w; x++ {
				mask[y*w+x] = 1
			}
		}
		return []engine.Tensor{engine.F32(mask, 1, 1, int64(h), int64(w)), engine.F32([]float32{0.9}, 1, 1)}, nil
	}
	return nil, fmt.Errorf("unexpected role %q", role)
}
func (s *samRunner) InputNames(role string) []string {
	if role == roleEncoder {
		return []string{"input_image"}
	}
	return []string{"image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"}
}
func (s *samRunner) OutputNames(role string) []string {
	if role == roleEncoder {
		return []string{"image_embeddings"}
	}
	return []string{"masks", "iou_predictions"}
}

func newSAMBackground(t testing.TB, files map[string]string) *backgroundModel {
	t.Helper()
	b, err := New(models.Config{Name: "background", Files: files})
	if err != nil {
		t.Fatal(err)
	}
	return b.(*backgroundModel)
}

// P3: method=sam prompts each seed separately, and each InferMasks call used to re-run the
// encoder on the same image — 6 encoder passes per request. It must be one, with the same
// decoder work and the same result.
func TestBackgroundSAMEncodesOnce(t *testing.T) {
	m := newSAMBackground(t, map[string]string{roleEncoder: "enc.onnx", roleDecoder: "dec.onnx"})
	r := &samRunner{}
	img := image.NewRGBA(image.Rect(0, 0, 80, 60))
	got, err := m.backgroundSAM(img, models.Prompt{}, r)
	if err != nil {
		t.Fatal(err)
	}
	seeds := len(samSeedPoints(80, 60))
	if r.enc != 1 {
		t.Errorf("encoder ran %d times for %d seeds, want 1", r.enc, seeds)
	}
	if r.dec != seeds {
		t.Errorf("decoder ran %d times, want one per seed (%d)", r.dec, seeds)
	}
	if got == nil || int(bitmapArea(got)) != 80*15 {
		t.Errorf("surface area = %v, want the bottom quarter (%d px)", bitmapArea(got), 80*15)
	}
}

// BenchmarkBackgroundSAMRealORT times method=sam end to end on the real MobileSAM sessions (CPU).
// Skipped without ORT_DYLIB_PATH or the weights. Measured 2026-10-03, 640x480, ORT 1.26 CPU,
// 6 seeds: 8.40 s/op encoding per seed, 2.20 s/op encoding once.
func BenchmarkBackgroundSAMRealORT(b *testing.B) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		b.Skip("ORT_DYLIB_PATH not set")
	}
	dir := filepath.Join("..", "..", "..", "models", "mobile-sam")
	encPath, decPath := filepath.Join(dir, "mobile_sam_encoder.onnx"), filepath.Join(dir, "mobile_sam_decoder_single.onnx")
	if _, err := os.Stat(encPath); err != nil {
		b.Skip("MobileSAM weights not present")
	}
	enc, err := engine.NewSession(encPath, nil, nil, []engine.Provider{engine.ProviderCPU})
	if err != nil {
		b.Fatal(err)
	}
	defer enc.Close()
	dec, err := engine.NewSession(decPath, nil, nil, []engine.Provider{engine.ProviderCPU})
	if err != nil {
		b.Fatal(err)
	}
	defer dec.Close()
	r := &sessRunner{s: map[string]*engine.Session{roleEncoder: enc, roleDecoder: dec}}
	m := newSAMBackground(b, map[string]string{roleEncoder: encPath, roleDecoder: decPath})
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.backgroundSAM(img, models.Prompt{}, r); err != nil {
			b.Fatal(err)
		}
	}
}

type sessRunner struct{ s map[string]*engine.Session }

func (r *sessRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	return r.s[role].RunNamed(in)
}
func (r *sessRunner) InputNames(role string) []string  { return r.s[role].InputNames() }
func (r *sessRunner) OutputNames(role string) []string { return r.s[role].OutputNames() }
