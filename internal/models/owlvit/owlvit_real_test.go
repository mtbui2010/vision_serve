package owlvit

// Real-weights harness — SKIPPED unless OWLV2_REAL_SCENE / OWLV2_REAL_TEMPLATE are set.
// Runs the actual owlv2_base_patch16.onnx (ORT_DYLIB_PATH must point at libonnxruntime)
// end to end through Infer and, when OWLV2_REAL_OUT is set, dumps the detections as JSON
// so they can be compared against an independent Python reference (HF Owlv2ImageProcessor
// + onnxruntime). Example:
//
//	OWLV2_REAL_SCENE=scene.jpg OWLV2_REAL_TEMPLATE=tmpl.png OWLV2_REAL_OUT=go.json \
//	  go test -run TestRealWeights ./internal/models/owlvit/

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

type sessionRunner struct{ s *engine.Session }

func (r sessionRunner) Run(_ string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	return r.s.RunNamed(context.Background(), in)
}
func (r sessionRunner) InputNames(string) []string  { return r.s.InputNames() }
func (r sessionRunner) OutputNames(string) []string { return r.s.OutputNames() }

func loadImage(t *testing.T, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func TestRealWeights(t *testing.T) {
	scenePath, tmplPath := os.Getenv("OWLV2_REAL_SCENE"), os.Getenv("OWLV2_REAL_TEMPLATE")
	if scenePath == "" || tmplPath == "" {
		t.Skip("set OWLV2_REAL_SCENE and OWLV2_REAL_TEMPLATE to run against the real ONNX")
	}
	_, here, _, _ := runtime.Caller(0)
	onnx := filepath.Join(filepath.Dir(here), "..", "..", "..", "models", "owlvit-base", "owlv2_base_patch16.onnx")
	if _, err := os.Stat(onnx); err != nil {
		t.Skipf("weights not present: %v", err)
	}
	sess, err := engine.NewSession(onnx, nil, nil, []engine.Provider{engine.ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	m, err := New(models.Config{Width: 960, Height: 960, MaxDet: 10, InstancePatchSize: 16}) // as shipped: threshold 0.9
	if err != nil {
		t.Fatal(err)
	}
	res, err := m.(*owlVIT).Infer(loadImage(t, scenePath),
		models.Prompt{TemplateImages: []image.Image{loadImage(t, tmplPath)}}, sessionRunner{sess})
	if err != nil {
		t.Fatal(err)
	}
	if tp := os.Getenv("OWLV2_REAL_TENSOR"); tp != "" { // raw scene tensor, little-endian f32
		tensor, _, err := m.(*owlVIT).preprocessImage(loadImage(t, scenePath))
		if err != nil {
			t.Fatal(err)
		}
		buf := make([]byte, 4*len(tensor.Data))
		for i, v := range tensor.Data {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
		}
		if err := os.WriteFile(tp, buf, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, d := range res.Detections {
		t.Logf("%.1f %.1f %.1f %.1f  %.3f", d.BBox[0], d.BBox[1], d.BBox[2], d.BBox[3], d.Conf)
	}
	if out := os.Getenv("OWLV2_REAL_OUT"); out != "" {
		b, _ := json.Marshal(res.Detections)
		if err := os.WriteFile(out, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
