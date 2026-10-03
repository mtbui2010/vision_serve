package lifecycle

import (
	"context"
	"errors"
	"image"
	"os"
	"path/filepath"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/templates"
)

// testPlain is a plain models.Model (single session, no prompt) for runtime tests.
type testPlain struct{ name string }

func (p testPlain) Name() string          { return p.name }
func (p testPlain) Task() models.Task     { return models.TaskEmbed }
func (p testPlain) InputName() string     { return "" }
func (p testPlain) OutputNames() []string { return nil }
func (p testPlain) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	return engine.F32([]float32{0}, 1, 1), models.PreprocessMeta{OrigWidth: b.Dx(), OrigHeight: b.Dy(), ScaleX: 1, ScaleY: 1}, nil
}
func (p testPlain) Postprocess([]engine.Tensor, models.PreprocessMeta) (models.Result, error) {
	return models.Result{}, nil
}

func init() {
	models.Register("test-plain", func(cfg models.Config) (models.Base, error) { return testPlain{cfg.Name}, nil })
}

func writePlainModel(t *testing.T, root, name string) {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "m.onnx"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	y := "name: " + name + "\ntask: embed\nlicense: MIT\narchitecture: test-plain\nmodel_file: m.onnx\n" +
		"input:\n  width: 8\n  height: 8\nruntime:\n  prefer: [cpu]\n"
	if err := os.WriteFile(filepath.Join(dir, "manifest.yaml"), []byte(y), 0o644); err != nil {
		t.Fatal(err)
	}
}

// The HTTP layer maps these sentinels to status codes (404 / 400 / 503) with errors.Is, so the
// runtime must wrap — not merely mention — them on every path that detects the condition.
func TestTypedErrors(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "pipe", "test-pipe", "")
	writeTestModel(t, root, "gone", "test-pipe", "")
	writePlainModel(t, root, "plain")
	reg := scanRegistry(t, root)
	if err := os.Remove(filepath.Join(root, "gone", "x.onnx")); err != nil { // weights deleted after the scan
		t.Fatal(err)
	}
	m, _ := newFakeManager(t, reg)
	m.SetTemplateStore(templates.New())
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))

	_, predictUnknown := m.Predict(context.Background(), "nope", img)
	_, badTemplate := m.PredictPrompt(context.Background(), "pipe", img, models.Prompt{TemplateName: "never-registered"})
	_, tensorOnPipeline := m.InferTensor(context.Background(), "pipe", engine.F32([]float32{1}, 1, 1))
	_, explainUnsupported := m.Explain(context.Background(), "pipe", img, ExplainRequest{})
	_, explainNegative := m.Explain(context.Background(), "pipe", img, ExplainRequest{DetectionIdx: -1})
	_, preprocessNoImage := m.Preprocess(context.Background(), "plain", nil, models.Prompt{})
	_, preprocessUnknown := m.Preprocess(context.Background(), "nope", img, models.Prompt{})

	for _, c := range []struct {
		name string
		err  error
		want error
	}{
		{"load: not in the registry", m.Load(context.Background(), "nope"), ErrModelNotFound},
		{"load: weights missing on disk", m.Load(context.Background(), "gone"), ErrModelNotFound},
		{"predict: not in the registry", predictUnknown, ErrModelNotFound},
		{"predict: unknown template", badTemplate, ErrInvalidRequest},
		{"tensor-in on a pipeline model", tensorOnPipeline, ErrInvalidRequest},
		{"explain on a model without an explain block", explainUnsupported, ErrInvalidRequest},
		{"explain with a negative detection index", explainNegative, ErrInvalidRequest},
		{"preprocess without an image", preprocessNoImage, ErrInvalidRequest},
		{"preprocess: not in the registry", preprocessUnknown, ErrModelNotFound},
	} {
		if !errors.Is(c.err, c.want) {
			t.Errorf("%s: err = %v, want it to wrap %v", c.name, c.err, c.want)
		}
	}
}
