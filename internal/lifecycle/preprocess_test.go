package lifecycle

import (
	"context"
	"errors"
	"image"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/templates"
)

// fakePipeline builds a token tensor and an image tensor, calls its session, and would go on to
// a second stage — which must never happen during a preprocess capture.
type fakePipeline struct{ secondStage bool }

func (f *fakePipeline) Name() string      { return "fake" }
func (f *fakePipeline) Task() models.Task { return models.TaskDetection }
func (f *fakePipeline) Roles() []string   { return []string{"enc", "dec"} }
func (f *fakePipeline) Infer(img image.Image, p models.Prompt, r models.Runner) (models.Result, error) {
	ids := engine.I64([]int64{101, 7, 102}, 1, 3)
	px := engine.F32([]float32{0.5, -0.5}, 1, 2)
	if _, err := r.Run("enc", map[string]engine.Tensor{"input_ids": ids, "pixel_values": px}); err != nil {
		return models.Result{}, err
	}
	f.secondStage = true
	_, err := r.Run("dec", nil)
	return models.Result{}, err
}

// The capture must hand back exactly what the model built for its FIRST session, and stop the
// pipeline there: preprocess is a debug view and must not run (or pay for) inference.
func TestRecordingRunnerCapturesFirstCallAndStops(t *testing.T) {
	f := &fakePipeline{}
	rec := &recordingRunner{}
	_, err := f.Infer(image.NewRGBA(image.Rect(0, 0, 4, 4)), models.Prompt{}, rec)
	if !errors.Is(err, errCaptured) {
		t.Fatalf("Infer error = %v, want the capture sentinel", err)
	}
	if f.secondStage {
		t.Fatal("the pipeline continued past the captured session call")
	}
	got := rec.named()
	if len(got) != 2 || got[0].Name != "input_ids" || got[1].Name != "pixel_values" || got[0].Role != "enc" {
		t.Fatalf("captured %+v, want input_ids + pixel_values of role enc (sorted by name)", got)
	}
	if got[0].Tensor.DataI64[0] != 101 || got[1].Tensor.Data[1] != -0.5 {
		t.Fatalf("captured values changed: %+v", got)
	}
}

// A pipeline that refuses the request before calling any session (a missing text prompt, an
// unknown template) is a 400 on /api/preprocess, as that endpoint always answered — the typed
// error mapping briefly turned it into a 500.
func TestPreprocessModelRefusalIsInvalidRequest(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "picky", "test-pipe", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	refusal := errors.New("picky requires a text prompt")
	testHooks.setInfer(t, "picky", func(models.Runner) (models.Result, error) { return models.Result{}, refusal })

	_, err := m.Preprocess(context.Background(), "picky", image.NewRGBA(image.Rect(0, 0, 4, 4)), models.Prompt{})
	if !errors.Is(err, ErrInvalidRequest) || !errors.Is(err, refusal) {
		t.Fatalf("Preprocess error = %v, want ErrInvalidRequest wrapping the model's refusal", err)
	}
	if _, err := m.Preprocess(context.Background(), "no-such-model", nil, models.Prompt{}); !errors.Is(err, ErrModelNotFound) {
		t.Fatalf("unknown model: %v, want ErrModelNotFound", err)
	}
}

// /api/preprocess resolves template_name exactly as predict does: an unknown set is the
// caller's error (400) — owlvit's preprocess used to fail with "no template images" (500).
func TestPreprocessResolvesTemplateName(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "inst", "test-pipe", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	store := templates.New()
	if err := store.Register("mug", []image.Image{image.NewRGBA(image.Rect(0, 0, 4, 4))}); err != nil {
		t.Fatal(err)
	}
	m.SetTemplateStore(store)
	testHooks.setInfer(t, "inst", func(r models.Runner) (models.Result, error) {
		_, err := r.Run("x", map[string]engine.Tensor{"in": engine.F32([]float32{1}, 1)})
		return models.Result{}, err
	})
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	if _, err := m.Preprocess(context.Background(), "inst", img, models.Prompt{TemplateName: "nope"}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("unknown template: %v, want ErrInvalidRequest", err)
	}
	if _, err := m.Preprocess(context.Background(), "inst", img, models.Prompt{TemplateName: "mug"}); err != nil {
		t.Fatalf("registered template: %v", err)
	}
}
