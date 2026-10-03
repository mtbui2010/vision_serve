package lifecycle

import (
	"errors"
	"image"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
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
