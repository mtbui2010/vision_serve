package lifecycle

import (
	"errors"
	"fmt"
	"image"
	"sort"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// NamedTensor is one model input as it would be bound to its ONNX session.
type NamedTensor struct {
	Role   string        // manifest role of the session ("model" for single-session models)
	Name   string        // ONNX input name
	Tensor engine.Tensor // exactly the values the session would receive
}

// PreprocessResult is what a model would feed its first ONNX session for one request.
type PreprocessResult struct {
	Inputs []NamedTensor
	// Meta maps model-input coordinates back to the original image. Only set for models whose
	// preprocessing produces it (plain models, and the RF-DETR stage of the hybrid router).
	Meta *models.PreprocessMeta
}

// Preprocess returns the tensors a model would feed its (first) ONNX session for img + prompt,
// WITHOUT running inference and without loading any session. It exists so a caller can compare
// the server's preprocessing with the one a model was trained with — a mismatch there serves a
// working model silently worse (BUGS_TO_FIX.md #1 letterbox: -7.35 mAP; #5 SigLIP padding: -4.3).
//
// The answer must be what serving REALLY does, never a re-implementation that could drift, so:
//   - a plain Model: its own Preprocess, the exact call Session.Predict makes;
//   - a PipelineModel: its own Infer, run against a recording Runner that captures the inputs of
//     the first session call and stops there. Whatever the model builds — resized pixels, a text
//     mask, token ids — is what gets returned.
func (m *Manager) Preprocess(name string, img image.Image, prompt models.Prompt) (PreprocessResult, error) {
	base, man, err := m.buildModel(name)
	if err != nil {
		return PreprocessResult{}, err
	}
	switch mdl := base.(type) {
	case models.Model:
		if img == nil {
			return PreprocessResult{}, fmt.Errorf("preprocess: %q takes an image", name)
		}
		in, meta, err := mdl.Preprocess(img)
		if err != nil {
			return PreprocessResult{}, err
		}
		inName := mdl.InputName()
		if inName == "" {
			if ins, _, err := engine.Inspect(man.ModelFilePath()); err == nil && len(ins) > 0 {
				inName = ins[0].Name
			}
		}
		return PreprocessResult{
			Inputs: []NamedTensor{{Role: "model", Name: inName, Tensor: in}},
			Meta:   &meta,
		}, nil
	case models.PipelineModel:
		rec := &recordingRunner{files: man.FilesAbs()}
		if img == nil {
			img = image.NewRGBA(image.Rect(0, 0, 1, 1)) // text-only models ignore the image
		}
		_, inferErr := mdl.Infer(img, prompt, rec)
		if rec.inputs == nil {
			if inferErr == nil {
				inferErr = errors.New("the model returned without calling any session")
			}
			return PreprocessResult{}, fmt.Errorf("preprocess %s: %w", name, inferErr)
		}
		res := PreprocessResult{Inputs: rec.named()}
		// The hybrid router can also say how its RF-DETR stage maps boxes back.
		if ep, ok := mdl.(models.ExplainPreprocessor); ok && img.Bounds().Dx() > 1 {
			if _, meta, err := ep.ExplainPreprocess(img); err == nil {
				res.Meta = &meta
			}
		}
		return res, nil
	default:
		return PreprocessResult{}, fmt.Errorf("preprocess: %q (%T) exposes no preprocessing", name, base)
	}
}

// errCaptured stops a pipeline right after its first session call has been recorded.
var errCaptured = errors.New("lifecycle: preprocess capture (no inference run)")

// recordingRunner is a models.Runner that runs nothing: it reports each role's real ONNX input
// and output names (read from the file, no session), records the first Run call and aborts the
// pipeline with errCaptured.
type recordingRunner struct {
	files  map[string]string
	role   string
	inputs map[string]engine.Tensor
}

func (r *recordingRunner) Run(role string, inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
	if r.inputs == nil {
		r.role, r.inputs = role, inputs
	}
	return nil, errCaptured
}

func (r *recordingRunner) InputNames(role string) []string  { return r.ioNames(role, true) }
func (r *recordingRunner) OutputNames(role string) []string { return r.ioNames(role, false) }

func (r *recordingRunner) ioNames(role string, inputs bool) []string {
	path, ok := r.files[role]
	if !ok {
		return nil
	}
	ins, outs, err := engine.Inspect(path)
	if err != nil {
		return nil
	}
	src := outs
	if inputs {
		src = ins
	}
	names := make([]string, len(src))
	for i, io := range src {
		names[i] = io.Name
	}
	return names
}

// named returns the recorded inputs sorted by name, so the response is deterministic.
func (r *recordingRunner) named() []NamedTensor {
	keys := make([]string, 0, len(r.inputs))
	for k := range r.inputs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]NamedTensor, 0, len(keys))
	for _, k := range keys {
		out = append(out, NamedTensor{Role: r.role, Name: k, Tensor: r.inputs[k]})
	}
	return out
}
