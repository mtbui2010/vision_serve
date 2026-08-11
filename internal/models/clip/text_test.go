package clip

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// fakeRunner stands in for lifecycle's Runner so Infer can be tested without an ONNX
// session: it records the tensor it was handed and replays a canned output.
type fakeRunner struct {
	inputNames []string
	gotInputs  map[string]engine.Tensor
	out        []engine.Tensor
	err        error
}

func (f *fakeRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	f.gotInputs = in
	if f.err != nil {
		return nil, f.err
	}
	return f.out, nil
}
func (f *fakeRunner) InputNames(role string) []string  { return f.inputNames }
func (f *fakeRunner) OutputNames(role string) []string { return []string{"text_embeds"} }

func newTestTextModel(t *testing.T) *textModel {
	t.Helper()
	tok := loadTestTokenizer(t) // skips when model assets are absent
	return &textModel{cfg: models.Config{Name: "clip-text"}, tok: tok}
}

func TestTextModelIdentity(t *testing.T) {
	m := newTestTextModel(t)
	if m.Task() != models.TaskEmbed {
		t.Errorf("Task() = %q, want %q", m.Task(), models.TaskEmbed)
	}
	if got := m.Roles(); !reflect.DeepEqual(got, []string{"model"}) {
		t.Errorf("Roles() = %v, want [model]", got)
	}
	if !models.IsRegistered("clip-text") {
		t.Error("architecture \"clip-text\" is not registered")
	}
}

// TestInferTokenizesAndShapes: the prompt is split into phrases, each becomes one row of
// an int64 [N,77] tensor bound to "input_ids", and the output rows come back in the SAME
// order (the contract a text-aligned class matrix depends on).
func TestInferTokenizesAndShapes(t *testing.T) {
	m := newTestTextModel(t)

	// 3 phrases × 2 dims, deliberately unnormalised so we can check normalisation.
	r := &fakeRunner{
		inputNames: []string{"input_ids"},
		out:        []engine.Tensor{engine.F32([]float32{3, 4, 0, 5, -6, 8}, 3, 2)},
	}

	res, err := m.Infer(nil, models.Prompt{Text: "cup. remote. water bottle."}, r)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}

	in, ok := r.gotInputs["input_ids"]
	if !ok {
		t.Fatalf("input_ids not bound, got keys %v", keys(r.gotInputs))
	}
	if in.Dtype != "i64" {
		t.Errorf("input dtype = %q, want i64", in.Dtype)
	}
	if want := []int64{3, ContextLength}; !reflect.DeepEqual(in.Shape, want) {
		t.Errorf("input shape = %v, want %v", in.Shape, want)
	}
	if len(in.DataI64) != 3*ContextLength {
		t.Fatalf("input data len = %d, want %d", len(in.DataI64), 3*ContextLength)
	}
	// Row 1 must be the tokenization of the SECOND phrase ("remote"), in order.
	wantRow := m.tok.EncodePadded("remote")
	if got := in.DataI64[ContextLength : 2*ContextLength]; !reflect.DeepEqual(got, wantRow) {
		t.Errorf("row 1 = %v…, want %v…", got[:5], wantRow[:5])
	}

	if res.Task != models.TaskEmbed {
		t.Errorf("Result.Task = %q, want %q", res.Task, models.TaskEmbed)
	}
	if len(res.Embeddings) != 3 {
		t.Fatalf("got %d embeddings, want 3", len(res.Embeddings))
	}
	// Rows must be L2-normalised so cosine vs a clip image embedding is a dot product.
	for i, e := range res.Embeddings {
		var n float64
		for _, v := range e {
			n += float64(v) * float64(v)
		}
		if math.Abs(math.Sqrt(n)-1) > 1e-5 {
			t.Errorf("embedding %d norm = %f, want 1", i, math.Sqrt(n))
		}
	}
	// [3,4] -> [0.6,0.8]
	if math.Abs(float64(res.Embeddings[0][0])-0.6) > 1e-6 ||
		math.Abs(float64(res.Embeddings[0][1])-0.8) > 1e-6 {
		t.Errorf("row 0 = %v, want [0.6 0.8]", res.Embeddings[0])
	}
}

// TestInferFallbackInputName: if the session's input is named differently, bind the
// first declared input instead of a name the graph does not have.
func TestInferFallbackInputName(t *testing.T) {
	m := newTestTextModel(t)
	r := &fakeRunner{
		inputNames: []string{"tokens"},
		out:        []engine.Tensor{engine.F32([]float32{1, 0}, 1, 2)},
	}
	if _, err := m.Infer(nil, models.Prompt{Text: "cup"}, r); err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if _, ok := r.gotInputs["tokens"]; !ok {
		t.Errorf("expected fallback to input name \"tokens\", got %v", keys(r.gotInputs))
	}
}

func TestInferErrors(t *testing.T) {
	m := newTestTextModel(t)

	if _, err := m.Infer(nil, models.Prompt{Text: "  . . "}, &fakeRunner{}); err == nil {
		t.Error("empty prompt must be an error, not an empty result")
	}

	r := &fakeRunner{inputNames: []string{"input_ids"}, err: errors.New("boom")}
	if _, err := m.Infer(nil, models.Prompt{Text: "cup"}, r); err == nil {
		t.Error("session failure must propagate")
	}

	// Batch mismatch: 2 prompts but the graph returned 1 row.
	r = &fakeRunner{
		inputNames: []string{"input_ids"},
		out:        []engine.Tensor{engine.F32([]float32{1, 0}, 1, 2)},
	}
	if _, err := m.Infer(nil, models.Prompt{Text: "cup. remote."}, r); err == nil {
		t.Error("batch-size mismatch must be an error")
	}
}

func TestDecodeTextEmbeddingsErrors(t *testing.T) {
	cases := []struct {
		name string
		outs []engine.Tensor
		n    int
	}{
		{"no outputs", nil, 1},
		{"rank 1", []engine.Tensor{engine.F32([]float32{1, 2}, 2)}, 1},
		{"short data", []engine.Tensor{{Data: []float32{1}, Shape: []int64{1, 2}}}, 1},
	}
	for _, c := range cases {
		if _, err := decodeTextEmbeddings(c.outs, c.n); err == nil {
			t.Errorf("%s: expected an error", c.name)
		}
	}
}

func TestSplitPhrases(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"cup. remote.", []string{"cup", "remote"}},
		{"cup.. remote.", []string{"cup", "remote"}},
		{"a photo of a water bottle", []string{"a photo of a water bottle"}},
		{"  ", nil},
		{"", nil},
	}
	for _, c := range cases {
		if got := SplitPhrases(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("SplitPhrases(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestL2Normalize(t *testing.T) {
	if got := l2Normalize([]float32{0, 0, 0}); !reflect.DeepEqual(got, []float32{0, 0, 0}) {
		t.Errorf("zero vector = %v, want unchanged (no NaN)", got)
	}
	got := l2Normalize([]float32{3, 4})
	if math.Abs(float64(got[0])-0.6) > 1e-6 || math.Abs(float64(got[1])-0.8) > 1e-6 {
		t.Errorf("l2Normalize([3 4]) = %v, want [0.6 0.8]", got)
	}
	// Must not modify the input in place (callers reuse the output tensor buffer).
	src := []float32{3, 4}
	_ = l2Normalize(src)
	if src[0] != 3 || src[1] != 4 {
		t.Errorf("input was mutated: %v", src)
	}
}

func TestNewTextRequiresDir(t *testing.T) {
	if _, err := NewText(models.Config{Name: "clip-text"}); err == nil {
		t.Error("NewText with no Dir must fail (tokenizer assets cannot be found)")
	}
	if _, err := NewText(models.Config{Name: "clip-text", Dir: "/nonexistent-dir"}); err == nil {
		t.Error("NewText with a bad Dir must fail")
	}
}

func keys(m map[string]engine.Tensor) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
