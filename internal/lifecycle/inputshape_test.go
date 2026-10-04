package lifecycle

import (
	"context"
	"errors"
	"fmt"
	"image"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/engine/onnxtest"
	"visionserve/internal/models"
)

// writeGraphModel is writeSingleModel with a header-only ONNX file declaring inputs, so the
// load-time shape check reads a real graph header.
func writeGraphModel(t *testing.T, root, name, arch, body string, inputs ...onnxtest.Input) {
	t.Helper()
	writeSingleModel(t, root, name, arch, body)
	onnxtest.Write(t, filepath.Join(root, name, "m.onnx"), inputs)
}

func sized(w, h int) string {
	return fmt.Sprintf("input:\n  width: %d\n  height: %d\n  normalize:\n"+
		"    mean: [0.485, 0.456, 0.406]\n    std: [0.229, 0.224, 0.225]\n", w, h)
}

// A manifest size the graph cannot take is a load error naming the manifest field and the graph
// dims (it used to load, and the first predict failed inside ONNX Runtime naming neither).
// Dynamic graph dims, a matching size and an NHWC export with an NHWC manifest all load.
func TestLoadRefusesAManifestSizeTheGraphCannotTake(t *testing.T) {
	root := t.TempDir()
	in := func(dims ...int64) onnxtest.Input { return onnxtest.Input{Name: "input", Dims: dims} }
	nhwc := "preprocess:\n  resize: squash\n  size: 560\n  layout: NHWC\n"
	cases := []struct {
		name, arch, body string
		graph            onnxtest.Input
		wantErr          []string // substrings; nil = loads
	}{
		{"size-mismatch", "rf-detr", sized(640, 640), in(1, 3, 560, 560),
			[]string{`manifest preprocess width×height 640×640 does not match m.onnx input "input" [1,3,560,560]`,
				"produces [1,3,640,640]"}},
		{"height-only", "rf-detr", sized(560, 448), in(1, 3, 560, 560),
			[]string{"width×height 560×448", "[1,3,560,560]", "produces [1,3,448,560]"}},
		{"same-size", "rf-detr", sized(560, 560), in(1, 3, 560, 560), nil},
		{"dynamic-hw", "rf-detr", sized(640, 640), in(1, 3, -1, -1), nil},
		{"dynamic-batch", "rf-detr", sized(560, 560), in(-1, 3, 560, 560), nil},
		{"dynamic-batch-wrong-size", "rf-detr", sized(640, 640), in(-1, 3, 560, 560),
			[]string{"width×height 640×640", "[-1,3,560,560]"}},
		{"nhwc-both", "rf-detr", nhwc, in(1, 560, 560, 3), nil},
		{"nhwc-manifest-nchw-graph", "rf-detr", nhwc, in(1, 3, 560, 560),
			[]string{"manifest preprocess layout NHWC does not match", "produces [1,560,560,3]"}},
		{"nchw-manifest-nhwc-graph", "rf-detr", sized(560, 560), in(1, 560, 560, 3),
			[]string{"manifest preprocess layout NCHW does not match", "[1,560,560,3]"}},
		{"channels", "rf-detr", sized(560, 560), in(1, 1, 560, 560),
			[]string{"manifest preprocessing (width×height 560×560, layout NCHW) does not match", "[1,1,560,560]"}},
		{"rank", "rf-detr", sized(560, 560), in(3, 560, 560), []string{"[3,560,560]"}},
		// keep_aspect: H and W vary per image, so only batch and channels are judged.
		{"keep-aspect-dynamic", "depth-anything-v2",
			"input:\n  width: 518\n  height: 518\n  keep_aspect: true\n  multiple_of: 14\n", in(1, 3, -1, -1), nil},
		{"keep-aspect-channels", "depth-anything-v2",
			"input:\n  width: 518\n  height: 518\n  keep_aspect: true\n  multiple_of: 14\n", in(1, 1, -1, -1),
			[]string{"[1,1,-1,-1]", "produces [1,3,-1,-1]"}},
	}
	for _, c := range cases {
		writeGraphModel(t, root, c.name, c.arch, c.body, c.graph)
	}
	m, op := newFakeManager(t, scanRegistry(t, root))
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			before := len(op.engines())
			err := m.Load(context.Background(), c.name)
			if c.wantErr == nil {
				if err != nil {
					t.Fatalf("Load: %v", err)
				}
				return
			}
			var se *InputShapeError
			if !errors.As(err, &se) {
				t.Fatalf("Load = %v, want an *InputShapeError", err)
			}
			for _, w := range c.wantErr {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q\nlacks %q", err, w)
				}
			}
			if n := len(op.engines()) - before; n != 0 {
				t.Errorf("a refused load opened %d session(s)", n)
			}
			// A load error, never a caller error: statusOf keeps it a 500.
			if errors.Is(err, ErrInvalidRequest) || errors.Is(err, ErrModelNotFound) {
				t.Errorf("shape error must not be a caller/not-found error: %v", err)
			}
		})
	}
}

// Several inputs and no InputName is ambiguous, and a file the header reader cannot parse is
// not judged here (creating the session reports it): both load as before.
func TestInputShapeSkipsWhatItCannotBind(t *testing.T) {
	root := t.TempDir()
	writeGraphModel(t, root, "two-inputs", "rf-detr", sized(640, 640),
		onnxtest.Input{Name: "a", Dims: []int64{1, 3, 560, 560}}, onnxtest.Input{Name: "b", Dims: []int64{1, 3, 560, 560}})
	writeSingleModel(t, root, "garbage", "rf-detr", sized(640, 640)) // m.onnx is "x"
	m, _ := newFakeManager(t, scanRegistry(t, root))
	for _, name := range []string{"two-inputs", "garbage"} {
		if err := m.Load(context.Background(), name); err != nil {
			t.Errorf("%s: Load = %v, want no check (ambiguous or unreadable)", name, err)
		}
	}
}

// shapeProbe is a plain model with a fixed InputName whose preprocessing always yields shape.
type shapeProbe struct {
	in    string
	shape []int64
}

func (p shapeProbe) Name() string          { return "probe" }
func (p shapeProbe) Task() models.Task     { return models.TaskEmbed }
func (p shapeProbe) InputName() string     { return p.in }
func (p shapeProbe) OutputNames() []string { return nil }
func (p shapeProbe) Postprocess([]engine.Tensor, models.PreprocessMeta) (models.Result, error) {
	return models.Result{}, nil
}
func (p shapeProbe) Preprocess(image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return engine.Tensor{Shape: p.shape}, models.PreprocessMeta{}, nil
}

// explainPipe is a PipelineModel whose explain role is fed ExplainPreprocess's tensor.
type explainPipe struct {
	testPipe
	shape []int64
}

func (p *explainPipe) ExplainPreprocess(image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return engine.Tensor{Shape: p.shape}, models.PreprocessMeta{}, nil
}

func init() {
	models.Register("test-shape-probe", func(cfg models.Config) (models.Base, error) {
		return shapeProbe{in: "pixels", shape: []int64{1, 3, int64(cfg.Height), int64(cfg.Width)}}, nil
	})
	models.Register("test-explain-pipe", func(cfg models.Config) (models.Base, error) {
		return &explainPipe{testPipe: testPipe{name: cfg.Name, roles: []string{"det"}},
			shape: []int64{1, 3, int64(cfg.Height), int64(cfg.Width)}}, nil
	})
}

// A plain model's named input is checked among several; a pipeline is checked only on its
// explain role, and only when that role's graph has a single input.
func TestInputShapeNamedInputAndExplainRole(t *testing.T) {
	root := t.TempDir()
	writeGraphModel(t, root, "named", "test-shape-probe", "input:\n  width: 8\n  height: 8\n",
		onnxtest.Input{Name: "a", Dims: []int64{1, 3, 16, 16}}, onnxtest.Input{Name: "pixels", Dims: []int64{1, 3, 4, 4}})
	writeGraphModel(t, root, "named-ok", "test-shape-probe", "input:\n  width: 4\n  height: 4\n",
		onnxtest.Input{Name: "a", Dims: []int64{1, 3, 16, 16}}, onnxtest.Input{Name: "pixels", Dims: []int64{1, 3, 4, 4}})

	explainManifest := "explain:\n  type: attention\n  role: det\n  outputs:\n    attention: attn\n"
	writeTestModel(t, root, "pipe-explain", "test-explain-pipe", explainManifest, "det")
	onnxtest.Write(t, filepath.Join(root, "pipe-explain", "det.onnx"), []onnxtest.Input{{Name: "images", Dims: []int64{1, 3, 16, 16}}})
	writeTestModel(t, root, "pipe-explain-2in", "test-explain-pipe", explainManifest, "det")
	onnxtest.Write(t, filepath.Join(root, "pipe-explain-2in", "det.onnx"),
		[]onnxtest.Input{{Name: "images", Dims: []int64{1, 3, 16, 16}}, {Name: "mask", Dims: []int64{1, 16, 16}}})
	writeTestModel(t, root, "pipe-no-explain", "test-explain-pipe", "", "det")
	onnxtest.Write(t, filepath.Join(root, "pipe-no-explain", "det.onnx"), []onnxtest.Input{{Name: "images", Dims: []int64{1, 3, 16, 16}}})

	m, _ := newFakeManager(t, scanRegistry(t, root))
	for name, want := range map[string]string{
		"named":            `width×height 8×8 does not match m.onnx input "pixels" [1,3,4,4]`,
		"named-ok":         "",
		"pipe-explain":     `width×height 8×8 does not match det.onnx input "images" [1,3,16,16]`,
		"pipe-explain-2in": "", // two inputs: which one the tensor feeds is not guessed
		"pipe-no-explain":  "", // no explain role: Infer alone knows what each role is fed
	} {
		err := m.Load(context.Background(), name)
		switch {
		case want == "" && err != nil:
			t.Errorf("%s: Load = %v, want success", name, err)
		case want != "" && (err == nil || !strings.Contains(err.Error(), want)):
			t.Errorf("%s: Load = %v, want an error containing %q", name, err, want)
		}
	}
}
