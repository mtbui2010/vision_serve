package lifecycle

import (
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/models"
	"visionserve/internal/vision/preprocess"

	_ "visionserve/internal/models/clip"
	_ "visionserve/internal/models/depth"
	_ "visionserve/internal/models/detr"
	_ "visionserve/internal/models/scrfd"
)

// writeSingleModel writes <root>/<name>/manifest.yaml (+ an empty m.onnx) for a single-session
// architecture; body holds the input: and/or preprocess: blocks.
func writeSingleModel(t *testing.T, root, name, arch, body string) {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "m.onnx"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	y := "name: " + name + "\ntask: detection\nlicense: MIT\narchitecture: " + arch +
		"\nmodel_file: m.onnx\nruntime:\n  prefer: [cpu]\n" + body
	if err := os.WriteFile(filepath.Join(dir, "manifest.yaml"), []byte(y), 0o644); err != nil {
		t.Fatal(err)
	}
}

func specTestImage(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{uint8(x * 7), uint8(y * 3), uint8((x + y) * 5), 255})
		}
	}
	return img
}

// A manifest that declares its preprocessing in a preprocess: block feeds the model exactly
// what the equivalent legacy input.* manifest feeds it (/api/preprocess runs this path).
func TestPreprocessBlockMatchesLegacyManifest(t *testing.T) {
	const imagenet = "    mean: [0.485, 0.456, 0.406]\n    std: [0.229, 0.224, 0.225]\n"
	pairs := []struct {
		arch, legacy, block string
	}{
		{"rf-detr",
			"input:\n  width: 560\n  height: 560\n  layout: NCHW\n  letterbox: false\n  normalize:\n" + imagenet,
			"preprocess:\n  resize: squash\n  size: 560\n  layout: NCHW\n" + strings.ReplaceAll(imagenet, "    ", "  ")},
		{"rt-detr",
			"input:\n  width: 640\n  height: 480\n  letterbox: true\n  normalize:\n" + imagenet,
			"preprocess:\n  resize: letterbox\n  width: 640\n  height: 480\n" + strings.ReplaceAll(imagenet, "    ", "  ")},
		{"scrfd",
			"input:\n  width: 640\n  height: 640\n  layout: NCHW\n  letterbox: true\n  normalize:\n    mean: [127.5, 127.5, 127.5]\n    std: [128.0, 128.0, 128.0]\n",
			"preprocess:\n  resize: top_left_pad\n  size: 640\n  rescale: false\n  mean: [127.5, 127.5, 127.5]\n  std: [128.0, 128.0, 128.0]\n"},
		// both: the shipped scrfd fields kept for old servers, plus the block
		{"scrfd",
			"input:\n  width: 640\n  height: 640\n  letterbox: true\n  normalize:\n    mean: [127.5, 127.5, 127.5]\n    std: [128.0, 128.0, 128.0]\n",
			"input:\n  width: 640\n  height: 640\n  letterbox: true\n  normalize:\n    mean: [127.5, 127.5, 127.5]\n    std: [128.0, 128.0, 128.0]\n" +
				"preprocess:\n  resize: top_left_pad\n  rescale: false\n"},
		{"clip",
			"input:\n  width: 224\n  height: 224\n  crop: center\n  normalize:\n    mean: [0.48145466, 0.4578275, 0.40821073]\n    std: [0.26862954, 0.26130258, 0.27577711]\n",
			"preprocess:\n  resize: center_crop\n  size: 224\n"},
		{"depth-anything-v2",
			"input:\n  width: 518\n  height: 518\n  keep_aspect: true\n  multiple_of: 14\n  normalize:\n" + imagenet,
			"preprocess:\n  resize: keep_aspect\n  size: 518\n  multiple_of: 14\n  resample: bicubic\n" + strings.ReplaceAll(imagenet, "    ", "  ")},
	}
	img := specTestImage(97, 173)
	for i, p := range pairs {
		root := t.TempDir()
		writeSingleModel(t, root, "legacy", p.arch, p.legacy)
		writeSingleModel(t, root, "block", p.arch, p.block)
		m, _ := newFakeManager(t, scanRegistry(t, root))
		a, err := m.Preprocess("legacy", img, models.Prompt{})
		if err != nil {
			t.Fatalf("%d %s legacy: %v", i, p.arch, err)
		}
		b, err := m.Preprocess("block", img, models.Prompt{})
		if err != nil {
			t.Fatalf("%d %s block: %v", i, p.arch, err)
		}
		ta, tb := a.Inputs[0].Tensor, b.Inputs[0].Tensor
		if len(ta.Data) == 0 || len(ta.Data) != len(tb.Data) || len(ta.Shape) != len(tb.Shape) {
			t.Fatalf("%d %s: shapes %v vs %v", i, p.arch, ta.Shape, tb.Shape)
		}
		for k := range ta.Shape {
			if ta.Shape[k] != tb.Shape[k] {
				t.Fatalf("%d %s: shapes %v vs %v", i, p.arch, ta.Shape, tb.Shape)
			}
		}
		for k := range ta.Data {
			if math.Float32bits(ta.Data[k]) != math.Float32bits(tb.Data[k]) {
				t.Fatalf("%d %s: value %d: legacy %v, block %v", i, p.arch, k, ta.Data[k], tb.Data[k])
			}
		}
		if *a.Meta != *b.Meta {
			t.Fatalf("%d %s: meta %+v vs %+v", i, p.arch, *a.Meta, *b.Meta)
		}
	}
}

// A declared mode the architecture cannot serve is a load error naming it; the same flag in the
// legacy fields keeps being ignored, as it always was.
func TestPreprocessBlockUnsupportedModeFailsLoad(t *testing.T) {
	root := t.TempDir()
	writeSingleModel(t, root, "detr-ka", "rf-detr", "preprocess:\n  resize: keep_aspect\n  size: 560\n  multiple_of: 14\n")
	writeSingleModel(t, root, "scrfd-lb", "scrfd", "preprocess:\n  resize: letterbox\n  size: 640\n")
	writeSingleModel(t, root, "scrfd-legacy", "scrfd", "input:\n  width: 640\n  height: 640\n  letterbox: true\n")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	img := specTestImage(31, 17)
	for name, want := range map[string]string{"detr-ka": `rfdetr does not support resize "keep_aspect" (supported: squash, letterbox)`,
		"scrfd-lb": `does not support resize "letterbox"`} {
		if _, err := m.Preprocess(name, img, models.Prompt{}); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: err = %v, want %q", name, err, want)
		}
		if err := m.Load(name); err == nil {
			t.Errorf("%s: Load succeeded", name)
		}
	}
	if _, err := m.Preprocess("scrfd-legacy", img, models.Prompt{}); err != nil {
		t.Fatalf("legacy scrfd: %v", err)
	}
}

// lifecycle hands every model the resolved spec: Legacy for a manifest without a block, the
// block (with the legacy aliases merged in) otherwise.
var recordedSpecs = map[string]*preprocess.Spec{} // model name -> cfg.Preprocess (test-spec-recorder)

func init() {
	models.Register("test-spec-recorder", func(cfg models.Config) (models.Base, error) {
		testHooks.mu.Lock()
		recordedSpecs[cfg.Name] = cfg.Preprocess
		testHooks.mu.Unlock()
		return &testPipe{name: cfg.Name, roles: []string{"x"}}, nil
	})
}

func TestBuildModelSetsPreprocessSpec(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "plain", "test-spec-recorder", "")
	writeTestModel(t, root, "declared", "test-spec-recorder", "preprocess:\n  resize: long_side_pad\n  multiple_of: 32\n  no_upscale: true\n")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	for _, name := range []string{"plain", "declared"} {
		if _, _, err := m.buildModel(name); err != nil {
			t.Fatal(err)
		}
	}
	testHooks.mu.Lock()
	got := []*preprocess.Spec{recordedSpecs["plain"], recordedSpecs["declared"]}
	testHooks.mu.Unlock()
	if got[0] == nil || got[1] == nil {
		t.Fatalf("specs %v", got)
	}
	if s := *got[0]; !s.Legacy || s.Resize != preprocess.Squash || s.Width != 8 {
		t.Fatalf("plain: %+v", s)
	}
	if s := *got[1]; s.Legacy || s.Resize != preprocess.LongSidePad || s.MultipleOf != 32 || !s.NoUpscale || s.Width != 8 {
		t.Fatalf("declared: %+v", s)
	}
}
