package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"visionserve/internal/cli/clireport"
	"visionserve/internal/engine"
	"visionserve/internal/engine/onnxtest"
	"visionserve/internal/registry"
	"visionserve/internal/vision/preprocess"

	// The architectures the fixtures use (main.go registers them in the binary).
	_ "visionserve/internal/models/classification"
	_ "visionserve/internal/models/depth"
	_ "visionserve/internal/models/detr"
)

var update = flag.Bool("update", false, "rewrite the golden files of inspect/import")

// cleanEnv makes the EP chain and pool settings deterministic for a test.
func cleanEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{"VISIONSERVE_EP", "VISIONSERVE_TENSORRT", "VS_POOL_OVERRIDE", "VISIONSERVE_POOL_THREADS", "VISIONSERVE_VERIFY"} {
		t.Setenv(k, "")
	}
	engine.SetTensorRT(false)
	t.Cleanup(func() { engine.SetTensorRT(false) })
}

// clsFixture is a tiny classifier: input pixels [batch,3,8,8], output logits [batch,4], 112
// float32 parameters (conv.w 4×3×3×3 = 108, fc.b 4).
func clsFixture(t *testing.T, path string) {
	t.Helper()
	onnxtest.WriteWith(t, path, onnxtest.Options{
		ProducerName: "onnxtest", ProducerVersion: "1.0", Opset: 17,
		Initializers: []onnxtest.Input{{Name: "conv.w", Dims: []int64{4, 3, 3, 3}}, {Name: "fc.b", Dims: []int64{4}}},
	}, []onnxtest.Input{{Name: "pixels", Dims: []int64{-1, 3, 8, 8}}}, onnxtest.Input{Name: "logits", Dims: []int64{-1, 4}})
}

// writeClsModel writes <root>/<dir>/ with the classifier, 4 labels, and a manifest whose input
// size is size×size, pinned to the ONNX file's real digest.
func writeClsModel(t *testing.T, root, dir string, size int) string {
	t.Helper()
	md := filepath.Join(root, dir)
	if err := os.MkdirAll(md, 0o755); err != nil {
		t.Fatal(err)
	}
	clsFixture(t, filepath.Join(md, "model.onnx"))
	if err := os.WriteFile(filepath.Join(md, "labels.txt"), []byte("cat\ndog\nbird\nfish\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sum, err := registry.FileSHA256(filepath.Join(md, "model.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	y := "name: " + dir + "\ntask: classification\nlicense: apache-2.0\narchitecture: efficientnet\nmodel_file: model.onnx\n" +
		"sha256: " + sum + "\nlabels: labels.txt\ninput:\n  width: " + itoa(size) + "\n  height: " + itoa(size) +
		"\n  normalize:\n    mean: [0.485, 0.456, 0.406]\n    std: [0.229, 0.224, 0.225]\n" +
		"postprocess:\n  type: classification\n  max_detections: 3\nruntime:\n  prefer: [cuda, cpu]\n  idle_unload_seconds: 300\n"
	if err := os.WriteFile(filepath.Join(md, "manifest.yaml"), []byte(y), 0o644); err != nil {
		t.Fatal(err)
	}
	return md
}

func itoa(n int) string { return strconv.Itoa(n) }

func textOf(t *testing.T, r *clireport.Report) string {
	t.Helper()
	var b bytes.Buffer
	if err := r.WriteText(&b); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

// golden compares got (with root replaced by $ROOT) against testdata/<name>; -update rewrites it.
func golden(t *testing.T, name, root, got string) {
	t.Helper()
	got = strings.ReplaceAll(got, root, "$ROOT")
	path := filepath.Join("testdata", name)
	if *update {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run go test ./internal/cli -run %s -update to create it)", err, t.Name())
	}
	if got != string(want) {
		t.Errorf("%s differs from the golden file (go test -update to accept):\n--- got ---\n%s\n--- want ---\n%s", name, got, want)
	}
}

// The text card of a model that is ready, and of the same model with a manifest size its graph
// cannot take. Every number in them follows from the fixture: 112 parameters × 4 bytes, an
// 8×8 input, 4 labels for 4 outputs.
func TestInspectGoldenText(t *testing.T) {
	cleanEnv(t)
	root := t.TempDir()
	models := filepath.Join(root, "models")
	writeClsModel(t, models, "demo-cls", 8)
	writeClsModel(t, models, "demo-bad-size", 16)

	r, err := inspect("demo-cls", inspectOptions{ModelsDir: models})
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != clireport.Pass {
		t.Errorf("verdict %s: %s", r.Verdict, r.Reason)
	}
	golden(t, "inspect_pass.txt", root, textOf(t, r))

	r, err = inspect("demo-bad-size", inspectOptions{ModelsDir: models})
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != clireport.Fail || !strings.Contains(r.Reason, "prepares a [1,3,16,16] tensor") ||
		!strings.Contains(r.Reason, `model.onnx input "pixels" takes [?,3,8,8]`) {
		t.Errorf("size mismatch: %s: %s", r.Verdict, r.Reason)
	}
	golden(t, "inspect_fail.txt", root, textOf(t, r))

	// The same model by folder path and by manifest path gives the same card.
	byDir, err := inspect(filepath.Join(models, "demo-cls"), inspectOptions{ModelsDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if textOf(t, byDir) != textOf(t, r0(t, models)) {
		t.Error("inspect <dir> and inspect <name> differ")
	}
}

func r0(t *testing.T, models string) *clireport.Report {
	r, err := inspect("demo-cls", inspectOptions{ModelsDir: models})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestInspectBareONNX(t *testing.T) {
	cleanEnv(t)
	root := t.TempDir()
	path := filepath.Join(root, "my net.onnx")
	clsFixture(t, path)
	r, err := inspect(path, inspectOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != clireport.Warn || !strings.Contains(r.Reason, "has no manifest") {
		t.Errorf("bare onnx: %s %s", r.Verdict, r.Reason)
	}
	d := r.Details.(inspectDetails)
	want := "visionserve import '" + path + "' --name my-net --task classification --license LICENSE-ID"
	if d.ImportCommand != want {
		t.Errorf("import command\n got %s\nwant %s", d.ImportCommand, want)
	}
	golden(t, "inspect_bare.txt", root, textOf(t, r))

	// Not an ONNX file: FAIL, not a crash.
	junk := filepath.Join(root, "junk.onnx")
	if err := os.WriteFile(junk, []byte("not a model"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := inspect(junk, inspectOptions{}); err != nil || r.Verdict != clireport.Fail {
		t.Errorf("junk: %v %+v", err, r)
	}
	// A missing path or unknown name is a usage error (exit 2).
	for _, target := range []string{filepath.Join(root, "nope.onnx"), "no-such-model"} {
		if _, err := inspect(target, inspectOptions{ModelsDir: root}); clireport.ExitCode(err) != clireport.ExitUsage {
			t.Errorf("%s: %v (exit %d)", target, err, clireport.ExitCode(err))
		}
	}
}

// A manifest the registry refuses (an AGPL licence) is a FAIL card naming the reason.
func TestInspectRefusedManifest(t *testing.T) {
	cleanEnv(t)
	models := t.TempDir()
	md := writeClsModel(t, models, "agpl", 8)
	raw, _ := os.ReadFile(filepath.Join(md, "manifest.yaml"))
	if err := os.WriteFile(filepath.Join(md, "manifest.yaml"), bytes.Replace(raw, []byte("apache-2.0"), []byte("AGPL-3.0"), 1), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := inspect("agpl", inspectOptions{ModelsDir: models})
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != clireport.Fail || !strings.Contains(r.Reason, `license "AGPL-3.0" is not allowed`) {
		t.Errorf("%s: %s", r.Verdict, r.Reason)
	}

	// A typo in a manifest key is a WARN (the setting silently keeps its default), by name and by
	// folder alike.
	md = writeClsModel(t, models, "typo", 8)
	f, err := os.OpenFile(filepath.Join(md, "manifest.yaml"), os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("idle_unload_second: 5\n")
	f.Close()
	for _, target := range []string{"typo", md} {
		r, err := inspect(target, inspectOptions{ModelsDir: models})
		if err != nil {
			t.Fatal(err)
		}
		if r.Verdict != clireport.Warn || !strings.Contains(r.Reason, "idle_unload_second") {
			t.Errorf("%s: %s %s", target, r.Verdict, r.Reason)
		}
	}
}

// --- JSON schema stability ---

// jsonPaths lists every key path of a JSON value (arrays as [], map keys of free-form maps
// collapsed to *), sorted.
func jsonPaths(t *testing.T, raw []byte) []string {
	t.Helper()
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, raw)
	}
	free := map[string]bool{".details.onnx[].opsets": true, ".details.onnx[].params_by_type": true,
		".details.onnx[].metadata": true, ".details.card.onnx[].opsets": true, ".details.card.onnx[].params_by_type": true,
		".details.card.onnx[].metadata": true}
	seen := map[string]bool{}
	var walk func(prefix string, v any)
	walk = func(prefix string, v any) {
		switch x := v.(type) {
		case map[string]any:
			for k, c := range x {
				p := prefix + "." + k
				if free[prefix] {
					p = prefix + ".*"
				}
				seen[p] = true
				walk(p, c)
			}
		case []any:
			for _, c := range x {
				walk(prefix+"[]", c)
			}
		}
	}
	walk("", v)
	out := make([]string, 0, len(seen))
	for p := range seen {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// The --json field names are API: this pins every key path of inspect (a model, a bare file) and
// import. Adding a field means adding it to the golden list; renaming or removing one fails.
func TestJSONSchemaStable(t *testing.T) {
	cleanEnv(t)
	root := t.TempDir()
	models := filepath.Join(root, "models")
	writeClsModel(t, models, "demo-cls", 8)
	photo := filepath.Join(root, "photo.png")
	writePNG(t, photo, 20, 10)

	var all []string
	r, err := inspect("demo-cls", inspectOptions{ModelsDir: models, Image: photo, ImageOut: filepath.Join(root, "in.png")})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := r.JSON()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range jsonPaths(t, raw) {
		all = append(all, "inspect-model "+p)
	}

	bare := filepath.Join(root, "bare.onnx")
	clsFixture(t, bare)
	r, err = inspect(bare, inspectOptions{})
	if err != nil {
		t.Fatal(err)
	}
	raw, _ = r.JSON()
	for _, p := range jsonPaths(t, raw) {
		all = append(all, "inspect-bare "+p)
	}

	r, err = importModel(importRequest{Source: bare, Name: "imp", Task: "classification", License: "MIT", ModelsDir: filepath.Join(root, "reg")}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	raw, _ = r.JSON()
	for _, p := range jsonPaths(t, raw) {
		all = append(all, "import "+p)
	}
	golden(t, "json_keys.txt", root, strings.Join(all, "\n")+"\n")
}

func writePNG(t *testing.T, path string, w, h int) {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{uint8(x * 12), uint8(y * 25), 100, 255})
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// --image runs the model's own preprocessing: the tensor, its range and mapping, and a picture
// that undoes the normalisation back to the (squashed) photo.
func TestInspectImage(t *testing.T) {
	cleanEnv(t)
	root := t.TempDir()
	models := filepath.Join(root, "models")
	writeClsModel(t, models, "demo-cls", 8)
	photo := filepath.Join(root, "photo.png")
	writePNG(t, photo, 16, 8)
	out := filepath.Join(root, "seen.png")
	r, err := inspect("demo-cls", inspectOptions{ModelsDir: models, Image: photo, ImageOut: out})
	if err != nil {
		t.Fatal(err)
	}
	d := r.Details.(inspectDetails)
	im := d.Image
	if im == nil || im.Picture != out || im.Input != "pixels" || len(im.Shape) != 4 || im.Shape[2] != 8 || im.Shape[3] != 8 {
		t.Fatalf("image details: %+v (findings %+v)", im, r.Findings)
	}
	if im.Meta == nil || im.Meta.ScaleX != 0.5 || im.Meta.ScaleY != 1 || im.Meta.OrigWidth != 16 {
		t.Errorf("meta: %+v", im.Meta)
	}
	if im.Inverted != "normalisation undone" {
		t.Errorf("picture from %q", im.Inverted)
	}
	if !fileExists(out) || r.Sections[len(r.Sections)-1].Image == nil {
		t.Error("the picture is neither written nor in the report")
	}
}

// tensorPicture undoes exactly what Spec.Apply did: a photo already at the model size comes back
// within rounding, for each normalisation form and layout.
func TestTensorPictureInvertsTheSpec(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 6, 4))
	for i := range src.Pix {
		src.Pix[i] = uint8((i * 37) % 256)
		if i%4 == 3 {
			src.Pix[i] = 255
		}
	}
	for name, s := range map[string]preprocess.Spec{
		"imagenet nchw": {Resize: preprocess.Squash, Width: 6, Height: 4, Mean: imagenetMean, Std: imagenetStd},
		"plain nhwc":    {Resize: preprocess.Squash, Width: 6, Height: 4, Layout: preprocess.NHWC},
		"0..255 units":  {Resize: preprocess.Squash, Width: 6, Height: 4, NoRescale: true, Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}},
		"raw hwc":       {Resize: preprocess.Squash, Width: 6, Height: 4, NoRescale: true, Layout: preprocess.HWC},
	} {
		tns, _, err := s.Apply(src)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		pic, how := tensorPicture(tns, &s)
		if pic == nil || how != "normalisation undone" {
			t.Fatalf("%s: no picture", name)
		}
		for y := 0; y < 4; y++ {
			for x := 0; x < 6; x++ {
				a := src.NRGBAAt(x, y)
				r, g, b, _ := pic.At(x, y).RGBA()
				for c, pair := range [][2]float64{{float64(a.R), float64(r >> 8)}, {float64(a.G), float64(g >> 8)}, {float64(a.B), float64(b >> 8)}} {
					if math.Abs(pair[0]-pair[1]) > 1 {
						t.Fatalf("%s: pixel (%d,%d) channel %d = %v, want %v", name, x, y, c, pair[1], pair[0])
					}
				}
			}
		}
	}
}

// Exit codes through the command entry points: usage errors are 2.
func TestInspectImportUsageExitCodes(t *testing.T) {
	cleanEnv(t)
	for _, args := range [][]string{
		{},
		{"a", "b"},
		{"--bogus-flag", "x"},
	} {
		if err := runInspect(args); clireport.ExitCode(err) != clireport.ExitUsage {
			t.Errorf("inspect %v: %v (exit %d)", args, err, clireport.ExitCode(err))
		}
	}
	for _, args := range [][]string{
		{},
		{"x.onnx"}, // no --name/--task
		{"--name", "a", "--task", "classification", "missing.onnx"},
	} {
		if err := runImport(args); clireport.ExitCode(err) != clireport.ExitUsage {
			t.Errorf("import %v: %v (exit %d)", args, err, clireport.ExitCode(err))
		}
	}
	if err := runInspect([]string{"-h"}); err != nil {
		t.Errorf("-h: %v", err)
	}
	var e *clireport.ExitError
	if !errors.As(clireport.Usagef("x"), &e) {
		t.Error("Usagef")
	}
}
