package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/cli/clireport"
	"visionserve/internal/engine/onnxtest"
	"visionserve/internal/registry"
)

func detrFixture(t *testing.T, path string, inDims []int64, outs ...onnxtest.Input) {
	t.Helper()
	if outs == nil {
		outs = []onnxtest.Input{{Name: "dets", Dims: []int64{1, 300, 4}}, {Name: "labels", Dims: []int64{1, 300, 91}}}
	}
	onnxtest.Write(t, path, []onnxtest.Input{{Name: "input", Dims: inDims}}, outs...)
}

func runImportReq(t *testing.T, q importRequest) *clireport.Report {
	t.Helper()
	r, err := importModel(q, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("usage error: %v", err)
	}
	return r
}

// A refused import is a FAIL naming the reason, and installs nothing.
func TestImportRefusals(t *testing.T) {
	cleanEnv(t)
	dir := t.TempDir()
	cls := filepath.Join(dir, "cls.onnx")
	clsFixture(t, cls)
	yolo := filepath.Join(dir, "yolo.onnx")
	onnxtest.Write(t, yolo, []onnxtest.Input{{Name: "images", Dims: []int64{1, 3, 640, 640}}}, onnxtest.Input{Name: "output0", Dims: []int64{1, 84, 8400}})
	detr := filepath.Join(dir, "detr.onnx")
	detrFixture(t, detr, []int64{1, 3, 560, 560})
	twoIn := filepath.Join(dir, "two.onnx")
	onnxtest.Write(t, twoIn, []onnxtest.Input{{Name: "image", Dims: []int64{1, 3, 8, 8}}, {Name: "size", Dims: []int64{2}}}, onnxtest.Input{Name: "y", Dims: []int64{1, 4}})
	half := filepath.Join(dir, "half.onnx")
	onnxtest.Write(t, half, []onnxtest.Input{{Name: "x", Dims: []int64{1, 3, 8, 8}, Elem: 10}}, onnxtest.Input{Name: "y", Dims: []int64{1, 4}})
	gray := filepath.Join(dir, "gray.onnx")
	onnxtest.Write(t, gray, []onnxtest.Input{{Name: "x", Dims: []int64{1, 1, 28, 28}}}, onnxtest.Input{Name: "y", Dims: []int64{1, 10}})
	dyn := filepath.Join(dir, "dyn.onnx")
	onnxtest.Write(t, dyn, []onnxtest.Input{{Name: "x", Dims: []int64{-1, 3, -1, -1}}}, onnxtest.Input{Name: "y", Dims: []int64{-1, 4}})
	ultra := filepath.Join(dir, "ultra.onnx")
	onnxtest.WriteWith(t, ultra, onnxtest.Options{Metadata: map[string]string{"author": "Ultralytics", "license": "AGPL-3.0 License (https://ultralytics.com/license)"}},
		[]onnxtest.Input{{Name: "x", Dims: []int64{1, 3, 8, 8}}}, onnxtest.Input{Name: "y", Dims: []int64{1, 4}})
	labels5 := filepath.Join(dir, "five.txt")
	if err := os.WriteFile(labels5, []byte("a\nb\nc\nd\ne\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	existing := filepath.Join(dir, "reg-existing")
	writeClsModel(t, existing, "taken", 8)

	for name, c := range map[string]struct {
		q    importRequest
		want string
	}{
		"no license":      {importRequest{Source: cls, Task: "classification"}, "a --license is required"},
		"agpl license":    {importRequest{Source: cls, Task: "classification", License: "AGPL-3.0"}, `license "AGPL-3.0" is not allowed`},
		"gpl license":     {importRequest{Source: cls, Task: "classification", License: "GPL-3.0"}, `license "GPL-3.0" is not allowed`},
		"agpl content":    {importRequest{Source: ultra, Task: "classification", License: "MIT"}, "Ultralytics / YOLO-World / FastSAM (AGPL)"},
		"unknown task":    {importRequest{Source: cls, Task: "yolo", License: "MIT"}, `task "yolo" is invalid`},
		"unimportable":    {importRequest{Source: cls, Task: "segmentation", License: "MIT"}, `cannot build a "segmentation" model from the ONNX header alone`},
		"yolo outputs":    {importRequest{Source: yolo, Task: "detection", License: "MIT"}, "not a DETR-style set of queries"},
		"cls outputs":     {importRequest{Source: detr, Task: "classification", License: "MIT"}, "must output one row of class scores"},
		"size mismatch":   {importRequest{Source: detr, Task: "detection", License: "MIT", Input: "640x640"}, `prepares a [1,3,640,640] tensor (input 640×640, layout NCHW) but import-test.onnx input "input" takes [1,3,560,560]`},
		"two inputs":      {importRequest{Source: twoIn, Task: "classification", License: "MIT"}, "the graph has 2 inputs (image, size)"},
		"float16 input":   {importRequest{Source: half, Task: "classification", License: "MIT"}, "is float16"},
		"grayscale":       {importRequest{Source: gray, Task: "classification", License: "MIT"}, "no axis of 3 colour channels"},
		"dynamic size":    {importRequest{Source: dyn, Task: "classification", License: "MIT"}, "pass --input WxH"},
		"labels count":    {importRequest{Source: cls, Task: "classification", License: "MIT", Labels: labels5}, "five.txt has 5 names but the model outputs 4 class scores"},
		"bad name":        {importRequest{Source: cls, Name: "../x", Task: "classification", License: "MIT"}, `name "../x" is invalid`},
		"exists":          {importRequest{Source: cls, Name: "taken", Task: "classification", License: "MIT", ModelsDir: existing}, "already exists"},
		"resize refused":  {importRequest{Source: cls, Task: "classification", License: "MIT", Resize: "letterbox"}, "does not support resize"},
		"layout mismatch": {importRequest{Source: cls, Task: "classification", License: "MIT", Layout: "NHWC"}, "--layout NHWC puts the colour channels at axis 3"},
	} {
		t.Run(name, func(t *testing.T) {
			if c.q.Name == "" {
				c.q.Name = "import-test"
			}
			if c.q.ModelsDir == "" {
				c.q.ModelsDir = filepath.Join(t.TempDir(), "reg")
			}
			if name == "size mismatch" { // the staged file keeps the source's name
				p := filepath.Join(t.TempDir(), "import-test.onnx")
				detrFixture(t, p, []int64{1, 3, 560, 560})
				c.q.Source = p
			}
			r := runImportReq(t, c.q)
			if r.Verdict != clireport.Fail || !strings.Contains(r.Reason, c.want) {
				t.Errorf("verdict %s, reason %q; want FAIL containing %q", r.Verdict, r.Reason, c.want)
			}
			if name != "exists" {
				if _, err := os.Stat(filepath.Join(c.q.ModelsDir, c.q.Name)); err == nil {
					t.Error("a refused import installed something")
				}
			}
			if clireport.ExitCode((&clireport.Output{}).Emit(&bytes.Buffer{}, &bytes.Buffer{}, r)) != clireport.ExitFail {
				t.Error("a refusal must exit 1")
			}
		})
	}
}

// A classifier and a DETR detector import, load-check and install; the installed manifest is
// valid for the registry, pins the ONNX bytes, and every assumption is listed.
func TestImportInstalls(t *testing.T) {
	cleanEnv(t)
	dir := t.TempDir()
	reg := filepath.Join(dir, "reg")
	cls := filepath.Join(dir, "cls.onnx")
	clsFixture(t, cls)
	labels := filepath.Join(dir, "names.txt")
	if err := os.WriteFile(labels, []byte("cat\ndog\nbird\nfish\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	r := runImportReq(t, importRequest{Source: cls, Name: "my-cls", Task: "classification", License: "mit", Labels: labels, ModelsDir: reg})
	if r.Verdict != clireport.Warn || !strings.Contains(r.Reason, "imported my-cls") || !strings.Contains(r.Reason, "3 settings were assumed") {
		t.Fatalf("%s: %s %+v", r.Verdict, r.Reason, r.Findings)
	}
	man, err := registry.LoadManifest(filepath.Join(reg, "my-cls", "manifest.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	sum, _ := registry.FileSHA256(cls)
	if man.License != "MIT" || man.ArchOrName() != "efficientnet" || man.Input.Width != 8 || man.Labels != "labels.txt" {
		t.Errorf("manifest: %+v", man)
	}
	if d, ok := man.SHA256.For(""); !ok || d != sum {
		t.Errorf("sha256 pin %q, want %q", d, sum)
	}
	if err := man.VerifyWeights(); err != nil {
		t.Errorf("installed weights do not verify: %v", err)
	}
	if got, _ := man.LoadLabels(); len(got) != 4 {
		t.Errorf("labels %v", got)
	}
	d := r.Details.(*importDetails)
	if d.InstalledTo != filepath.Join(reg, "my-cls") || d.Card == nil || len(d.Assumed) != 3 {
		t.Errorf("details: installed %q, card %v, assumed %q", d.InstalledTo, d.Card != nil, d.Assumed)
	}
	if !strings.Contains(d.Manifest, "resize: squash   # assumed") {
		t.Errorf("assumed lines are not marked in the manifest:\n%s", d.Manifest)
	}

	// The same name again: refused without --force, replaced with it.
	if r := runImportReq(t, importRequest{Source: cls, Name: "my-cls", Task: "classification", License: "MIT", ModelsDir: reg}); r.Verdict != clireport.Fail {
		t.Errorf("re-import without --force: %s", r.Verdict)
	}
	if r := runImportReq(t, importRequest{Source: cls, Name: "my-cls", Task: "classification", License: "MIT", ModelsDir: reg, Force: true,
		Mean: "0.5,0.5,0.5", Std: "0.5,0.5,0.5", Resize: "squash", Labels: labels}); r.Verdict != clireport.Warn || len(r.Details.(*importDetails).Assumed) != 1 {
		t.Errorf("--force with explicit settings: %s %s %q", r.Verdict, r.Reason, r.Details.(*importDetails).Assumed)
	}

	// DETR: NHWC graph, dynamic size given with --input, labels count differs (a warning).
	detr := filepath.Join(dir, "det.onnx")
	detrFixture(t, detr, []int64{1, -1, -1, 3})
	r = runImportReq(t, importRequest{Source: detr, Name: "my-det", Task: "detection", License: "Apache-2.0", Input: "320x240", Labels: labels, ModelsDir: reg})
	if r.Verdict != clireport.Warn {
		t.Fatalf("detr: %s %s", r.Verdict, r.Reason)
	}
	man, err = registry.LoadManifest(filepath.Join(reg, "my-det", "manifest.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	spec, _ := man.PreprocessSpec()
	if man.ArchOrName() != "rf-detr" || spec.Width != 320 || spec.Height != 240 || spec.Layout != "NHWC" || man.Postprocess.ConfThreshold != 0.5 {
		t.Errorf("detr manifest: arch %s spec %+v", man.ArchOrName(), spec)
	}
	warned := false
	for _, f := range r.Findings {
		warned = warned || (f.Level == clireport.Warn && strings.Contains(f.Text, "has 4 names but the model outputs 91 class scores"))
	}
	if !warned {
		t.Errorf("no labels-count warning: %+v", r.Findings)
	}

	// --dry-run checks everything and installs nothing.
	r = runImportReq(t, importRequest{Source: cls, Name: "dry", Task: "classification", License: "MIT", ModelsDir: reg, DryRun: true})
	if r.Verdict == clireport.Fail || !strings.Contains(r.Reason, "dry run") {
		t.Errorf("dry run: %s %s", r.Verdict, r.Reason)
	}
	if _, err := os.Stat(filepath.Join(reg, "dry")); err == nil {
		t.Error("--dry-run installed the model")
	}
}

// External data travels with the model and is pinned; a model whose external data leaves its
// folder is refused.
func TestImportExternalData(t *testing.T) {
	cleanEnv(t)
	dir := t.TempDir()
	src := filepath.Join(dir, "big.onnx")
	onnxtest.WriteWith(t, src, onnxtest.Options{External: []onnxtest.ExternalInit{{Name: "w", Dims: []int64{4}, Location: "big.onnx.data"}}},
		[]onnxtest.Input{{Name: "x", Dims: []int64{1, 3, 8, 8}}}, onnxtest.Input{Name: "y", Dims: []int64{1, 4}})
	if err := os.WriteFile(filepath.Join(dir, "big.onnx.data"), make([]byte, 16), 0o644); err != nil {
		t.Fatal(err)
	}
	reg := filepath.Join(dir, "reg")
	r := runImportReq(t, importRequest{Source: src, Name: "big", Task: "classification", License: "MIT", ModelsDir: reg})
	if r.Verdict == clireport.Fail {
		t.Fatalf("%s", r.Reason)
	}
	if !fileExists(filepath.Join(reg, "big", "big.onnx.data")) {
		t.Error("external data not installed")
	}
	man, err := registry.LoadManifest(filepath.Join(reg, "big", "manifest.yaml"))
	if err != nil || man.SHA256Files["big.onnx.data"] == "" || man.VerifyWeights() != nil {
		t.Errorf("external data pin: %v %v", man.SHA256Files, err)
	}

	escape := filepath.Join(dir, "escape.onnx")
	onnxtest.WriteWith(t, escape, onnxtest.Options{External: []onnxtest.ExternalInit{{Name: "w", Dims: []int64{4}, Location: "../w.data"}}},
		[]onnxtest.Input{{Name: "x", Dims: []int64{1, 3, 8, 8}}}, onnxtest.Input{Name: "y", Dims: []int64{1, 4}})
	if r := runImportReq(t, importRequest{Source: escape, Name: "esc", Task: "classification", License: "MIT", ModelsDir: reg}); r.Verdict != clireport.Fail ||
		!strings.Contains(r.Reason, "outside its folder") {
		t.Errorf("escaping external data: %s %s", r.Verdict, r.Reason)
	}
	missing := filepath.Join(dir, "missing.onnx")
	onnxtest.WriteWith(t, missing, onnxtest.Options{External: []onnxtest.ExternalInit{{Name: "w", Dims: []int64{4}, Location: "nothere.data"}}},
		[]onnxtest.Input{{Name: "x", Dims: []int64{1, 3, 8, 8}}}, onnxtest.Input{Name: "y", Dims: []int64{1, 4}})
	if r := runImportReq(t, importRequest{Source: missing, Name: "mis", Task: "classification", License: "MIT", ModelsDir: reg}); r.Verdict != clireport.Fail ||
		!strings.Contains(r.Reason, "which is missing") {
		t.Errorf("missing external data: %s %s", r.Verdict, r.Reason)
	}
}

func TestParseImportFlags(t *testing.T) {
	for in, want := range map[string][2]int{"224x224": {224, 224}, "640X480": {640, 480}, "320×240": {320, 240}, "512": {512, 512}} {
		w, h, err := parseWxH(in)
		if err != nil || w != want[0] || h != want[1] {
			t.Errorf("parseWxH(%q) = %d,%d,%v", in, w, h, err)
		}
	}
	for _, bad := range []string{"", "x", "0x10", "-1x5", "99999x1", "axb"} {
		if _, _, err := parseWxH(bad); err == nil {
			t.Errorf("parseWxH(%q) accepted", bad)
		}
	}
	if v, err := parseTriple("--mean", "[0.5, 0.25,1]"); err != nil || v[1] != 0.25 {
		t.Errorf("parseTriple: %v %v", v, err)
	}
	for _, bad := range []string{"1,2", "1,2,NaN", "1,2,Inf", "a,b,c"} {
		if _, err := parseTriple("--mean", bad); err == nil {
			t.Errorf("parseTriple(%q) accepted", bad)
		}
	}
	if _, err := parseTriple("--std", "1,0,1"); err == nil {
		t.Error("a zero std must be refused")
	}
}
