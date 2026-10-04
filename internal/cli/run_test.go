package cli

import (
	"encoding/base64"
	"encoding/binary"
	"flag"
	"image"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"visionserve/pkg/api"
)

// parseRunFlags parses args with the request-option flags of `visionserve run`.
func parseRunFlags(t *testing.T, args ...string) *runOptions {
	t.Helper()
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var o runOptions
	addRequestFlags(fs, &o)
	if err := fs.Parse(args); err != nil {
		t.Fatalf("parse %v: %v", args, err)
	}
	return &o
}

// Every option of POST /api/predict is reachable from `run`, as a flag named after its form field
// (dashes for underscores), or through the stand-in listed here.
func TestRunHasAFlagForEveryRequestOption(t *testing.T) {
	standIn := map[string]string{
		"model":         "positional argument",
		"image_base64":  "positional argument (the image file)",
		"depth_base64":  "--depth FILE",
		"template_name": "--template FILE (no server-side store to name in-process)",
		"encoding":      "run prints the result itself; JSON numbers only",
	}
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	addRequestFlags(fs, &runOptions{})
	rt := reflect.TypeOf(api.PredictJSONRequest{})
	for i := 0; i < rt.NumField(); i++ {
		name, _, _ := strings.Cut(rt.Field(i).Tag.Get("json"), ",")
		if _, ok := standIn[name]; ok {
			continue
		}
		if fs.Lookup(strings.ReplaceAll(name, "_", "-")) == nil {
			t.Errorf("request option %q has no `run` flag --%s", name, strings.ReplaceAll(name, "_", "-"))
		}
	}
	for _, f := range []string{"depth", "template"} {
		if fs.Lookup(f) == nil {
			t.Errorf("missing stand-in flag --%s", f)
		}
	}
}

func TestRunFlagsBecomeRequestFields(t *testing.T) {
	o := parseRunFlags(t,
		"--prompt", "cat.", "--box", "1,2,3,4", "--point", "5,6,1",
		"--box-threshold", "0.35", "--text-threshold", "0.2",
		"--min-size", "0.1", "--max-size", "90", "--roi", "0.1,0.1,0.5,0.5",
		"--method", "dual", "--bg-max-area", "40", "--fg-min-area", "0.5",
		"--grid-size", "12", "--dilate", "-3",
		"--gripper-min", "10", "--gripper-max", "80",
		"--claim-threshold", "0.05", "--crop-temp", "0.02",
	)
	req, store, err := o.request("m")
	if err != nil {
		t.Fatal(err)
	}
	if store != nil {
		t.Errorf("no --template: want no template store, got one")
	}
	want := api.PredictJSONRequest{
		Model: "m", Prompt: "cat.", Box: "1,2,3,4", Point: "5,6,1",
		BoxThreshold: 0.35, TextThreshold: 0.2, MinSize: 0.1, MaxSize: 90, ROI: "0.1,0.1,0.5,0.5",
		Method: "dual", BgMaxArea: 40, FgMinArea: 0.5, GridSize: 12, Dilate: -3,
		GripperMin: 10, GripperMax: 80, ClaimThreshold: 0.05, CropTemp: 0.02,
	}
	if !reflect.DeepEqual(req.PredictJSONRequest, want) {
		t.Errorf("request\n got %+v\nwant %+v", req.PredictJSONRequest, want)
	}
	// ...and reach the model input the same way as a server request.
	p, err := req.ToPrompt(100, 100)
	if err != nil {
		t.Fatal(err)
	}
	if p.BoxThresh != 0.35 || p.TextThresh != 0.2 || p.GripperMin != 10 || p.GripperMax != 80 ||
		p.ClaimThresh != 0.05 || p.CropTemp != 0.02 || p.Method != "dual" || p.Dilate != -3 {
		t.Errorf("prompt does not carry the flags: %+v", p)
	}
}

func writeFile(t *testing.T, name string, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRunDepthFile(t *testing.T) {
	// uint16, 3x2, at an explicit size; 0 = no reading.
	u16 := []uint16{0, 65535, 32768, 1, 2, 3}
	raw := make([]byte, 2*len(u16))
	for i, v := range u16 {
		binary.LittleEndian.PutUint16(raw[2*i:], v)
	}
	o := parseRunFlags(t, "--depth", writeFile(t, "d.raw", raw), "--depth-width", "3", "--depth-height", "2")
	req, _, err := o.request("background")
	if err != nil {
		t.Fatal(err)
	}
	if req.DepthBase64 != base64.StdEncoding.EncodeToString(raw) {
		t.Fatalf("depth_base64 is not the file's bytes")
	}
	p, err := req.ToPrompt(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if p.DepthW != 3 || p.DepthH != 2 || len(p.Depth) != 6 {
		t.Fatalf("depth %dx%d len %d, want 3x2 len 6", p.DepthW, p.DepthH, len(p.Depth))
	}
	if !math.IsNaN(float64(p.Depth[0])) || p.Depth[1] != 1 {
		t.Errorf("depth values %v: want NaN (no reading) then 1", p.Depth[:2])
	}

	// float32, no size: the image's size.
	f32 := make([]byte, 4*4)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(f32[4*i:], math.Float32bits(float32(i+1)))
	}
	o = parseRunFlags(t, "--depth", writeFile(t, "d.f32", f32), "--depth-dtype", "float32")
	req, _, err = o.request("background")
	if err != nil {
		t.Fatal(err)
	}
	p, err = req.ToPrompt(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if p.DepthW != 2 || p.DepthH != 2 || p.Depth[3] != 4 {
		t.Errorf("float32 depth %dx%d %v, want 2x2 ending in 4", p.DepthW, p.DepthH, p.Depth)
	}

	// A file whose size does not match is the server's own 400-class error.
	o = parseRunFlags(t, "--depth", writeFile(t, "short.raw", raw[:5]))
	req, _, err = o.request("background")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := req.ToPrompt(3, 2); err == nil || !strings.Contains(err.Error(), "invalid depth map") {
		t.Errorf("mismatched depth file: got %v, want invalid depth map", err)
	}
}

func TestRunRequestFlagMistakes(t *testing.T) {
	raw := writeFile(t, "d.raw", make([]byte, 8))
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"--depth", raw, "--depth-dtype", "uint8"}, "--depth-dtype must be uint16 or float32"},
		{[]string{"--depth", raw, "--depth-width", "2"}, "go together"},
		{[]string{"--depth", raw, "--depth-width", "-2", "--depth-height", "-2"}, "go together"},
		{[]string{"--depth-width", "2", "--depth-height", "2"}, "need --depth"},
		{[]string{"--depth-dtype", "float32"}, "need --depth"},
		{[]string{"--depth", filepath.Join(t.TempDir(), "missing.raw")}, "--depth: open"},
		{[]string{"--box-threshold", "NaN"}, "--box-threshold must be a finite number"},
		{[]string{"--crop-temp", "+Inf"}, "--crop-temp must be a finite number"},
		{[]string{"--template", filepath.Join(t.TempDir(), "missing.png")}, "--template: failed to open image"},
	} {
		_, _, err := parseRunFlags(t, tc.args...).request("m")
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%v: got %v, want an error containing %q", tc.args, err, tc.want)
		}
	}
}

func TestRunTemplatesAreRegisteredInProcess(t *testing.T) {
	var paths []string
	for _, name := range []string{"a.png", "b.png"} {
		p := filepath.Join(t.TempDir(), name)
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, image.NewRGBA(image.Rect(0, 0, 4, 3))); err != nil {
			t.Fatal(err)
		}
		f.Close()
		paths = append(paths, p)
	}
	req, store, err := parseRunFlags(t, "--template", paths[0], "--template", paths[1]).request("owlvit-base")
	if err != nil {
		t.Fatal(err)
	}
	if req.TemplateName != runTemplateSet {
		t.Errorf("template_name = %q, want %q", req.TemplateName, runTemplateSet)
	}
	if store == nil || len(store.Get(runTemplateSet)) != 2 {
		t.Fatalf("store does not hold the two templates under %q", runTemplateSet)
	}
	if b := store.Get(runTemplateSet)[0].Bounds(); b.Dx() != 4 || b.Dy() != 3 {
		t.Errorf("template size %v, want 4x3", b)
	}
}
