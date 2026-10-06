package server

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"visionserve/internal/registry"
	"visionserve/internal/templates"

	_ "visionserve/internal/models/background" // background registers accepts_depth
	_ "visionserve/internal/models/detr"       // rf-detr / rt-detr register their useful side
)

// GET /api/models carries the client-resize hint: the bounded side follows the preprocessing
// mode (squash → shorter side, letterbox → longer side), an unregistered architecture gets none,
// and runtime.max_useful_side overrides both ways (N = longer side N, 0 = never). Both keys are
// always present (null = send full resolution), so a client can tell "no hint" from an old server.
func TestModelsListsUsefulSide(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		d := filepath.Join(dir, name)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		y := "name: " + name + "\nlicense: Apache-2.0\n" + body
		if err := os.WriteFile(filepath.Join(d, "manifest.yaml"), []byte(y), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("squash", "task: detection\narchitecture: rf-detr\nmodel_file: m.onnx\ninput: {width: 560, height: 560}\n")
	write("letterbox", "task: detection\narchitecture: rt-detr\nmodel_file: m.onnx\ninput: {width: 640, height: 480, letterbox: true}\n")
	write("block", "task: detection\narchitecture: rf-detr\nmodel_file: m.onnx\npreprocess: {resize: letterbox, size: 384}\n")
	write("masks", "task: segmentation\narchitecture: mobile-sam\nfiles: {encoder: e.onnx, decoder: d.onnx}\ninput: {width: 1024, height: 1024}\n")
	write("never", "task: detection\narchitecture: rf-detr\nmodel_file: m.onnx\ninput: {width: 560, height: 560}\nruntime: {max_useful_side: 0}\n")
	write("override", "task: segmentation\narchitecture: mobile-sam\nfiles: {encoder: e.onnx, decoder: d.onnx}\ninput: {width: 1024, height: 1024}\nruntime: {max_useful_side: 3000}\n")
	reg := registry.New(dir)
	if _, err := reg.Scan(); err != nil {
		t.Fatal(err)
	}
	s := newServer(reg, &fakeRuntime{}, templates.New(), "")
	rec := do(s.http.Handler, httptest.NewRequest("GET", "/api/models", nil))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var raw []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	type hint struct{ long, short any }
	want := map[string]hint{
		"squash":    {nil, float64(1120)},
		"letterbox": {float64(1280), nil},
		"block":     {float64(768), nil},
		"masks":     {nil, nil},
		"never":     {nil, nil},
		"override":  {float64(3000), nil},
	}
	if len(raw) != len(want) {
		t.Fatalf("got %d models, want %d: %s", len(raw), len(want), rec.Body)
	}
	for _, m := range raw {
		name, _ := m["name"].(string)
		for _, key := range []string{"max_useful_side", "max_useful_short_side"} {
			if _, ok := m[key]; !ok {
				t.Errorf("%s: key %q missing (it must be present, null for no hint)", name, key)
			}
		}
		w := want[name]
		if m["max_useful_side"] != w.long || m["max_useful_short_side"] != w.short {
			t.Errorf("%s: max_useful_side=%v max_useful_short_side=%v, want %v / %v",
				name, m["max_useful_side"], m["max_useful_short_side"], w.long, w.short)
		}
	}
}

// GET /api/models says which models read an uploaded depth map (accepts_depth): `background`
// with a MiDaS session does; one without it (sam/cv/automask only) never reaches the plane fit,
// and a detector never reads depth. The key is always present, false included.
func TestModelsListsAcceptsDepth(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		d := filepath.Join(dir, name)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		y := "name: " + name + "\nlicense: Apache-2.0\n" + body
		if err := os.WriteFile(filepath.Join(d, "manifest.yaml"), []byte(y), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("bg-midas", "task: segmentation\narchitecture: background\nfiles: {depth: d.onnx, encoder: e.onnx, decoder: m.onnx}\ninput: {width: 1024, height: 1024}\n")
	write("bg-sam", "task: segmentation\narchitecture: background\nfiles: {encoder: e.onnx, decoder: m.onnx}\ninput: {width: 1024, height: 1024}\n")
	write("det", "task: detection\narchitecture: rf-detr\nmodel_file: m.onnx\ninput: {width: 560, height: 560}\n")
	reg := registry.New(dir)
	if _, err := reg.Scan(); err != nil {
		t.Fatal(err)
	}
	s := newServer(reg, &fakeRuntime{}, templates.New(), "")
	rec := do(s.http.Handler, httptest.NewRequest("GET", "/api/models", nil))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var raw []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{"bg-midas": true, "bg-sam": false, "det": false}
	if len(raw) != len(want) {
		t.Fatalf("got %d models, want %d: %s", len(raw), len(want), rec.Body)
	}
	for _, m := range raw {
		name, _ := m["name"].(string)
		v, ok := m["accepts_depth"]
		if !ok {
			t.Errorf("%s: key accepts_depth missing (it must be present, false included)", name)
			continue
		}
		if v != want[name] {
			t.Errorf("%s: accepts_depth=%v, want %v", name, v, want[name])
		}
		if want[name] && (m["max_useful_side"] != nil || m["max_useful_short_side"] != nil) {
			t.Errorf("%s accepts depth but has a client-resize hint: %v", name, m)
		}
	}
}
