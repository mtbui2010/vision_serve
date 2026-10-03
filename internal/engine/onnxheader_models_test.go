package engine

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestONNXHeaderMatchesORTOnModels checks the header reader against ORT's own probe on every
// real .onnx file in the model registry. Inputs must be identical (names, shapes, element
// types), and so must output names and element types, in order.
//
// Output shapes may differ in one way only. ORT runs ONNX shape inference when it loads a graph
// and merges the result into the declared output types, so a dimension the file declares
// symbolic can come back concrete: GroundingDINO's pred_boxes is declared
// [Gatherpred_boxes_dim_0, Gatherpred_boxes_dim_1, 4] and ORT reports [-1, 900, 4]. The reader
// reports the declaration ([-1, -1, 4]) — it would need ONNX shape inference to do otherwise.
// Allowed: same rank, every dimension equal or declared -1. Anything else fails.
//
// The weights are not committed, so it runs only where they were pulled; it walks
// $VISIONSERVE_ONNX_DIR if set, else the repository's models/. Each file costs a full ORT
// session load (seconds for the large graphs), hence the -short skip.
func TestONNXHeaderMatchesORTOnModels(t *testing.T) {
	if testing.Short() {
		t.Skip("loads every model in the registry into ONNX Runtime")
	}
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (libonnxruntime.so)")
	}
	dir := os.Getenv("VISIONSERVE_ONNX_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "models")
	}
	files := onnxFilesUnder(t, dir)
	if len(files) == 0 {
		t.Skipf("no .onnx files under %s (weights not pulled; set VISIONSERVE_ONNX_DIR)", dir)
	}
	if err := ensureORT(); err != nil {
		t.Skip(err)
	}

	var headerTotal, ortTotal time.Duration
	var compared, identical int
	for _, f := range files {
		name, _ := filepath.Rel(dir, f)
		t.Run(name, func(t *testing.T) {
			t0 := time.Now()
			hin, hout, err := readONNXHeader(f)
			dh := time.Since(t0)
			if err != nil {
				t.Fatalf("header reader: %v", err)
			}
			t0 = time.Now()
			oin, oout, err := inspectORT(f)
			do := time.Since(t0)
			if err != nil {
				t.Skipf("ORT cannot load it either: %v", err)
			}
			compared++
			headerTotal += dh
			ortTotal += do
			if !reflect.DeepEqual(hin, oin) {
				t.Errorf("inputs differ\nheader %+v\n   ORT %+v", hin, oin)
			}
			if reflect.DeepEqual(hout, oout) {
				if reflect.DeepEqual(hin, oin) {
					identical++
				}
			} else if len(hout) != len(oout) {
				t.Errorf("outputs differ\nheader %+v\n   ORT %+v", hout, oout)
			} else {
				for i := range hout {
					h, o := hout[i], oout[i]
					switch {
					case h.Name != o.Name || h.ElemType != o.ElemType || !refinedBy(h.Shape, o.Shape):
						t.Errorf("output %d differs: header %+v, ORT %+v", i, h, o)
					case !reflect.DeepEqual(h.Shape, o.Shape):
						t.Logf("output %q: declared %v, ORT's shape inference resolves %v", h.Name, h.Shape, o.Shape)
					}
				}
			}
			t.Logf("%d in, %d out; header %v, ORT %v", len(hin), len(hout), dh, do)
		})
	}
	t.Logf("%d files compared, %d identical in every field, the rest differ only in output dims ORT's "+
		"shape inference resolved; header reader %v total, ORT GetInputOutputInfo %v total",
		compared, identical, headerTotal, ortTotal)
}

// refinedBy reports whether ORT's output shape o is the declared shape h, possibly with some
// declared-unknown (-1) dimensions resolved by ORT's load-time shape inference.
func refinedBy(h, o []int64) bool {
	if len(h) != len(o) {
		return false
	}
	for i := range h {
		if h[i] != o[i] && h[i] != -1 {
			return false
		}
	}
	return true
}

// onnxFilesUnder lists the .onnx files below dir, following symlinks (each real file once) and
// leaving out files over 2 GB.
func onnxFilesUnder(t *testing.T, dir string) []string {
	t.Helper()
	seen := map[string]bool{}
	var files []string
	var walk func(d string)
	walk = func(d string) {
		real, err := filepath.EvalSymlinks(d)
		if err != nil || seen[real] {
			return
		}
		seen[real] = true
		entries, err := os.ReadDir(d)
		if err != nil {
			return
		}
		for _, e := range entries {
			p := filepath.Join(d, e.Name())
			st, err := os.Stat(p) // follows symlinks
			if err != nil {
				continue
			}
			if st.IsDir() {
				walk(p)
				continue
			}
			if !strings.HasSuffix(e.Name(), ".onnx") || st.Size() > 2<<30 {
				continue
			}
			if real, err := filepath.EvalSymlinks(p); err == nil && !seen[real] {
				seen[real] = true
				files = append(files, p)
			}
		}
	}
	walk(dir)
	sort.Strings(files)
	return files
}
