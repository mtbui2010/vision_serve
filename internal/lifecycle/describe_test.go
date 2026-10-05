package lifecycle

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/engine/onnxtest"
)

// Describe reports what Load decides, without opening a session: the same refusal (an input
// size the graph cannot take), the compared shapes, and per role the sessions Load would open.
func TestDescribeMatchesLoad(t *testing.T) {
	t.Setenv("VS_POOL_OVERRIDE", "")
	t.Setenv("VISIONSERVE_EP", "")
	t.Setenv("VISIONSERVE_TENSORRT", "")
	root := t.TempDir()
	in := onnxtest.Input{Name: "input", Dims: []int64{-1, 3, 560, 560}}
	writeGraphModel(t, root, "fits", "rf-detr", sized(560, 560), in)
	writeGraphModel(t, root, "too-big", "rf-detr", sized(640, 640), in)
	writeTestModel(t, root, "pipe", "test-pipe", "  threads:\n    x: 1\n", "x", "y")
	writeSingleModel(t, root, "no-weights", "rf-detr", sized(560, 560))
	if err := os.Remove(filepath.Join(root, "no-weights", "m.onnx")); err != nil {
		t.Fatal(err)
	}
	m, op := newFakeManager(t, scanRegistry(t, root))

	fits := m.Describe("fits")
	if fits.BuildErr != nil || fits.Model == nil || fits.Manifest == nil {
		t.Fatalf("fits: %+v", fits)
	}
	if !reflect.DeepEqual(fits.Input.Graph, []int64{-1, 3, 560, 560}) || !reflect.DeepEqual(fits.Input.Produced, []int64{1, 3, 560, 560}) ||
		fits.Input.Input != "input" || fits.Input.Skipped != "" {
		t.Errorf("fits input: %+v", fits.Input)
	}
	want := []SessionPlan{{Role: "model", Path: filepath.Join(root, "fits", "m.onnx"), Pool: 1, ThreadsFrom: ThreadsORTDefault}}
	if !reflect.DeepEqual(fits.Sessions, want) {
		t.Errorf("fits sessions = %+v, want %+v", fits.Sessions, want)
	}
	if !reflect.DeepEqual(fits.Providers, []engine.Provider{engine.ProviderCPU}) {
		t.Errorf("providers = %v", fits.Providers)
	}

	big := m.Describe("too-big")
	var shapeErr *InputShapeError
	if !errors.As(big.BuildErr, &shapeErr) || big.Input.Err != big.BuildErr {
		t.Fatalf("too-big: BuildErr = %v", big.BuildErr)
	}
	// The same error Load returns.
	if err := m.Load(context.Background(), "too-big"); err == nil || err.Error() != big.BuildErr.Error() {
		t.Errorf("Load = %v, Describe = %v", err, big.BuildErr)
	}

	pipe := m.Describe("pipe")
	wantPipe := []SessionPlan{
		{Role: "x", Path: filepath.Join(root, "pipe", "x.onnx"), Pool: 1, Threads: 1, ThreadsFrom: ThreadsManifest},
		{Role: "y", Path: filepath.Join(root, "pipe", "y.onnx"), Pool: 1, ThreadsFrom: ThreadsORTDefault},
	}
	if pipe.BuildErr != nil || !reflect.DeepEqual(pipe.Sessions, wantPipe) {
		t.Errorf("pipe: err %v sessions %+v, want %+v", pipe.BuildErr, pipe.Sessions, wantPipe)
	}
	if pipe.Input.Skipped == "" {
		t.Error("a pipeline without an explain role must say why its input was not judged")
	}

	nw := m.Describe("no-weights")
	if !errors.Is(nw.BuildErr, ErrModelNotFound) || nw.Manifest == nil || nw.Model != nil {
		t.Errorf("no-weights: %+v", nw)
	}
	if un := m.Describe("unknown"); un.Manifest != nil || !errors.Is(un.BuildErr, ErrModelNotFound) {
		t.Errorf("unknown: %+v", un)
	}
	if n := len(op.engines()); n != 0 {
		t.Errorf("Describe opened %d sessions, want none", n)
	}
}

func TestSessionThreads(t *testing.T) {
	t.Setenv("VISIONSERVE_POOL_THREADS", "")
	if th, from := sessionThreads(1, 0, false); th != 0 || from != ThreadsORTDefault {
		t.Errorf("lone session: %d %s", th, from)
	}
	if _, from := sessionThreads(4, 0, false); from != ThreadsPoolRule {
		t.Errorf("pool: %s", from)
	}
	t.Setenv("VISIONSERVE_POOL_THREADS", "1")
	if th, from := sessionThreads(4, 0, false); th != 1 || from != ThreadsPoolEnv {
		t.Errorf("pool env: %d %s", th, from)
	}
	t.Setenv("VISIONSERVE_POOL_THREADS", "junk")
	if _, from := sessionThreads(4, 0, false); from != ThreadsPoolRule {
		t.Errorf("bad pool env falls back to the rule: %s", from)
	}
	if th, from := sessionThreads(4, 1, true); th != 1 || from != ThreadsManifest {
		t.Errorf("manifest: %d %s", th, from)
	}
}
