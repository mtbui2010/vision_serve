package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeWeights(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// The bug this exists to prevent: ORT derives its engine-cache filename from graph properties,
// and those collide between two fine-tunes of the SAME architecture. With one shared cache
// directory, a cache built from a 17-class RF-DETR made the 22-class sibling load in 14 s and
// then serve the 17-class model's class head. Under `exact`/`folded` that would be SILENT —
// those paths read only tensors whose shapes match across checkpoints.
func TestEngineKeyDistinguishesDifferentWeights(t *testing.T) {
	dir := t.TempDir()
	a := writeWeights(t, dir, "model-17class.onnx", "seventeen")
	b := writeWeights(t, dir, "model-22class.onnx", "twenty-two-different-length")

	if engineKey(a) == engineKey(b) {
		t.Fatal("two different weights files share an engine-cache key — one would be served " +
			"the other's compiled engine")
	}
}

// Two registry entries very often point at ONE file (several models here are symlinks into the
// NAS). Those should share an engine: rebuilding costs minutes, and the graph is identical.
func TestEngineKeyFollowsSymlinksToOneEngine(t *testing.T) {
	dir := t.TempDir()
	real := writeWeights(t, dir, "real.onnx", "weights")
	link := filepath.Join(dir, "alias.onnx")
	if err := os.Symlink(real, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if engineKey(real) != engineKey(link) {
		t.Error("a symlink and its target produced different keys — the same engine would be " +
			"compiled twice")
	}
}

// A re-export that overwrites a path IN PLACE is a different graph at the same name. Size or
// mtime must move the key, or the stale engine is served for the new weights — the same silent
// wrong answer, arriving later.
func TestEngineKeyChangesWhenWeightsAreRewritten(t *testing.T) {
	dir := t.TempDir()
	p := writeWeights(t, dir, "model.onnx", "first export")
	before := engineKey(p)

	time.Sleep(10 * time.Millisecond) // ensure a distinct mtime even at coarse resolution
	writeWeights(t, dir, "model.onnx", "second export, different length entirely")

	if engineKey(p) == before {
		t.Error("rewriting the weights in place left the key unchanged — the stale engine would " +
			"be reused for a different graph")
	}
}

func TestTRTOptionsNamespacesEnginesButSharesTiming(t *testing.T) {
	cache := t.TempDir()
	t.Setenv("VISIONSERVE_TRT_CACHE", cache)
	dir := t.TempDir()
	a := writeWeights(t, dir, "a.onnx", "aaa")
	b := writeWeights(t, dir, "b.onnx", "bbbb")

	oa, ob := TRTOptions(a), TRTOptions(b)
	if len(oa) == 0 || len(ob) == 0 {
		t.Fatal("expected options with a writable cache dir")
	}
	if oa["trt_engine_cache_path"] == ob["trt_engine_cache_path"] {
		t.Error("engine cache paths must differ per weights file")
	}
	// The timing cache holds GPU kernel-autotuning measurements, is graph-independent, and is
	// what makes the FIRST build of an unseen model faster. Namespacing it would be a
	// self-inflicted slowdown.
	if oa["trt_timing_cache_path"] != ob["trt_timing_cache_path"] {
		t.Error("the timing cache is graph-independent and must stay shared")
	}
	for _, o := range []map[string]string{oa, ob} {
		if !strings.HasPrefix(o["trt_engine_cache_path"], cache) {
			t.Errorf("engine cache %q escaped the configured root %q", o["trt_engine_cache_path"], cache)
		}
		if _, err := os.Stat(o["trt_engine_cache_path"]); err != nil {
			t.Errorf("engine cache dir was not created: %v", err)
		}
	}
}

// A read-only or HOME-less host must degrade to uncached TensorRT, not fail the load.
func TestTRTOptionsNilWhenCacheUnresolvable(t *testing.T) {
	t.Setenv("VISIONSERVE_TRT_CACHE", "")
	t.Setenv("HOME", "")
	if runtimeHomeResolvable() {
		t.Skip("HOME still resolvable on this platform")
	}
	if o := TRTOptions("whatever.onnx"); o != nil {
		t.Errorf("expected nil options with no resolvable cache dir, got %v", o)
	}
}

func runtimeHomeResolvable() bool {
	_, err := os.UserHomeDir()
	return err == nil
}
