package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/engine"
)

// clearTensorRT isolates a test from the caller's EP environment and the process-wide flag.
func clearTensorRT(t *testing.T) {
	t.Helper()
	t.Setenv("VISIONSERVE_TENSORRT", "")
	t.Setenv("VISIONSERVE_EP", "")
	t.Setenv("VISIONSERVE_VERIFY", "")
	engine.SetTensorRT(false)
	t.Cleanup(func() { engine.SetTensorRT(false) })
}

// notADir returns a --models path that makes the registry scan fail, so runServe returns right
// after applying its flags instead of starting a server.
func notADir(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestServeTensorRTFlagSetsEngine(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want bool
	}{
		{nil, false},
		{[]string{"--tensorrt"}, true},
		{[]string{"--tensorrt=false"}, false},
	} {
		clearTensorRT(t)
		args := append([]string{"--models", notADir(t)}, tc.args...)
		if err := runServe(args); err == nil {
			t.Fatalf("runServe(%v): want the registry error, got nil", args)
		}
		if got := engine.TensorRTRequested(); got != tc.want {
			t.Errorf("serve %v: TensorRTRequested() = %v, want %v", tc.args, got, tc.want)
		}
	}
}

func TestRunTensorRTFlagSetsEngine(t *testing.T) {
	clearTensorRT(t)
	// No positionals: runRun parses flags, applies them, then fails on usage.
	if err := runRun([]string{"--tensorrt"}); err == nil {
		t.Fatal("runRun(--tensorrt): want a usage error, got nil")
	}
	if !engine.TensorRTRequested() {
		t.Error("run --tensorrt did not turn the TensorRT opt-in on")
	}
}

// TestTensorRTFlagChangesTheChain: end to end through the CLI helpers, the shipped [cuda, cpu]
// chain becomes [tensorrt, cuda, cpu] with the flag and stays [cuda, cpu] without it.
func TestTensorRTFlagChangesTheChain(t *testing.T) {
	clearTensorRT(t)
	chain := func() []engine.Provider {
		t.Helper()
		got, err := engine.ResolveProviders([]string{"cuda", "cpu"})
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	if got := chain(); len(got) != 2 || got[0] != engine.ProviderCUDA {
		t.Errorf("default chain = %v, want [cuda cpu]", got)
	}
	on := true
	applyTensorRTFlag(&on)
	if got := chain(); len(got) != 3 || got[0] != engine.ProviderTensorRT || got[1] != engine.ProviderCUDA {
		t.Errorf("--tensorrt chain = %v, want [tensorrt cuda cpu]", got)
	}
}

func TestEPStatus(t *testing.T) {
	clearTensorRT(t)
	if s := strings.Join(epStatus(), "\n"); !strings.Contains(s, "CUDA → CPU (default)") ||
		!strings.Contains(s, "--tensorrt") || !strings.Contains(s, "VISIONSERVE_TENSORRT=1") {
		t.Errorf("default epStatus:\n%s", s)
	}

	engine.SetTensorRT(true)
	s := strings.Join(epStatus(), "\n")
	if engine.TRTAvailable() {
		if !strings.Contains(s, "TensorRT → CUDA → CPU") || !strings.Contains(s, "6.8") {
			t.Errorf("epStatus with --tensorrt and libnvinfer:\n%s", s)
		}
	} else if !strings.Contains(s, "requested") || !strings.Contains(s, "falling back to CUDA") {
		t.Errorf("epStatus with --tensorrt, no libnvinfer:\n%s", s)
	}

	t.Setenv("VISIONSERVE_EP", "cpu")
	if s := strings.Join(epStatus(), "\n"); !strings.Contains(s, "VISIONSERVE_EP=cpu replaces") {
		t.Errorf("epStatus under VISIONSERVE_EP:\n%s", s)
	}
}
