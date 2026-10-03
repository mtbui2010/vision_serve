package engine

import (
	"context"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// identityONNX is a 1x3 float Identity graph (opset 13), embedded so the test needs no fixture
// file (*.onnx is gitignored).
const identityONNX = "08083a3f0a100a017812017922084964656e746974791201675a130a0178120e0a0c080112080a0208010a02080362130a0179120e0a0c080112080a0208010a02080342040a00100d"

// An EP this ORT build does not ship must not be recorded as the active EP. With the CPU-only
// wheel (or the CPU Docker image) appending CUDA fails, the session is still created — on CPU —
// and the old code, which ignored the append error, reported device "gpu:0" for CPU work.
func TestUnavailableEPIsNotReportedAsActive(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (libonnxruntime.so)")
	}
	if err := ensureORT(); err != nil {
		t.Skip(err)
	}
	raw, err := hex.DecodeString(identityONNX)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "identity.onnx")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	// Does THIS ORT build accept the CUDA EP at all?
	opts, err := ort.NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	cudaErr := applyProvider(opts, ProviderCUDA, path)
	opts.Destroy()

	s, err := NewSession(path, []string{"x"}, []string{"y"}, []Provider{ProviderCUDA, ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if cudaErr != nil && s.ActiveEP() != ProviderCPU {
		t.Fatalf("this ORT build has no CUDA EP (%v), yet the session reports %s — device would say %q",
			cudaErr, providerNames([]Provider{s.ActiveEP()}), DeviceString(s.ActiveEP()))
	}
	t.Logf("CUDA append: %v; active EP: %s", cudaErr, providerNames([]Provider{s.ActiveEP()}))
}

// A call after Close must return ErrClosed — never panic ("send on closed channel") or hang.
func TestRunAfterCloseReturnsErrClosed(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (libonnxruntime.so)")
	}
	raw, _ := hex.DecodeString(identityONNX)
	path := filepath.Join(t.TempDir(), "identity.onnx")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	in := []Tensor{F32([]float32{1, 2, 3}, 1, 3)}
	s, err := NewSession(path, []string{"x"}, []string{"y"}, []Provider{ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(context.Background(), in); err != nil {
		t.Fatal(err)
	}
	_ = s.Close()
	if _, err := s.Run(context.Background(), in); !errors.Is(err, ErrClosed) {
		t.Fatalf("Run after Close: %v, want ErrClosed", err)
	}

	s2, err := NewSession(path, []string{"x"}, []string{"y"}, []Provider{ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	pool := NewSessionPool([]*Session{s2})
	_ = pool.Close()
	done := make(chan error, 1)
	go func() { _, err := pool.Run(context.Background(), in); done <- err }()
	select {
	case err := <-done:
		if !errors.Is(err, ErrClosed) {
			t.Fatalf("pool.Run after Close: %v, want ErrClosed", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("pool.Run after Close hung")
	}
}
