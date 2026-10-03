package engine

import (
	"context"
	"errors"
	"os"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// NewSessionWith must hand its options to the session creation on the worker thread, and
// NewSession must keep ORT's defaults (zero options).
func TestSessionOptionsReachCreation(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (NewSession initializes ONNX Runtime first)")
	}
	if err := ensureORT(); err != nil {
		t.Skip(err)
	}
	stop := errors.New("captured")
	var got []SessionOptions
	prev := createORTSession
	createORTSession = func(_ string, _, _ []string, _ []Provider, so SessionOptions) (*ort.DynamicAdvancedSession, Provider, error) {
		got = append(got, so)
		return nil, ProviderCPU, stop
	}
	defer func() { createORTSession = prev }()

	in, out := []string{"x"}, []string{"y"}
	if _, err := NewSessionWith("unused.onnx", in, out, []Provider{ProviderCPU}, SessionOptions{IntraOpThreads: 3}); !errors.Is(err, stop) {
		t.Fatalf("NewSessionWith: err = %v, want the creation error", err)
	}
	if _, err := NewSession("unused.onnx", in, out, []Provider{ProviderCPU}); !errors.Is(err, stop) {
		t.Fatalf("NewSession: err = %v, want the creation error", err)
	}
	if len(got) != 2 || got[0].IntraOpThreads != 3 || got[1] != (SessionOptions{}) {
		t.Fatalf("creation saw options %+v, want [{3} {0}]", got)
	}
}

func TestNegativeIntraOpThreadsRejected(t *testing.T) {
	if _, err := NewSessionWith("unused.onnx", nil, nil, []Provider{ProviderCPU}, SessionOptions{IntraOpThreads: -1}); err == nil {
		t.Fatal("IntraOpThreads -1 accepted")
	}
}

// A real session with a capped intra-op pool loads and runs.
func TestRealSessionWithIntraOpThreads(t *testing.T) {
	path := identityModel(t)
	s, err := NewSessionWith(path, []string{"x"}, []string{"y"}, []Provider{ProviderCPU}, SessionOptions{IntraOpThreads: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	outs, err := s.Run(context.Background(), []Tensor{F32([]float32{4, 5, 6}, 1, 3)})
	if err != nil {
		t.Fatal(err)
	}
	if got := outs[0].Data; len(got) != 3 || got[0] != 4 || got[2] != 6 {
		t.Fatalf("identity output %v", got)
	}
}
