package engine

import (
	"context"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// A panic inside a job used to unwind the worker goroutine and kill the whole server — the worker
// is not a request goroutine, so net/http's per-request recovery never sees it. It must come back
// as an error, and the worker must keep serving the session.
func TestJobPanicBecomesErrorAndWorkerSurvives(t *testing.T) {
	s := &Session{jobs: make(chan func()), closeErr: make(chan error, 1)}
	go func() { // the worker's job loop, without ORT
		for fn := range s.jobs {
			fn()
		}
		s.closeErr <- nil
	}()
	defer s.Close()

	_, err := s.submit(context.Background(), func() ([]Tensor, error) { panic("boom") })
	if !errors.Is(err, ErrInferencePanic) {
		t.Fatalf("panicking job: err = %v, want ErrInferencePanic", err)
	}
	outs, err := s.submit(context.Background(), func() ([]Tensor, error) { return []Tensor{F32([]float32{7}, 1)}, nil })
	if err != nil || len(outs) != 1 || outs[0].Data[0] != 7 {
		t.Fatalf("job after a panic: outs=%v err=%v — the worker did not survive", outs, err)
	}
}

func identityModel(t *testing.T) string {
	t.Helper()
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (libonnxruntime.so)")
	}
	raw, err := hex.DecodeString(identityONNX)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "identity.onnx")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// Decision: a recovered panic leaves the session USABLE, not broken. The panic is in Go code around
// the ORT call (cgo cannot unwind a Go panic through C), ORT's own failures arrive as error
// statuses, and the deferred Destroy of the job's tensors runs during the unwind — so the native
// session is in the state it was before the job.
func TestRealSessionUsableAfterJobPanic(t *testing.T) {
	path := identityModel(t)
	s, err := NewSession(path, []string{"x"}, []string{"y"}, []Provider{ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	_, err = s.submit(context.Background(), func() ([]Tensor, error) {
		in, _ := ort.NewTensor(ort.NewShape(1, 3), []float32{1, 2, 3})
		defer in.Destroy()
		var nilMap map[string]int
		//lint:ignore SA5000 deliberate: the test needs a real runtime panic, not a panic(string)
		nilMap["x"] = 1
		return nil, nil
	})
	if !errors.Is(err, ErrInferencePanic) {
		t.Fatalf("err = %v, want ErrInferencePanic", err)
	}
	outs, err := s.Run(context.Background(), []Tensor{F32([]float32{1, 2, 3}, 1, 3)})
	if err != nil {
		t.Fatalf("Run after a recovered panic: %v", err)
	}
	if got := outs[0].Data; len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Fatalf("identity output %v after a recovered panic", got)
	}
}

// A panic while CREATING the session (on the worker thread) must fail NewSession, not the process.
func TestPanicDuringSessionCreationIsAnError(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (NewSession initializes ONNX Runtime first)")
	}
	if err := ensureORT(); err != nil {
		t.Skip(err)
	}
	prev := createORTSession
	createORTSession = func(string, []string, []string, []Provider, SessionOptions) (*ort.DynamicAdvancedSession, Provider, error) {
		panic("simulated crash in session creation")
	}
	defer func() { createORTSession = prev }()
	if _, err := NewSession("unused.onnx", []string{"x"}, []string{"y"}, []Provider{ProviderCPU}); !errors.Is(err, ErrInferencePanic) {
		t.Fatalf("NewSession with a panicking creation: err = %v, want ErrInferencePanic", err)
	}
}
