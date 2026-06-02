// Package engine wraps ONNX Runtime (binding github.com/yalue/onnxruntime_go).
//
// Principle (CLAUDE.md): do NOT write your own inference engine. All inference goes through ORT.
// Why ORT: the same .onnx file runs on both GPU (TensorRT/CUDA EP) and CPU
// — matching the edge↔server goal.
//
// The binding requires the libonnxruntime.so shared library at runtime. The path comes from
// the ORT_DYLIB_PATH environment variable (e.g. /usr/local/lib/libonnxruntime.so). On Jetson
// use an ORT build with TensorRT/CUDA EP.
package engine

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	initOnce sync.Once
	initErr  error
)

// ensureORT initializes the ORT environment exactly once per process.
func ensureORT() error {
	initOnce.Do(func() {
		if path := os.Getenv("ORT_DYLIB_PATH"); path != "" {
			ort.SetSharedLibraryPath(path)
		}
		// If ORT_DYLIB_PATH is not set, the binding locates the library via the OS
		// default mechanism (LD_LIBRARY_PATH). Report a clear error if init fails.
		if err := ort.InitializeEnvironment(); err != nil {
			initErr = fmt.Errorf("engine: failed to initialize ONNX Runtime (set ORT_DYLIB_PATH to libonnxruntime.so?): %w", err)
		}
	})
	return initErr
}

// IOInfo describes the name + shape of an I/O tensor of the model (probed from the ONNX file).
type IOInfo struct {
	Name  string
	Shape []int64
}

// Inspect probes the list of inputs/outputs (name + shape) from the ONNX file without creating a session.
// Used so the engine can auto-bind I/O names when the model does not declare them explicitly.
func Inspect(modelPath string) (inputs, outputs []IOInfo, err error) {
	if err = ensureORT(); err != nil {
		return nil, nil, err
	}
	in, out, e := ort.GetInputOutputInfo(modelPath)
	if e != nil {
		return nil, nil, fmt.Errorf("engine: failed to read I/O info from %s: %w", modelPath, e)
	}
	conv := func(src []ort.InputOutputInfo) []IOInfo {
		dst := make([]IOInfo, 0, len(src))
		for _, s := range src {
			dst = append(dst, IOInfo{Name: s.Name, Shape: append([]int64(nil), s.Dimensions...)})
		}
		return dst
	}
	return conv(in), conv(out), nil
}

// Session is a live ONNX session. Thread-safe: Run is protected by a mutex
// because an ORT session is not guaranteed safe when called concurrently with the same input binding.
// (see CLAUDE.md: session access must be thread-safe.)
type Session struct {
	mu          sync.Mutex
	sess        *ort.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
}

// NewSession creates a session from the ONNX file with an EP fallback chain (TensorRT→CUDA→CPU).
// If inputNames/outputNames are empty, they are auto-probed from the ONNX file.
// A failure to append an EP (e.g. TensorRT missing on the host) is NOT fatal — it falls back to the next EP.
func NewSession(modelPath string, inputNames, outputNames []string, providers []Provider) (*Session, error) {
	if err := ensureORT(); err != nil {
		return nil, err
	}

	// Auto-probe I/O names if not provided.
	if len(inputNames) == 0 || len(outputNames) == 0 {
		in, out, err := Inspect(modelPath)
		if err != nil {
			return nil, err
		}
		if len(inputNames) == 0 {
			for _, i := range in {
				inputNames = append(inputNames, i.Name)
			}
		}
		if len(outputNames) == 0 {
			for _, o := range out {
				outputNames = append(outputNames, o.Name)
			}
		}
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("engine: failed to create SessionOptions: %w", err)
	}
	defer opts.Destroy()

	// ORT C++ prints RED errors to stderr when a GPU provider (TensorRT/CUDA) cannot load its libs
	// (e.g. libnvinfer/libcudnn missing on a CPU-only host). This is NORMAL fallback behavior
	// (TensorRT→CUDA→CPU) but it is confusing. We redirect fd 2 around session creation:
	// if the session is created SUCCESSFULLY (after fallback) -> swallow the noise; if it FAILS
	// -> reprint everything so we don't hide the real error.
	var sess *ort.DynamicAdvancedSession
	captured, runErr := captureStderr(func() error {
		appendProviders(opts, providers)
		var e error
		sess, e = ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, opts)
		return e
	})
	if runErr != nil {
		if captured != "" {
			fmt.Fprint(os.Stderr, captured) // real error: restore ORT's original log
		}
		return nil, fmt.Errorf("engine: failed to create session for %s: %w", modelPath, runErr)
	}
	return &Session{sess: sess, inputNames: inputNames, outputNames: outputNames}, nil
}

// appendProviders adds the EPs in priority order. CPU is the default and is skipped.
// Each append error is just noted and ignored (fallback), without breaking the whole session.
func appendProviders(opts *ort.SessionOptions, providers []Provider) {
	for _, p := range providers {
		switch p {
		case ProviderTensorRT:
			if trt, err := ort.NewTensorRTProviderOptions(); err == nil {
				_ = opts.AppendExecutionProviderTensorRT(trt)
				trt.Destroy()
			}
		case ProviderCUDA:
			if cuda, err := ort.NewCUDAProviderOptions(); err == nil {
				_ = opts.AppendExecutionProviderCUDA(cuda)
				cuda.Destroy()
			}
		case ProviderCPU:
			// The CPU EP is ORT's default — no need to append.
		}
	}
}

// Run runs inference: takes input tensors (in the model's input order) and returns the
// output tensors (always float32) in outputNames order. Thread-safe.
//
// Inputs may be float32 (default) or int64 (Dtype=="i64", e.g. GroundingDINO text
// tokens). Outputs are read back as float32 (all current models output float32).
func (s *Session) Run(inputs []Tensor) ([]Tensor, error) {
	if len(inputs) != len(s.inputNames) {
		return nil, fmt.Errorf("engine: input count %d != model input count %d", len(inputs), len(s.inputNames))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.runLocked(inputs)
}

// RunNamed runs inference binding inputs BY NAME (robust when a model has many inputs
// whose ONNX order is not obvious, e.g. the SAM decoder's 6 inputs). Every input name
// the session declares must be present in the map.
func (s *Session) RunNamed(inputs map[string]Tensor) ([]Tensor, error) {
	ordered := make([]Tensor, 0, len(s.inputNames))
	for _, name := range s.inputNames {
		t, ok := inputs[name]
		if !ok {
			return nil, fmt.Errorf("engine: missing input %q (model wants %v)", name, s.inputNames)
		}
		ordered = append(ordered, t)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.runLocked(ordered)
}

// runLocked is the shared inference core; the caller must hold s.mu.
func (s *Session) runLocked(inputs []Tensor) ([]Tensor, error) {
	inVals := make([]ort.Value, 0, len(inputs))
	for i, t := range inputs {
		var (
			tensor ort.Value
			err    error
		)
		if t.isI64() {
			tensor, err = ort.NewTensor(ort.NewShape(t.Shape...), t.DataI64)
		} else {
			tensor, err = ort.NewTensor(ort.NewShape(t.Shape...), t.Data)
		}
		if err != nil {
			destroyValues(inVals)
			return nil, fmt.Errorf("engine: failed to create input tensor #%d (%q): %w", i, s.inputNames[i], err)
		}
		inVals = append(inVals, tensor)
	}
	defer destroyValues(inVals)

	// nil outputs -> ORT allocates; we read them back after Run.
	outVals := make([]ort.Value, len(s.outputNames))
	if err := s.sess.Run(inVals, outVals); err != nil {
		return nil, fmt.Errorf("engine: Run failed: %w", err)
	}
	defer destroyValues(outVals)

	outs := make([]Tensor, 0, len(outVals))
	for i, v := range outVals {
		ft, ok := v.(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("engine: output %q is not float32 (unsupported dtype)", s.outputNames[i])
		}
		data := ft.GetData()
		cp := make([]float32, len(data))
		copy(cp, data) // copy because the value is Destroyed in defer
		outs = append(outs, Tensor{Data: cp, Shape: append([]int64(nil), ft.GetShape()...)})
	}
	return outs, nil
}

// InputNames returns the model input names in binding order.
func (s *Session) InputNames() []string { return s.inputNames }

// OutputNames returns output names in the order Run returns them.
func (s *Session) OutputNames() []string { return s.outputNames }

// Close releases the session (to avoid VRAM leaks). Must be called via lifecycle on unload.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sess != nil {
		err := s.sess.Destroy()
		s.sess = nil
		return err
	}
	return nil
}

func destroyValues(vals []ort.Value) {
	for _, v := range vals {
		if v != nil {
			v.Destroy()
		}
	}
}
