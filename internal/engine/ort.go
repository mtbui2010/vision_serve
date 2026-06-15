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
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Trace, when VISIONSERVE_TRACE is set, surfaces the execution-provider selection and
// ORT's own diagnostic messages during session creation. This makes a silent fallback
// to CPU (e.g. a GPU EP failing because libcudnn/libnvinfer is missing from
// LD_LIBRARY_PATH) visible, instead of being swallowed. Use it to answer "is this
// actually running on the GPU?".
var Trace = os.Getenv("VISIONSERVE_TRACE") != ""

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

// Session is a live ONNX session. Thread-safe AND OS-thread-pinned: every call to the
// underlying ORT session (create, Run, Destroy) is funnelled onto ONE dedicated OS thread
// owned by this Session's worker goroutine (see worker / submit).
//
// Why pinned, not just mutex-serialized: ORT's CUDA execution provider allocates GPU
// resources PER OS THREAD (a cublas + cudnn handle and a memory arena, created lazily on the
// first Run seen on each thread). Go freely migrates a goroutine across OS threads between
// calls — and cgo calls in particular spawn fresh threads — so a plain mutex would let each
// inference land on a different thread, leaking a new CUDA context every time until
// `cublasCreate` fails with "CUBLAS failure 3: the resource allocation failed" after a few
// requests. Pinning to one thread means exactly one CUDA per-thread context per session for
// its whole lifetime. (see CLAUDE.md: session access must be thread-safe + VRAM-safe.)
type Session struct {
	sess        *ort.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
	activeEP    Provider // EP that was actually loaded (first one whose libs were available)

	jobs      chan func() // work funnelled onto the dedicated OS thread; closed by Close
	closeOnce sync.Once
	closeErr  chan error // worker sends the Destroy() result here after jobs drains
}

// ActiveEP returns the execution provider that is actually running this session.
func (s *Session) ActiveEP() Provider { return s.activeEP }

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

	s := &Session{
		inputNames:  inputNames,
		outputNames: outputNames,
		jobs:        make(chan func()),
		closeErr:    make(chan error, 1),
	}
	// Start the dedicated, OS-thread-pinned worker. It creates the ORT session on that thread
	// so the session's CUDA per-thread context is bound to the same thread that will Run and
	// Destroy it — exactly one context for the session's lifetime (see Session doc).
	ready := make(chan error, 1)
	go s.worker(modelPath, providers, ready)
	if err := <-ready; err != nil {
		return nil, err
	}
	return s, nil
}

// worker owns the session's single OS thread for its entire lifetime: it creates the ORT
// session, runs every job serially, and destroys the session — all on the same locked thread.
// This is what keeps ORT's CUDA EP to one per-thread context per session (see Session doc).
func (s *Session) worker(modelPath string, providers []Provider, ready chan<- error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	sess, ep, err := createSession(modelPath, s.inputNames, s.outputNames, providers)
	if err != nil {
		ready <- err
		return
	}
	s.sess = sess
	s.activeEP = ep
	ready <- nil // happens-before NewSession's return: s.sess/s.activeEP are safely published

	for fn := range s.jobs {
		fn()
	}
	s.closeErr <- s.sess.Destroy() // jobs closed by Close: destroy on the same locked thread
}

// createSession builds an ORT session, trying each EP in priority order with that single EP
// appended. If session creation FAILS (not just a missing lib, but the EP cannot build/partition
// this specific graph — e.g. TensorRT rejecting the MobileSAM decoder's Float shape tensor
// `orig_im_size`), it falls back to the NEXT EP in the chain instead of failing the whole load.
// CPU is always the last candidate, so a session is always created. Must run on the worker thread.
//
// ORT prints RED errors to stderr while an EP fails over. On eventual success we swallow that
// noise (it is normal fallback); if ALL EPs are exhausted we reprint the last error.
func createSession(modelPath string, inputNames, outputNames []string, providers []Provider) (*ort.DynamicAdvancedSession, Provider, error) {
	candidates := availableProviders(providers)
	if Trace {
		fmt.Fprintf(os.Stderr, "engine: [trace] creating session for %s — EP chain: %s\n",
			filepath.Base(modelPath), providerNames(candidates))
	}

	var sess *ort.DynamicAdvancedSession
	var activeEP Provider
	var lastErr error
	var lastCaptured string
	for _, ep := range candidates {
		opts, err := ort.NewSessionOptions()
		if err != nil {
			return nil, activeEP, fmt.Errorf("engine: failed to create SessionOptions: %w", err)
		}
		var s *ort.DynamicAdvancedSession
		captured, runErr := captureStderr(func() error {
			applyProvider(opts, ep)
			var e error
			s, e = ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, opts)
			return e
		})
		opts.Destroy()
		if runErr == nil {
			sess, activeEP = s, ep
			if Trace && captured != "" {
				fmt.Fprintf(os.Stderr, "engine: [trace] ORT messages for %s on %s:\n%s",
					filepath.Base(modelPath), providerNames([]Provider{ep}), captured)
			}
			break
		}
		lastErr, lastCaptured = runErr, captured
		if Trace {
			fmt.Fprintf(os.Stderr, "engine: [trace] EP %s failed for %s (%v) — falling back to next EP\n",
				providerNames([]Provider{ep}), filepath.Base(modelPath), runErr)
		}
	}
	if sess == nil {
		if lastCaptured != "" {
			fmt.Fprint(os.Stderr, lastCaptured) // real error: restore ORT's original log
		}
		return nil, activeEP, fmt.Errorf("engine: failed to create session for %s (all EPs exhausted): %w",
			filepath.Base(modelPath), lastErr)
	}
	return sess, activeEP, nil
}

// providerNames renders the requested EP order for trace output.
func providerNames(providers []Provider) string {
	names := make([]string, 0, len(providers))
	for _, p := range providers {
		switch p {
		case ProviderTensorRT:
			names = append(names, "tensorrt")
		case ProviderCUDA:
			names = append(names, "cuda")
		case ProviderCoreML:
			names = append(names, "coreml")
		case ProviderDirectML:
			names = append(names, "directml")
		case ProviderOpenVINO:
			names = append(names, "openvino")
		case ProviderCPU:
			names = append(names, "cpu")
		}
	}
	return strings.Join(names, " → ")
}

// availableProviders filters the requested chain to EPs worth attempting, preserving order:
// it drops TensorRT when libnvinfer.so.10 is absent (loading the TRT provider lib without it
// causes a hard C abort: dlopen → libnvinfer missing → SIGABRT), and guarantees CPU is the
// final candidate so a session can always be created.
func availableProviders(providers []Provider) []Provider {
	out := make([]Provider, 0, len(providers)+1)
	seenCPU := false
	for _, p := range providers {
		if p == ProviderTensorRT && !TRTAvailable() {
			continue
		}
		if p == ProviderCPU {
			seenCPU = true
		}
		out = append(out, p)
	}
	if !seenCPU {
		out = append(out, ProviderCPU)
	}
	return out
}

// applyProvider appends exactly ONE execution provider to opts. CPU needs no append
// (ORT's built-in default); callers attempt providers one at a time so a per-graph EP
// failure can fall back to the next candidate (see NewSession).
func applyProvider(opts *ort.SessionOptions, p Provider) {
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
	case ProviderCoreML:
		_ = opts.AppendExecutionProviderCoreML(0)
	case ProviderDirectML:
		_ = opts.AppendExecutionProviderDirectML(0)
	case ProviderOpenVINO:
		_ = opts.AppendExecutionProviderOpenVINO(map[string]string{})
	case ProviderCPU:
		// ORT built-in; nothing to append.
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
	return s.submit(func() ([]Tensor, error) { return s.runOnThread(inputs) })
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
	return s.submit(func() ([]Tensor, error) { return s.runOnThread(ordered) })
}

// submit funnels one unit of inference work onto the session's dedicated OS thread and waits
// for its result. The worker processes jobs serially, so this also serializes concurrent
// callers (replacing the old mutex) while guaranteeing the ORT call runs on the pinned thread.
func (s *Session) submit(work func() ([]Tensor, error)) ([]Tensor, error) {
	type result struct {
		outs []Tensor
		err  error
	}
	ch := make(chan result, 1)
	s.jobs <- func() {
		outs, err := work()
		ch <- result{outs, err}
	}
	r := <-ch
	return r.outs, r.err
}

// runOnThread is the shared inference core; it ONLY ever executes on the worker's locked OS
// thread (via submit), so it touches s.sess without further locking.
func (s *Session) runOnThread(inputs []Tensor) ([]Tensor, error) {
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
// It closes the job channel so the worker drains any in-flight jobs, destroys the session on
// its own pinned thread (releasing that thread's CUDA per-thread context), then exits.
// Idempotent and safe to call once; concurrent inference must have stopped first.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() {
		close(s.jobs)
		err = <-s.closeErr
	})
	return err
}

func destroyValues(vals []ort.Value) {
	for _, v := range vals {
		if v != nil {
			v.Destroy()
		}
	}
}
