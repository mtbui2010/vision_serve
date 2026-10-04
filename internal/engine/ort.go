// Package engine wraps ONNX Runtime (binding github.com/yalue/onnxruntime_go).
//
// Principle (CLAUDE.md): do NOT write your own inference engine. All inference goes through ORT.
// Why ORT: the same .onnx file runs on both GPU (CUDA EP by default, TensorRT EP opt-in) and
// CPU — matching the edge↔server goal.
//
// The binding requires the libonnxruntime.so shared library at runtime. The path comes from
// the ORT_DYLIB_PATH environment variable (e.g. /usr/local/lib/libonnxruntime.so). On Jetson
// use an ORT build with the CUDA EP (and the TensorRT EP if you opt into it).
package engine

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
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
	// ortLibPath is the name the binding dlopen()ed ONNX Runtime by: ORT_DYLIB_PATH, or the
	// binding's own default. setDeterministicCompute looks the loaded library up by it.
	ortLibPath = "onnxruntime.so"
)

// ensureORT initializes the ORT environment exactly once per process.
func ensureORT() error {
	initOnce.Do(func() {
		if path := os.Getenv("ORT_DYLIB_PATH"); path != "" {
			ort.SetSharedLibraryPath(path)
			ortLibPath = path
		}
		// If ORT_DYLIB_PATH is not set, the binding locates the library via the OS
		// default mechanism (LD_LIBRARY_PATH). Report a clear error if init fails.
		if err := ort.InitializeEnvironment(); err != nil {
			initErr = fmt.Errorf("engine: failed to initialize ONNX Runtime (set ORT_DYLIB_PATH to libonnxruntime.so?): %w", err)
		}
	})
	return initErr
}

// ErrClosed is returned by a session (or pool) used after Close — e.g. a request that raced an
// unload. Callers report it as an ordinary error; it must never be a panic or a hang.
var ErrClosed = errors.New("engine: session is closed")

// ErrInferencePanic wraps a panic recovered on a session's worker thread (during a job or while
// creating the session). The worker is not a request goroutine, so net/http's per-request
// recovery never sees it: unrecovered, one bad call took the whole server down.
var ErrInferencePanic = errors.New("engine: panic on the session thread")

// recoverJob turns a panic on the worker thread into an error wrapping ErrInferencePanic, and
// logs the stack (the error itself only carries the panic value).
func recoverJob(p any) error {
	fmt.Fprintf(os.Stderr, "engine: recovered panic on a session thread: %v\n%s", p, debug.Stack())
	return fmt.Errorf("%w: %v", ErrInferencePanic, p)
}

// IOInfo describes the name + shape of an I/O tensor of the model (probed from the ONNX file).
type IOInfo struct {
	Name string
	// Shape has one entry per dimension: its size, or -1 when symbolic or unknown. It is nil for
	// a scalar, a tensor of unknown rank and a non-tensor value (sequence, map, optional).
	//
	// It is the shape the file DECLARES (bar Inspect's ORT fallback). For an output, a live ORT
	// session can know more: its load-time shape inference resolves some declared-symbolic dims
	// (GroundingDINO's pred_boxes is declared [-1, -1, 4]; ORT reports [-1, 900, 4]). Read real
	// output shapes from Run.
	Shape []int64
	// ElemType is the ONNX element type (TensorProto.DataType: 1 float32, 7 int64, 9 bool,
	// 10 float16, ...); 0 for a non-tensor value.
	ElemType int32
}

// Inspect probes the list of inputs/outputs (name + shape) from the ONNX file without creating a session.
// Used so the engine can auto-bind I/O names when the model does not declare them explicitly.
//
// It reads the ONNX header in pure Go (onnxheader.go) — milliseconds, weights untouched, no ORT
// needed. Only a file that reader cannot parse goes to ORT's GetInputOutputInfo, which builds a
// full temporary session (seconds for a large graph); that fallback is logged once per file.
func Inspect(modelPath string) (inputs, outputs []IOInfo, err error) {
	in, out, herr := readONNXHeader(modelPath)
	if herr == nil {
		return in, out, nil
	}
	if errors.Is(herr, fs.ErrNotExist) || errors.Is(herr, fs.ErrPermission) {
		return nil, nil, fmt.Errorf("engine: failed to read I/O info from %s: %w", modelPath, herr)
	}
	if _, logged := inspectFallbackLogged.LoadOrStore(modelPath, true); !logged {
		fmt.Fprintf(os.Stderr, "engine: could not read the ONNX header of %s (%v) — "+
			"asking ONNX Runtime instead, which loads the whole graph\n", modelPath, herr)
	}
	return inspectORT(modelPath)
}

// inspectFallbackLogged holds the paths whose header-parse failure has been logged, so a file
// probed on every request (lifecycle's preprocess runner) does not log on every request.
var inspectFallbackLogged sync.Map

// inspectORT is the ORT-based probe: correct for anything ORT can load, but it builds a full
// session just to read the I/O list.
func inspectORT(modelPath string) (inputs, outputs []IOInfo, err error) {
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
			dst = append(dst, IOInfo{
				Name:     s.Name,
				Shape:    append([]int64(nil), s.Dimensions...),
				ElemType: int32(s.DataType),
			})
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

	jobs chan func() // work funnelled onto the dedicated OS thread; closed by Close
	// jobsMu guards sending on jobs against Close closing it: submit holds the read lock while
	// it hands work over, Close takes the write lock to mark the session closed. A call that
	// arrives after Close gets ErrClosed instead of a "send on closed channel" panic, which in
	// a goroutine the model spawned itself (MobileSAM automask) used to crash the process.
	jobsMu    sync.RWMutex
	closed    bool
	closeOnce sync.Once
	closeErr  chan error // worker sends the Destroy() result here after jobs drains
}

// ActiveEP returns the execution provider that is actually running this session.
func (s *Session) ActiveEP() Provider { return s.activeEP }

// SessionOptions tunes one session beyond its EP chain. The zero value keeps ONNX Runtime's
// defaults.
type SessionOptions struct {
	// IntraOpThreads sizes the session's intra-op thread pool (0 = ORT's default, one thread per
	// physical core). Every session owns its own pool and its idle threads spin, so N pooled
	// copies of a graph running at once at the default would put N×cores busy threads on the
	// machine. The thread count does not change the outputs.
	IntraOpThreads int
}

// NewSession creates a session from the ONNX file with an EP fallback chain (as resolved by
// ResolveProviders: CUDA→CPU by default, TensorRT→CUDA→CPU with the TensorRT opt-in).
// If inputNames/outputNames are empty, they are auto-probed from the ONNX file (Inspect: a header
// read, so the graph is loaded once — by the session itself, not again by the probe).
// A failure to append an EP (e.g. TensorRT missing on the host) is NOT fatal — it falls back to the next EP.
func NewSession(modelPath string, inputNames, outputNames []string, providers []Provider) (*Session, error) {
	return NewSessionWith(modelPath, inputNames, outputNames, providers, SessionOptions{})
}

// NewSessionWith is NewSession with explicit SessionOptions.
func NewSessionWith(modelPath string, inputNames, outputNames []string, providers []Provider, so SessionOptions) (*Session, error) {
	if so.IntraOpThreads < 0 {
		return nil, fmt.Errorf("engine: IntraOpThreads must be >= 0, got %d", so.IntraOpThreads)
	}
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
	go s.worker(modelPath, providers, so, ready)
	if err := <-ready; err != nil {
		return nil, err
	}
	return s, nil
}

// worker owns the session's single OS thread for its entire lifetime: it creates the ORT
// session, runs every job serially, and destroys the session — all on the same locked thread.
// This is what keeps ORT's CUDA EP to one per-thread context per session (see Session doc).
func (s *Session) worker(modelPath string, providers []Provider, so SessionOptions, ready chan<- error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	sess, ep, err := func() (sess *ort.DynamicAdvancedSession, ep Provider, err error) {
		defer func() {
			if p := recover(); p != nil {
				sess, err = nil, recoverJob(p)
			}
		}()
		return createORTSession(modelPath, s.inputNames, s.outputNames, providers, so)
	}()
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

// createORTSession is createSession; tests replace it to inject a failure on the worker thread.
var createORTSession = createSession

// createSession builds an ORT session, trying each EP in priority order with that single EP
// appended. If session creation FAILS (not just a missing lib, but the EP cannot build/partition
// this specific graph — e.g. TensorRT rejecting the MobileSAM decoder's Float shape tensor
// `orig_im_size`), it falls back to the NEXT EP in the chain instead of failing the whole load.
// CPU is always the last candidate, so a session is always created. Must run on the worker thread.
//
// ORT prints RED errors to stderr while an EP fails over. On eventual success we swallow that
// noise (it is normal fallback); if ALL EPs are exhausted we reprint the last error. Only ORT's
// own log lines are held back — see stderr.go for the capture and why it is still an fd swap.
func createSession(modelPath string, inputNames, outputNames []string, providers []Provider, so SessionOptions) (*ort.DynamicAdvancedSession, Provider, error) {
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
		if so.IntraOpThreads > 0 {
			if err := opts.SetIntraOpNumThreads(so.IntraOpThreads); err != nil {
				opts.Destroy()
				return nil, activeEP, fmt.Errorf("engine: set %d intra-op threads: %w", so.IntraOpThreads, err)
			}
		}
		// The CPU session runs without ORT's memory pattern (the CPU arena stays on). With both on,
		// the first Run allocates its tensors one by one from the arena, and the second plans one
		// block for all of them — which the fragmented first-run chunks cannot hold, so the arena
		// grows a new region and keeps both: measured on CPU (fresh server, 5–10 identical
		// requests, VmHWM) grounding-dino 2.24 → 3.47 GB from the 2nd request on, rf-detr 0.32 →
		// 0.48 GB, MobileSAM automask 0.65 → 0.98 GB (640×480), and more for every new input shape
		// or orig_im_size. Off, each stays at its first-request peak; latency did not move
		// (rf-detr 0.15 s, grounding-dino 1.8 s, automask ~4 s, within run-to-run noise), and the
		// pattern only places buffers, so outputs are bit-identical. Turning the arena off instead
		// was worse on both counts (automask +15 % memory and +15–20 % latency). GPU EPs keep
		// ORT's defaults: not measured there.
		if ep == ProviderCPU {
			if err := opts.SetMemPattern(false); err != nil {
				opts.Destroy()
				return nil, activeEP, fmt.Errorf("engine: disable the memory pattern: %w", err)
			}
		}
		var s *ort.DynamicAdvancedSession
		// An EP this ORT build does not ship (CUDA on the CPU-only wheel or the CPU Docker image)
		// fails HERE, at append time. The session would still be created — on CPU — so this EP
		// must be skipped, not merely ignored: ignoring it is how a CPU run reported "gpu:0".
		if err := applyProvider(opts, ep, modelPath); err != nil {
			opts.Destroy()
			if Trace {
				fmt.Fprintf(os.Stderr, "engine: [trace] EP %s unavailable in this ONNX Runtime build (%v) — skipping\n",
					providerNames([]Provider{ep}), err)
			}
			continue
		}
		applyDeterministic(opts, ep) // GPU EPs only; best effort (deterministic.go)
		create := func() error {
			var e error
			s, e = ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, opts)
			return e
		}
		var captured string
		var runErr error
		if ep == ProviderCPU {
			// The CPU EP cannot be dropped and has no next EP to fall back to, so there is nothing to
			// capture — and skipping the process-wide fd swap (and its lock) lets CPU loads run in
			// parallel. See stderr.go.
			runErr = create()
		} else {
			captured, runErr = captureStderr(create)
		}
		opts.Destroy()
		if runErr == nil {
			sess, activeEP = s, ep
			// Session creation can SUCCEED while ORT quietly drops the EP we registered and runs
			// on CPU — what happens when libonnxruntime_providers_cuda.so cannot resolve
			// libcudnn.so.9. runErr is nil, so nothing here notices, and activeEP goes on
			// claiming "gpu:0" for CPU work. That silence cost a full measurement sweep in this
			// project: it was caught only because the process held no VRAM, and it had already
			// produced a plausible-looking table of wrong numbers.
			if ep != ProviderCPU && epWasDropped(captured) {
				fmt.Fprintf(os.Stderr,
					"engine: WARNING %s requested %s but ONNX Runtime fell back to CPU — "+
						"inference will be slow, and `device` would otherwise misreport it as GPU. "+
						"Usually a missing cuDNN/CUDA runtime on LD_LIBRARY_PATH; try "+
						"`source scripts/gpu-env.sh`. ORT said:\n%s",
					filepath.Base(modelPath), providerNames([]Provider{ep}), captured)
				activeEP = ProviderCPU
			}
			if Trace { // the EP that runs it, then anything ORT said while creating the session
				fmt.Fprintf(os.Stderr, "engine: [trace] session for %s active on EP %s\n%s",
					filepath.Base(modelPath), providerNames([]Provider{activeEP}), captured)
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
//
// It returns the error when the EP cannot be appended — typically because this ORT build does
// not include it. The caller MUST then skip the EP: a session created from these options runs
// on CPU, and recording the requested EP as active would report a GPU that is not in use.
func applyProvider(opts *ort.SessionOptions, p Provider, modelPath string) error {
	switch p {
	case ProviderTensorRT:
		trt, err := ort.NewTensorRTProviderOptions()
		if err != nil {
			return err
		}
		defer trt.Destroy()
		// Persist compiled engines; without this every load re-compiles (minutes). See TRTOptions.
		if trtOpts := TRTOptions(modelPath); len(trtOpts) > 0 {
			if err := trt.Update(trtOpts); err != nil && Trace {
				fmt.Fprintf(os.Stderr, "engine: [trace] TensorRT options rejected (%v) — continuing without engine cache\n", err)
			}
		}
		return opts.AppendExecutionProviderTensorRT(trt)
	case ProviderCUDA:
		cuda, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return err
		}
		defer cuda.Destroy()
		return opts.AppendExecutionProviderCUDA(cuda)
	case ProviderCoreML:
		return opts.AppendExecutionProviderCoreML(0)
	case ProviderDirectML:
		return opts.AppendExecutionProviderDirectML(0)
	case ProviderOpenVINO:
		return opts.AppendExecutionProviderOpenVINO(map[string]string{})
	}
	return nil // ProviderCPU: ORT built-in, nothing to append
}

// Run runs inference: takes input tensors (in the model's input order) and returns the
// output tensors (always float32) in outputNames order. Thread-safe.
//
// Inputs may be float32 (default) or int64 (Dtype=="i64", e.g. GroundingDINO text
// tokens). Outputs are read back as float32 (all current models output float32).
//
// ctx bounds only the wait for the session's worker (see Runnable).
func (s *Session) Run(ctx context.Context, inputs []Tensor) ([]Tensor, error) {
	if len(inputs) != len(s.inputNames) {
		return nil, fmt.Errorf("engine: input count %d != model input count %d", len(inputs), len(s.inputNames))
	}
	return s.submit(ctx, func() ([]Tensor, error) { return s.runOnThread(inputs, nil) })
}

// RunNamed runs inference binding inputs BY NAME (robust when a model has many inputs
// whose ONNX order is not obvious, e.g. the SAM decoder's 6 inputs). Every input name
// the session declares must be present in the map.
func (s *Session) RunNamed(ctx context.Context, inputs map[string]Tensor) ([]Tensor, error) {
	return s.RunNamedInto(ctx, inputs, nil)
}

// RunNamedInto is RunNamed with caller-owned buffers for some outputs: ONNX Runtime writes each
// output named in into straight into that tensor's Data (float32, and its Shape must be exactly
// the shape the run produces, or Run fails), and the returned tensor for that output aliases it.
// The other outputs are allocated by ORT and copied out as in RunNamed.
//
// It is for large outputs a caller consumes and drops at once (MobileSAM's full-resolution
// mask logits, 30 MB at 3200×2400): the caller reuses one buffer across calls, where RunNamed
// costs an ORT arena allocation plus a Go copy per call. The values are the same either way.
//
// ctx follows RunNamed's contract (submit): a done ctx never starts the run, and once the worker
// has taken it RunNamedInto returns only after ORT has finished. So when it returns — result,
// error or ctx — nothing writes into the buffers any more, and the caller may free them.
func (s *Session) RunNamedInto(ctx context.Context, inputs, into map[string]Tensor) ([]Tensor, error) {
	ordered := make([]Tensor, 0, len(s.inputNames))
	for _, name := range s.inputNames {
		t, ok := inputs[name]
		if !ok {
			return nil, fmt.Errorf("engine: missing input %q (model wants %v)", name, s.inputNames)
		}
		ordered = append(ordered, t)
	}
	var outs []Tensor
	if len(into) > 0 {
		outs = make([]Tensor, len(s.outputNames))
		n := 0
		for i, name := range s.outputNames {
			if t, ok := into[name]; ok {
				if t.isI64() || len(t.Data) == 0 {
					return nil, fmt.Errorf("engine: output buffer %q must be a non-empty float32 tensor", name)
				}
				outs[i] = t
				n++
			}
		}
		if n != len(into) {
			return nil, fmt.Errorf("engine: output buffers %v name an output the model does not have (outputs %v)",
				mapKeys(into), s.outputNames)
		}
	}
	return s.submit(ctx, func() ([]Tensor, error) { return s.runOnThread(ordered, outs) })
}

func mapKeys(m map[string]Tensor) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// submit funnels one unit of inference work onto the session's dedicated OS thread and waits
// for its result. The worker processes jobs serially, so this also serializes concurrent
// callers (replacing the old mutex) while guaranteeing the ORT call runs on the pinned thread.
//
// The hand-over is where a caller queues behind the job running now, so that wait follows ctx: a
// caller whose ctx is done before the worker takes its job gives up, and the job never runs. Once
// the worker has it, submit waits for the result whatever ctx does: ORT cannot be interrupted.
func (s *Session) submit(ctx context.Context, work func() ([]Tensor, error)) ([]Tensor, error) {
	type result struct {
		outs []Tensor
		err  error
	}
	if err := ctx.Err(); err != nil { // an idle worker must not win over a ctx already done
		return nil, gaveUp(err)
	}
	ch := make(chan result, 1)
	s.jobsMu.RLock()
	if s.closed {
		s.jobsMu.RUnlock()
		return nil, ErrClosed
	}
	job := func() {
		var r result
		// A panic in the job must not unwind the worker (it would kill the process, and with it
		// every other session). The session stays usable: the panic is in Go code around the ORT
		// call — cgo cannot unwind a Go panic through C, ORT reports its own failures as error
		// statuses — and the job's deferred tensor Destroy calls run during the unwind.
		defer func() {
			if p := recover(); p != nil {
				r = result{nil, recoverJob(p)}
			}
			ch <- r
		}()
		r.outs, r.err = work()
	}
	select {
	case s.jobs <- job:
	case <-ctx.Done():
		s.jobsMu.RUnlock()
		return nil, gaveUp(ctx.Err())
	}
	s.jobsMu.RUnlock()
	// Work accepted before Close still runs: the worker drains jobs before destroying the session.
	r := <-ch
	return r.outs, r.err
}

// runOnThread is the shared inference core; it ONLY ever executes on the worker's locked OS
// thread (via submit), so it touches s.sess without further locking.
//
// into, when non-nil, is index-aligned with the outputs: an entry with Data is a caller-owned
// buffer ORT writes that output into (see RunNamedInto); a zero entry lets ORT allocate.
func (s *Session) runOnThread(inputs, into []Tensor) ([]Tensor, error) {
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

	// nil outputs -> ORT allocates; we read them back after Run. A caller buffer is wrapped
	// as-is, so ORT writes that output straight into Go memory.
	outVals := make([]ort.Value, len(s.outputNames))
	defer destroyValues(outVals)
	for i := range into {
		if into[i].Data == nil {
			continue
		}
		v, err := ort.NewTensor(ort.NewShape(into[i].Shape...), into[i].Data)
		if err != nil {
			return nil, fmt.Errorf("engine: failed to wrap output buffer %q: %w", s.outputNames[i], err)
		}
		outVals[i] = v
	}
	if err := s.sess.Run(inVals, outVals); err != nil {
		return nil, fmt.Errorf("engine: Run failed: %w", err)
	}

	outs := make([]Tensor, 0, len(outVals))
	for i, v := range outVals {
		ft, ok := v.(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("engine: output %q is not float32 (unsupported dtype)", s.outputNames[i])
		}
		data := ft.GetData()
		shape := append([]int64(nil), ft.GetShape()...)
		if i < len(into) && into[i].Data != nil {
			outs = append(outs, Tensor{Data: data, Shape: shape}) // the caller's buffer: no copy
			continue
		}
		cp := make([]float32, len(data))
		copy(cp, data) // copy because the value is Destroyed in defer
		outs = append(outs, Tensor{Data: cp, Shape: shape})
	}
	return outs, nil
}

// InputNames returns the model input names in binding order.
func (s *Session) InputNames() []string { return s.inputNames }

// OutputNames returns output names in the order Run returns them.
func (s *Session) OutputNames() []string { return s.outputNames }

// Close releases the session (to avoid VRAM leaks). Must be called via lifecycle on unload.
// It marks the session closed (later calls get ErrClosed), closes the job channel so the
// worker drains the jobs it already accepted (they still run), destroys the session on its
// own pinned thread (releasing that thread's CUDA per-thread context), then exits.
// Safe to call more than once and concurrently: the first call does the work and returns the
// Destroy error; every other call waits for it to finish and returns nil.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() {
		s.jobsMu.Lock()
		s.closed = true
		close(s.jobs)
		s.jobsMu.Unlock()
		err = <-s.closeErr
	})
	return err
}

// gaveUp is the error of a call that stopped waiting for a session because its ctx ended. It wraps
// the ctx error, so a caller can tell it from an inference failure with errors.Is.
func gaveUp(ctxErr error) error {
	return fmt.Errorf("engine: stopped waiting for the session, nothing was run: %w", ctxErr)
}

func destroyValues(vals []ort.Value) {
	for _, v := range vals {
		if v != nil {
			v.Destroy()
		}
	}
}

// epWasDropped reports whether ORT's own diagnostics say it abandoned the execution provider we
// registered and fell back to CPU, even though session creation returned no error.
//
// Matching log text is brittle and is chosen deliberately over the alternatives: the ORT Go
// binding exposes no "which EP is this session actually using" query, and probing by running a
// tensor would cost a real inference on every session creation. The markers below are the two
// shapes ORT emits — a provider shared library that will not load, and its explicit fallback
// notice. A missed match degrades to today's behaviour (silent), never to a false alarm on a
// healthy session, because a working GPU session prints neither.
func epWasDropped(ortOutput string) bool {
	if ortOutput == "" {
		return false
	}
	s := strings.ToLower(ortOutput)
	for _, marker := range []string{
		"failed to load library", // provider .so missing a dependency (libcudnn, libnvinfer)
		"falling back to cpuexecutionprovider",
		"failed to create cudaexecutionprovider",
		"failed to create tensorrtexecutionprovider",
	} {
		if strings.Contains(s, marker) {
			return true
		}
	}
	return false
}
