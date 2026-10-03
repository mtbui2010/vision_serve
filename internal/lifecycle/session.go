package lifecycle

import (
	"context"
	"fmt"
	"image"
	"sort"
	"strings"
	"sync"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/pkg/api"
)

// Session wraps a live model: its pre/postprocess (models.*) plus its heavy ONNX
// session(s) (engine.Session). One Session serves many requests concurrently;
// engine.Session.Run locks a mutex, so inference is safely serialized.
//
// Two modes (exactly one is set):
//   - simple: a models.Model + ONE engine.Session (pre → infer → post). e.g. rf-detr.
//   - pipeline: a models.PipelineModel + a map role→engine.Session. The model drives
//     its own multi-stage, prompted inference via a Runner. e.g. MobileSAM, GroundingDINO.
type Session struct {
	name        string
	task        api.Task
	device      string // "gpu:0", "cpu", … or "mixed(…)" — from the active EP(s) at load (pipelineDevice)
	idleTimeout time.Duration

	// man is the manifest this session was built from, captured at load time. Everything that
	// runs against the loaded sessions (explain) reads it, never the registry: a rescan may have
	// replaced the entry with an edited manifest that no longer matches what is in memory.
	man *registry.Manifest

	model  models.Model    // simple mode
	engine engine.Runnable // simple mode (a single session, or a pool when VS_POOL_OVERRIDE>1)

	pipeline models.PipelineModel       // pipeline mode
	engines  map[string]engine.Runnable // pipeline mode (role → session or pool)
	// exclusive (pipeline mode): the model implements models.Exclusive and asked for it, so its
	// Infer runs holding inferLock — one request at a time on THIS loaded model, other models and
	// other models' sessions unaffected. Decided once at load. inferLock is a 1-slot semaphore
	// rather than a sync.Mutex so a request waiting for it can give up when its client leaves.
	exclusive bool
	inferLock chan struct{}

	// explainEngine is the same ONNX file loaded with all outputs including explain tensors.
	// nil until the first /api/explain call (lazy load). Lifecycle owns it.
	explainEngine engine.Runnable

	mu       sync.Mutex
	lastUsed time.Time

	// refs counts requests currently using this session and retired marks it removed from the
	// Manager's live map; both are guarded by Manager.mu (see Manager.acquire / retireLocked).
	// A retired session is closed by whoever drops the last reference — never under a request.
	refs    int
	retired bool
}

func newSimpleSession(name string, task api.Task, m models.Model, eng engine.Runnable, idle time.Duration, now time.Time) *Session {
	return &Session{
		name:        name,
		task:        task,
		device:      engine.RunnableDevice(eng),
		model:       m,
		engine:      eng,
		idleTimeout: idle,
		lastUsed:    now,
	}
}

func newPipelineSession(name string, task api.Task, p models.PipelineModel, engs map[string]engine.Runnable, idle time.Duration, now time.Time) *Session {
	ex, ok := p.(models.Exclusive)
	return &Session{
		name:        name,
		task:        task,
		device:      pipelineDevice(engs),
		pipeline:    p,
		engines:     engs,
		exclusive:   ok && ex.Exclusive(),
		inferLock:   make(chan struct{}, 1),
		idleTimeout: idle,
		lastUsed:    now,
	}
}

// pipelineDevice reports where a pipeline's roles run. When every role is on the same device the
// answer is that device ("gpu:0", "cpu" — the format clients already parse); otherwise it names
// each role, sorted: "mixed(decoder=cpu,encoder=gpu:0)". Each EP falls back on its own (TensorRT
// can reject one graph and accept the other), so roles do not necessarily share one. A pipeline
// with no session reports "cpu".
func pipelineDevice(engs map[string]engine.Runnable) string {
	if len(engs) == 0 {
		return "cpu"
	}
	roles := make([]string, 0, len(engs))
	for role := range engs {
		roles = append(roles, role)
	}
	sort.Strings(roles)
	parts := make([]string, len(roles))
	uniform := true
	first := engine.RunnableDevice(engs[roles[0]])
	for i, role := range roles {
		dev := engine.RunnableDevice(engs[role])
		uniform = uniform && dev == first
		parts[i] = role + "=" + dev
	}
	if uniform {
		return first
	}
	return "mixed(" + strings.Join(parts, ",") + ")"
}

// inferPipeline runs the pipeline model's Infer, under the session's lock when it is Exclusive.
// Waiting for the lock follows ctx; so does every session call the model makes through the runner,
// so a request whose client leaves mid-pipeline stops before its next stage instead of running it.
func (s *Session) inferPipeline(ctx context.Context, img image.Image, prompt models.Prompt) (api.Result, error) {
	if s.exclusive {
		if err := ctx.Err(); err != nil { // a free lock must not win over a ctx already done
			return api.Result{}, gaveUp(s.name, err)
		}
		select {
		case s.inferLock <- struct{}{}:
			defer func() { <-s.inferLock }()
		case <-ctx.Done():
			return api.Result{}, gaveUp(s.name, ctx.Err())
		}
	}
	return s.pipeline.Infer(img, prompt, runner{ctx: ctx, engines: s.engines})
}

// gaveUp is the error of a request that stopped waiting (for a load, a session, the model's lock)
// because its context ended. It wraps the ctx error: the server tells it from a failure with
// errors.Is(err, ctx.Err()).
func gaveUp(name string, ctxErr error) error {
	return fmt.Errorf("lifecycle: %q: the request is gone, nothing was run for it: %w", name, ctxErr)
}

// Predict runs the full pipeline: preprocess → infer (ORT) → postprocess (simple), or
// model-driven multi-stage inference (pipeline). prompt is ignored by simple models. ctx bounds
// every wait for a session (see engine.Runnable); an inference already running is not interrupted.
func (s *Session) Predict(ctx context.Context, img image.Image, prompt models.Prompt, now time.Time) (api.Result, error) {
	start := now

	var (
		res api.Result
		err error
	)
	if s.pipeline != nil {
		res, err = s.inferPipeline(ctx, img, prompt)
	} else {
		res, err = s.predictSimple(ctx, img)
	}
	if err != nil {
		return api.Result{}, err
	}

	s.touch(now)
	res.Model = s.name
	res.Task = s.task
	res.Device = s.device
	// Hint: on the CUDA EP, set only when TensorRT was opted into but libnvinfer is missing.
	if s.device == "gpu:0" {
		res.Hint = engine.TRTHint()
	}
	res.DurationMs = float64(time.Since(start).Microseconds()) / 1000.0
	return res, nil
}

// PredictTensor runs ORT on a caller-supplied, already-preprocessed input tensor, BYPASSING
// server-side image decode + preprocess (the "tensor-in" path). It exists for (a) clients that
// already hold a pixel/feature tensor in memory (no JPEG/PNG round-trip), and (b) benchmarking
// the serving+inference layer without the image-decode cost, comparable to Triton's tensor-in API.
//
// Simple (single-session) models only. Any Detection coordinates are in MODEL-INPUT space — there
// is no original image to map back to — so the tensor-in path is intended for classification /
// embedding (no coordinate mapping) or callers that map coordinates themselves.
func (s *Session) PredictTensor(ctx context.Context, in engine.Tensor, now time.Time) (api.Result, error) {
	if s.pipeline != nil {
		return api.Result{}, fmt.Errorf("%w: tensor-in not supported for multi-session model %q", ErrInvalidRequest, s.name)
	}
	start := now
	outs, err := s.engine.Run(ctx, []engine.Tensor{in})
	if err != nil {
		return api.Result{}, err
	}
	// Identity preprocess meta (no original image). W,H come from an NCHW shape if present.
	w, h := 0, 0
	if len(in.Shape) == 4 {
		h, w = int(in.Shape[2]), int(in.Shape[3])
	}
	res, err := s.model.Postprocess(outs, models.PreprocessMeta{
		OrigWidth: w, OrigHeight: h, ScaleX: 1, ScaleY: 1,
	})
	if err != nil {
		return api.Result{}, err
	}
	s.touch(now)
	res.Model = s.name
	res.Task = s.task
	res.Device = s.device
	res.DurationMs = float64(time.Since(start).Microseconds()) / 1000.0
	return res, nil
}

// predictSimple is the classic single-session pre→infer→post path.
func (s *Session) predictSimple(ctx context.Context, img image.Image) (api.Result, error) {
	in, meta, err := s.model.Preprocess(img)
	if err != nil {
		return api.Result{}, err
	}
	outs, err := s.engine.Run(ctx, []engine.Tensor{in})
	if err != nil {
		return api.Result{}, err
	}
	return s.model.Postprocess(outs, meta)
}

// slots is how many requests this model can run at once: the largest session pool among its
// roles (1 for single sessions, and for an Exclusive model whatever its pools). Admission control
// bounds each model at a multiple of it.
func (s *Session) slots() int {
	if s.exclusive {
		return 1
	}
	n := runnableSlots(s.engine)
	for _, e := range s.engines {
		n = max(n, runnableSlots(e))
	}
	return n
}

// runnableSlots is the concurrency of one session or pool (engine.SessionPool reports Size).
func runnableSlots(r engine.Runnable) int {
	if p, ok := r.(interface{ Size() int }); ok && p.Size() > 1 {
		return p.Size()
	}
	return 1
}

// runner is the lifecycle-backed implementation of models.Runner: it exposes the
// loaded sessions to a PipelineModel by role, without giving away ownership. ctx is the request's:
// each call waits for its session only while the request is still wanted.
type runner struct {
	ctx     context.Context
	engines map[string]engine.Runnable
}

func (r runner) Run(role string, inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
	s, ok := r.engines[role]
	if !ok {
		return nil, fmt.Errorf("lifecycle: no ONNX session for role %q", role)
	}
	return s.RunNamed(r.ctx, inputs)
}

func (r runner) InputNames(role string) []string {
	if s, ok := r.engines[role]; ok {
		return s.InputNames()
	}
	return nil
}

func (r runner) OutputNames(role string) []string {
	if s, ok := r.engines[role]; ok {
		return s.OutputNames()
	}
	return nil
}

func (s *Session) touch(now time.Time) {
	s.mu.Lock()
	s.lastUsed = now
	s.mu.Unlock()
}

// idleFor returns how long the model has been idle up to now.
func (s *Session) idleFor(now time.Time) time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return now.Sub(s.lastUsed)
}

// ExplainEngine returns the lazily-created explain session (nil if not yet loaded).
func (s *Session) ExplainEngine() engine.Runnable {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.explainEngine
}

// SetExplainEngine stores the explain session if none is set yet and reports whether it did;
// the loser of a concurrent lazy-init closes its own copy.
func (s *Session) SetExplainEngine(r engine.Runnable) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.explainEngine != nil {
		return false
	}
	s.explainEngine = r
	return true
}

// close releases the ONNX session(s) (avoid VRAM leaks).
func (s *Session) close() error {
	var firstErr error
	if s.engine != nil {
		if err := s.engine.Close(); err != nil {
			firstErr = err
		}
	}
	if ex := s.ExplainEngine(); ex != nil {
		if err := ex.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	for _, e := range s.engines {
		if err := e.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
