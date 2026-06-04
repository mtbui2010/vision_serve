package lifecycle

import (
	"fmt"
	"image"
	"sync"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
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
	device      string        // "gpu:0" or "cpu" — set from the active EP on load
	idleTimeout time.Duration

	model  models.Model    // simple mode
	engine *engine.Session // simple mode

	pipeline models.PipelineModel       // pipeline mode
	engines  map[string]*engine.Session // pipeline mode (role → session)

	mu       sync.Mutex
	lastUsed time.Time
}

func newSimpleSession(name string, task api.Task, m models.Model, eng *engine.Session, idle time.Duration, now time.Time) *Session {
	return &Session{
		name:        name,
		task:        task,
		device:      engine.DeviceString(eng.ActiveEP()),
		model:       m,
		engine:      eng,
		idleTimeout: idle,
		lastUsed:    now,
	}
}

func newPipelineSession(name string, task api.Task, p models.PipelineModel, engs map[string]*engine.Session, idle time.Duration, now time.Time) *Session {
	// Determine device from the first session in the map (all roles share the same EP).
	dev := "cpu"
	for _, e := range engs {
		dev = engine.DeviceString(e.ActiveEP())
		break
	}
	return &Session{
		name:        name,
		task:        task,
		device:      dev,
		pipeline:    p,
		engines:     engs,
		idleTimeout: idle,
		lastUsed:    now,
	}
}

// Predict runs the full pipeline: preprocess → infer (ORT) → postprocess (simple), or
// model-driven multi-stage inference (pipeline). prompt is ignored by simple models.
func (s *Session) Predict(img image.Image, prompt models.Prompt, now time.Time) (api.Result, error) {
	start := now

	var (
		res api.Result
		err error
	)
	if s.pipeline != nil {
		res, err = s.pipeline.Infer(img, prompt, runner{s.engines})
	} else {
		res, err = s.predictSimple(img)
	}
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
func (s *Session) predictSimple(img image.Image) (api.Result, error) {
	in, meta, err := s.model.Preprocess(img)
	if err != nil {
		return api.Result{}, err
	}
	outs, err := s.engine.Run([]engine.Tensor{in})
	if err != nil {
		return api.Result{}, err
	}
	return s.model.Postprocess(outs, meta)
}

// runner is the lifecycle-backed implementation of models.Runner: it exposes the
// loaded sessions to a PipelineModel by role, without giving away ownership.
type runner struct {
	engines map[string]*engine.Session
}

func (r runner) Run(role string, inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
	s, ok := r.engines[role]
	if !ok {
		return nil, fmt.Errorf("lifecycle: no ONNX session for role %q", role)
	}
	return s.RunNamed(inputs)
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

// close releases the ONNX session(s) (avoid VRAM leaks).
func (s *Session) close() error {
	if s.engine != nil {
		return s.engine.Close()
	}
	var firstErr error
	for _, e := range s.engines {
		if err := e.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
