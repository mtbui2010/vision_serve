// Package lifecycle manages the in-memory model lifecycle: lazy load, running multiple
// models concurrently, and auto-unload after an idle period.
//
// Every ONNX session MUST go through the Manager (CLAUDE.md) — never created directly in a handler.
//
// Files:
//   - manager.go: the Manager, its constructor and the request entrypoints (Predict*, Close).
//   - admission.go: Admit, the per-model bound on requests running + waiting.
//   - load.go:    Load (singleflight), building a model from its manifest, creating sessions.
//   - lease.go:   acquire/release of a live session, Unload, retiring a session.
//   - reaper.go:  the idle auto-unload loop.
//   - explain.go: Score-CAM / attention heatmaps.
//   - session.go: one loaded model (its sessions + pre/postprocess).
package lifecycle

import (
	"fmt"
	"image"
	"sync"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/internal/templates"
	"visionserve/pkg/api"
)

// Manager holds the live models and coordinates thread-safe load/unload.
type Manager struct {
	reg  *registry.Registry
	tmpl *templates.Store // nil = no template support

	mu   sync.Mutex
	live map[string]*Session
	// loading holds the in-progress load of each model being loaded (see Load).
	loading    map[string]*loadCall
	lastRescan time.Time // last on-demand registry rescan (see rescanAllowed)
	// closed is set by Close: nothing may load afterwards.
	closed bool

	// admitted counts the requests each model currently holds (running + waiting); maxQueue is the
	// configured bound: queueAuto, queueUnbounded or a positive limit. See admission.go.
	admitted map[string]int
	maxQueue int

	// idleOverrideSec, when >= 0, overrides every model's manifest
	// idle_unload_seconds at Load time (0 = never auto-unload). -1 keeps the
	// per-manifest value. Set once at startup via SetIdleUnloadOverride.
	idleOverrideSec int

	// openRunnable creates the ONNX session(s) for one weights file; nil means newRunnable. Tests
	// replace it to load models without ONNX Runtime.
	openRunnable func(path string, inputNames, outputNames []string, n int, providers []engine.Provider) (engine.Runnable, error)

	stop chan struct{}
	once sync.Once
}

// NewManager creates a manager bound to a registry and starts the idle reaper.
func NewManager(reg *registry.Registry) *Manager {
	m := &Manager{
		reg:             reg,
		live:            map[string]*Session{},
		loading:         map[string]*loadCall{},
		idleOverrideSec: -1, // -1 = use each manifest's idle_unload_seconds
		admitted:        map[string]int{},
		maxQueue:        maxQueueFromEnv(),
		stop:            make(chan struct{}),
	}
	go m.reaper()
	return m
}

// SetTemplateStore wires a template store into the manager so PredictPrompt can resolve
// TemplateName to TemplateImages before invoking instance_detection models.
// Call once at startup (before serving any requests).
func (m *Manager) SetTemplateStore(tmpl *templates.Store) {
	m.tmpl = tmpl
}

// SetIdleUnloadOverride overrides every model's manifest idle_unload_seconds.
// sec >= 0 applies to ALL models (0 = never auto-unload — models stay resident so
// the first request after a long idle pause is not slowed by a reload); sec < 0
// restores the per-manifest value. Call once at startup, before loading any model.
func (m *Manager) SetIdleUnloadOverride(sec int) {
	m.idleOverrideSec = sec
}

// Predict ensures the model is loaded then runs inference (no prompt). Back-compat
// entrypoint for plain models like rf-detr.
func (m *Manager) Predict(name string, img image.Image) (api.Result, error) {
	return m.PredictPrompt(name, img, models.Prompt{})
}

// PredictPrompt ensures the model is loaded then runs inference with a prompt.
// Plain models ignore the prompt; prompted models (SAM, GroundingDINO, Grounded-SAM)
// use it. This is the entrypoint for handlers/CLI.
func (m *Manager) PredictPrompt(name string, img image.Image, prompt models.Prompt) (api.Result, error) {
	if err := m.Load(name); err != nil {
		return api.Result{}, err
	}
	s, release, err := m.acquire(name)
	if err != nil {
		return api.Result{}, err
	}
	defer release()
	if err := m.resolveTemplates(&prompt); err != nil {
		return api.Result{}, err
	}
	return s.Predict(img, prompt, time.Now())
}

// resolveTemplates fills prompt.TemplateImages from the registered set prompt.TemplateName
// (instance_detection models), for predict and preprocess alike.
func (m *Manager) resolveTemplates(prompt *models.Prompt) error {
	if prompt.TemplateName == "" || m.tmpl == nil {
		return nil
	}
	imgs := m.tmpl.Get(prompt.TemplateName)
	if len(imgs) == 0 {
		return fmt.Errorf("lifecycle: %w: template %q not found — register via POST /api/templates",
			ErrInvalidRequest, prompt.TemplateName)
	}
	prompt.TemplateImages = imgs
	return nil
}

// InferTensor ensures the model is loaded then runs the tensor-in path (no decode/preprocess).
// See Session.PredictTensor. Simple models only.
func (m *Manager) InferTensor(name string, in engine.Tensor) (api.Result, error) {
	if err := m.Load(name); err != nil {
		return api.Result{}, err
	}
	s, release, err := m.acquire(name)
	if err != nil {
		return api.Result{}, err
	}
	defer release()
	return s.PredictTensor(in, time.Now())
}

// Loaded returns the names of the models currently in memory.
func (m *Manager) Loaded() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, 0, len(m.live))
	for k := range m.live {
		out = append(out, k)
	}
	return out
}

// IsLoaded reports whether a model is currently in memory.
func (m *Manager) IsLoaded(name string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	_, ok := m.live[name]
	return ok
}

// Close stops the reaper and releases all models (called on server shutdown). A load still in
// progress is cancelled: its sessions are closed when it finishes, and no new load starts.
func (m *Manager) Close() {
	m.once.Do(func() { close(m.stop) })
	m.mu.Lock()
	m.closed = true
	for _, call := range m.loading {
		call.cancelled = true
	}
	var idle []*Session
	for name := range m.live {
		if s := m.retireLocked(name); s != nil {
			idle = append(idle, s)
		}
	}
	m.mu.Unlock()
	for _, s := range idle { // sessions still serving a request are closed by their last release
		_ = s.close()
	}
}
