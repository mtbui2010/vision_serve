// Package lifecycle manages the in-memory model lifecycle: lazy load, running multiple
// models concurrently, and auto-unload after an idle period.
//
// Every ONNX session MUST go through the Manager (CLAUDE.md) — never created directly in a handler.
package lifecycle

import (
	"fmt"
	"image"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/explain"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/internal/templates"
	"visionserve/pkg/api"
)

// reaperInterval is the interval for scanning idle models to auto-unload.
const reaperInterval = 30 * time.Second

// Manager holds the live models and coordinates thread-safe load/unload.
type Manager struct {
	reg  *registry.Registry
	tmpl *templates.Store // nil = no template support

	mu   sync.Mutex
	live map[string]*Session
	// loading holds one channel per model currently being loaded; it is closed when that load
	// finishes, successfully or not (see Load).
	loading    map[string]chan struct{}
	lastRescan time.Time // last on-demand registry rescan (see rescanAllowed)

	// idleOverrideSec, when >= 0, overrides every model's manifest
	// idle_unload_seconds at Load time (0 = never auto-unload). -1 keeps the
	// per-manifest value. Set once at startup via SetIdleUnloadOverride.
	idleOverrideSec int

	stop chan struct{}
	once sync.Once
}

// NewManager creates a manager bound to a registry and starts the idle reaper.
func NewManager(reg *registry.Registry) *Manager {
	m := &Manager{
		reg:             reg,
		live:            map[string]*Session{},
		loading:         map[string]chan struct{}{},
		idleOverrideSec: -1, // -1 = use each manifest's idle_unload_seconds
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

// Load loads a model into memory (idempotent: returns immediately if already loaded).
// This is where models.Model is built from the manifest and engine.Session is created.
func (m *Manager) Load(name string) error {
	// One load per name at a time (singleflight). Two first requests used to both hash the
	// weights and build every ONNX session — twice the VRAM at peak — and throw one copy away.
	// Waiters re-check after the leader finishes; if it failed, the next one retries the load.
	for {
		m.mu.Lock()
		if _, ok := m.live[name]; ok {
			m.mu.Unlock()
			return nil
		}
		wait, busy := m.loading[name]
		if !busy {
			done := make(chan struct{})
			m.loading[name] = done
			m.mu.Unlock()
			err := m.load(name)
			m.mu.Lock()
			delete(m.loading, name)
			close(done)
			m.mu.Unlock()
			return err
		}
		m.mu.Unlock()
		<-wait
	}
}

// load builds the model and its sessions; only Load (which guarantees one per name) calls it.
func (m *Manager) load(name string) error {
	base, man, err := m.buildModel(name)
	if err != nil {
		return err
	}
	providers, err := man.Providers()
	if err != nil {
		return err
	}

	idleSec := man.Runtime.IdleUnloadSeconds
	if m.idleOverrideSec >= 0 {
		idleSec = m.idleOverrideSec // global --idle-unload-seconds override (0 = never)
	}
	idle := time.Duration(idleSec) * time.Second
	now := time.Now()
	task := api.Task(man.Task)

	// Build the appropriate session kind (heavy ONNX sessions all created here so
	// lifecycle owns them — CLAUDE.md: sessions must go through the Manager).
	var sess *Session
	switch mdl := base.(type) {
	case models.PipelineModel:
		filesAbs := man.FilesAbs()
		if len(filesAbs) == 0 {
			return fmt.Errorf("lifecycle: model %q is multi-session but its manifest has no 'files' map", name)
		}
		// Collect per-role pool sizes (default 1 = single session).
		// Copied, never aliased: a model may return nil (hybrid without SAM) or its own map, and
		// the override below writes into this one — a nil map write panicked the load.
		poolSizes := map[string]int{}
		if ps, ok := mdl.(models.PoolSizer); ok {
			for role, n := range ps.PoolSizes() {
				poolSizes[role] = n
			}
		}
		// VS_POOL_OVERRIDE forces every role's pool size (eval pool×concurrency sweep).
		if ov := poolOverride(); ov > 0 {
			for _, role := range mdl.Roles() {
				poolSizes[role] = ov
			}
		}
		engines := map[string]engine.Runnable{}
		for _, role := range mdl.Roles() {
			path, ok := filesAbs[role]
			if !ok {
				closeEngines(engines)
				return fmt.Errorf("lifecycle: role %q of model %q not found in manifest 'files'", role, name)
			}
			n := poolSizes[role]
			if n <= 1 {
				es, err := engine.NewSession(path, nil, nil, providers)
				if err != nil {
					closeEngines(engines)
					return err
				}
				engines[role] = es
			} else {
				// Session pool: n identical sessions for concurrent inference.
				sessions := make([]*engine.Session, n)
				for i := range sessions {
					s, err := engine.NewSession(path, nil, nil, providers)
					if err != nil {
						for j := 0; j < i; j++ {
							_ = sessions[j].Close()
						}
						closeEngines(engines)
						return err
					}
					sessions[i] = s
				}
				engines[role] = engine.NewSessionPool(sessions)
			}
		}
		sess = newPipelineSession(man.Name, task, mdl, engines, idle, now)
	case models.Model:
		inName, outNames := nilIfEmpty(mdl.InputName()), mdl.OutputNames()

		// Determine detect-only output names when explain is configured.
		// This ensures the detect session never computes explain tensors.
		detectOutNames := outNames
		if man.Explain != nil && len(outNames) == 0 {
			_, probed, err2 := engine.Inspect(man.ModelFilePath())
			if err2 == nil {
				explainNames := man.Explain.ExplainOutputNames()
				filtered := make([]string, 0, len(probed))
				for _, info := range probed {
					if !explainNames[info.Name] {
						filtered = append(filtered, info.Name)
					}
				}
				if len(filtered) > 0 {
					detectOutNames = filtered
				}
			}
			// if Inspect fails, fall back to auto-probe (minor overhead, not fatal)
		}

		var run engine.Runnable
		if ov := poolOverride(); ov > 1 {
			// VS_POOL_OVERRIDE>1: wrap N identical sessions in a pool so a single-session
			// (classification/detection) model can serve inferences concurrently (eval sweep).
			sessions := make([]*engine.Session, ov)
			for i := range sessions {
				s, err := engine.NewSession(man.ModelFilePath(), inName, detectOutNames, providers)
				if err != nil {
					for j := 0; j < i; j++ {
						_ = sessions[j].Close()
					}
					return err
				}
				sessions[i] = s
			}
			run = engine.NewSessionPool(sessions)
		} else {
			eng, err := engine.NewSession(man.ModelFilePath(), inName, detectOutNames, providers)
			if err != nil {
				return err
			}
			run = eng
		}
		sess = newSimpleSession(man.Name, task, mdl, run, idle, now)
	default:
		return fmt.Errorf("lifecycle: model %q implements neither Model nor PipelineModel", name)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	// Re-check: another request may have finished loading while we built ours.
	if _, ok := m.live[name]; ok {
		_ = sess.close() // drop the extra one we just built
		return nil
	}
	m.live[name] = sess
	return nil
}

// closeEngines releases a partially-built set of sessions/pools on a load error.
func closeEngines(engines map[string]engine.Runnable) {
	for _, e := range engines {
		_ = e.Close()
	}
}

// buildModel resolves a registry entry, runs the load-time checks (weights present, verified
// weights, labels, EP chain) and constructs the model object from its manifest. It creates NO
// ONNX session: Load adds those, and Preprocess uses the bare object to show what the model
// would be fed.
func (m *Manager) buildModel(name string) (models.Base, *registry.Manifest, error) {
	entry, ok := m.reg.Get(name)
	if !ok && m.rescanAllowed() {
		// Model not in registry — it may have been pulled while the server was running.
		// Re-scan before giving up, at most once per second: every unknown name used to
		// trigger a full scan (ReadDir + ~50 YAML parses), so a client spamming random
		// names could burn CPU at will.
		m.reg.Scan() //nolint:errcheck — scan errors are non-fatal (logged at startup)
		entry, ok = m.reg.Get(name)
	}
	if !ok {
		return nil, nil, fmt.Errorf("lifecycle: model %q not found in registry", name)
	}
	man := entry.Manifest

	if !man.WeightsExist() {
		return nil, nil, fmt.Errorf("lifecycle: no weights for %q at %s — download them per the README in the model directory", name, man.ModelFilePath())
	}

	// License-policy hardening: when the manifest pins a sha256 (and/or a verified
	// source allowlist is configured), bind the declared license to the audited
	// bytes/origin. A no-op for manifests that declare neither (backward compatible).
	if err := man.VerifyWeights(); err != nil {
		return nil, nil, fmt.Errorf("lifecycle: weight verification failed for %q: %w", name, err)
	}

	if err := checkKeepAspectGraph(man); err != nil {
		return nil, nil, err
	}

	labels, err := man.LoadLabels()
	if err != nil {
		return nil, nil, err
	}
	if _, err := man.Providers(); err != nil { // validate the EP chain at the same point Load always did
		return nil, nil, err
	}

	cfg := models.Config{
		Name:       man.Name,
		Width:      man.Input.Width,
		Height:     man.Input.Height,
		Layout:     man.Input.Layout,
		Mean:       man.Input.Normalize.Mean,
		Std:        man.Input.Normalize.Std,
		Letterbox:  man.Input.Letterbox,
		Crop:       man.Input.Crop,
		KeepAspect: man.Input.KeepAspect,
		MultipleOf: man.Input.MultipleOf,
		PostType:   man.Postprocess.Type,
		BoxFormat:  man.Postprocess.BoxFormat,
		ConfThresh: man.Postprocess.ConfThreshold,
		TextThresh: man.Postprocess.TextThreshold,
		MaxDet:     man.Postprocess.MaxDetections,
		Labels:     labels,
		Dir:        man.Dir(),
		Files:      man.FilesAbs(),
		Detector:   man.Detector,
		Segmenter:  man.Segmenter,
		GripperMin: man.Grasp.GripperMin,
		GripperMax: man.Grasp.GripperMax,
	}
	if man.Instance != nil {
		cfg.InstanceSimThreshold = man.Instance.SimThreshold
		cfg.InstanceMaxTemplates = man.Instance.MaxTemplates
		cfg.InstancePatchSize = man.Instance.PatchSize
	}

	base, err := models.New(man.ArchOrName(), cfg)
	if err != nil {
		return nil, nil, err
	}
	return base, man, nil
}

// Unload releases a model from memory. Not an error if the model is not loaded.
func (m *Manager) Unload(name string) error {
	m.mu.Lock()
	s := m.retireLocked(name)
	m.mu.Unlock()
	if s == nil {
		return nil // not loaded, or still in use: the last request closes it
	}
	return s.close()
}

// acquire leases the live session for name: it cannot be closed (by Unload, the idle reaper or
// Close) until the returned release runs. Without the lease a request could keep using a
// session that was closed under it — a "send on closed channel" panic, or a hang on a pool.
// It also marks the session used NOW, so a request arriving at the edge of the idle timeout is
// not reaped mid-flight.
func (m *Manager) acquire(name string) (*Session, func(), error) {
	m.mu.Lock()
	s := m.live[name]
	if s == nil {
		m.mu.Unlock()
		return nil, nil, fmt.Errorf("lifecycle: model %q was just unloaded", name)
	}
	s.refs++
	m.mu.Unlock()
	s.touch(time.Now())
	return s, func() { m.release(s) }, nil
}

func (m *Manager) release(s *Session) {
	m.mu.Lock()
	s.refs--
	closeNow := s.retired && s.refs == 0
	m.mu.Unlock()
	if closeNow {
		_ = s.close()
	}
}

// retireLocked removes name from the live map (caller holds m.mu). It returns the session when
// nobody is using it — the caller must then close it, OUTSIDE m.mu, since closing a GPU session
// can take a while and m.mu gates every request. A session still in use is left to the last
// release to close.
func (m *Manager) retireLocked(name string) *Session {
	s := m.live[name]
	if s == nil {
		return nil
	}
	delete(m.live, name)
	s.retired = true
	if s.refs > 0 {
		return nil
	}
	return s
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
	// Resolve template images for instance_detection models.
	if prompt.TemplateName != "" && m.tmpl != nil {
		imgs := m.tmpl.Get(prompt.TemplateName)
		if len(imgs) == 0 {
			return api.Result{}, fmt.Errorf("lifecycle: template %q not found — register via POST /api/templates", prompt.TemplateName)
		}
		prompt.TemplateImages = imgs
	}
	return s.Predict(img, prompt, time.Now())
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

// ExplainRequest carries explain parameters from the HTTP handler.
type ExplainRequest struct {
	Class        string  // filter by class name (e.g. "cup"); empty = use DetectionIdx
	DetectionIdx int     // 0-based index; used when Class is empty (default 0)
	TopChannels  int     // Score-CAM: number of channels to sample (0 = use manifest default)
	Alpha        float32 // PNG overlay opacity [0,1] (0 = use default 0.5)
}

// ExplainResult holds the raw heatmap (H×W float32 in [0,1]).
type ExplainResult struct {
	Heatmap []float32 // row-major [H*W], values in [0,1]
	Width   int
	Height  int
}

// loadExplainSession lazily creates the explain session (all outputs including explain tensors).
// Thread-safe — multiple concurrent /api/explain calls race to create it; only one wins.
func (m *Manager) loadExplainSession(name string) error {
	s, release, err := m.acquire(name)
	if err != nil {
		return err
	}
	defer release()
	if s.ExplainEngine() != nil {
		return nil // already created
	}

	entry, ok := m.reg.Get(name)
	if !ok {
		return fmt.Errorf("lifecycle: model %q not in registry", name)
	}
	man := entry.Manifest

	if man.Explain == nil {
		return fmt.Errorf("model %q does not support explain (no explain block in manifest)", name)
	}

	providers, err := man.Providers()
	if err != nil {
		return err
	}

	var explainEng engine.Runnable

	if s.pipeline != nil {
		// PipelineModel: create explain session for the role that owns the explain outputs.
		if man.Explain.Role == "" {
			return fmt.Errorf("model %q is a pipeline model — explain block must set 'role' (e.g. role: rfdetr)", name)
		}
		filesAbs := man.FilesAbs()
		rolePath, ok := filesAbs[man.Explain.Role]
		if !ok {
			return fmt.Errorf("lifecycle: explain role %q not in files map for model %q", man.Explain.Role, name)
		}
		explainEng, err = engine.NewSession(rolePath, nil, nil, providers)
		if err != nil {
			return fmt.Errorf("lifecycle: failed to create explain session for role %q in %q: %w", man.Explain.Role, name, err)
		}
	} else {
		// Plain Model: create session with ALL outputs (detect + explain tensors).
		explainEng, err = engine.NewSession(man.ModelFilePath(), nil, nil, providers)
		if err != nil {
			return fmt.Errorf("lifecycle: failed to create explain session for %q: %w", name, err)
		}
	}

	if !s.SetExplainEngine(explainEng) {
		_ = explainEng.Close() // another request won the race, discard ours
	}
	return nil
}

// Explain runs heatmap inference for the named model.
// Returns raw float32 heatmap; rendering (PNG / numpy response) is done by the handler.
func (m *Manager) Explain(name string, img image.Image, req ExplainRequest) (ExplainResult, error) {
	// Ensure the model is loaded.
	if err := m.Load(name); err != nil {
		return ExplainResult{}, err
	}
	// Ensure explain session exists (lazy).
	if err := m.loadExplainSession(name); err != nil {
		return ExplainResult{}, err
	}

	s, release, err := m.acquire(name)
	if err != nil {
		return ExplainResult{}, err
	}
	defer release()

	entry, ok := m.reg.Get(name)
	if !ok {
		return ExplainResult{}, fmt.Errorf("lifecycle: model %q not in registry", name)
	}
	man := entry.Manifest

	// Build the Explainer from the manifest config.
	exp, err := explain.New(man.Explain)
	if err != nil {
		return ExplainResult{}, err
	}

	// Preprocess: plain Model uses its own Preprocess(); PipelineModel uses ExplainPreprocessor.
	var inputTensor engine.Tensor
	var meta models.PreprocessMeta
	detectionIdx := req.DetectionIdx

	if s.pipeline != nil {
		ep, ok := s.pipeline.(models.ExplainPreprocessor)
		if !ok {
			return ExplainResult{}, fmt.Errorf(
				"lifecycle: pipeline model %q does not implement ExplainPreprocessor", name)
		}
		inputTensor, meta, err = ep.ExplainPreprocess(img)
		if err != nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: preprocess for explain failed: %w", err)
		}
		// Class-based detection index not supported for pipeline models; use req.DetectionIdx.
	} else {
		mdl := s.model
		if mdl == nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: model %q has no simple Model", name)
		}
		inputTensor, meta, err = mdl.Preprocess(img)
		if err != nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: preprocess for explain failed: %w", err)
		}
		// Resolve class → detectionIdx (plain models only).
		if req.Class != "" {
			// A class that is not detected is an error, not "explain detection 0": that used to
			// return a heatmap for some other object, labelled as the requested class.
			detectOuts, derr := s.engine.Run([]engine.Tensor{inputTensor})
			if derr != nil {
				return ExplainResult{}, fmt.Errorf("lifecycle: explain: detection pass failed: %w", derr)
			}
			res, derr := mdl.Postprocess(detectOuts, meta)
			if derr != nil {
				return ExplainResult{}, fmt.Errorf("lifecycle: explain: %w", derr)
			}
			found := false
			for i, d := range res.Detections {
				if d.Class == req.Class {
					detectionIdx, found = i, true
					break
				}
			}
			if !found {
				return ExplainResult{}, fmt.Errorf("lifecycle: explain: no %q detection in this image", req.Class)
			}
		}
	}

	// Run the explain session (all outputs: detect + explain tensors).
	explainEng := s.ExplainEngine()
	outputs, err := explainEng.Run([]engine.Tensor{inputTensor})
	if err != nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: explain session inference failed: %w", err)
	}
	outputNames := explainEng.OutputNames()

	origW := meta.OrigWidth
	origH := meta.OrigHeight

	// Override topChannels in the explain config for this request.
	if req.TopChannels > 0 && man.Explain.TopChannels != req.TopChannels {
		// Build a temporary config copy with the per-request top_channels.
		cfgCopy := *man.Explain
		cfgCopy.TopChannels = req.TopChannels
		exp, err = explain.New(&cfgCopy)
		if err != nil {
			return ExplainResult{}, err
		}
	}

	var heatmap []float32
	var W, H int

	if man.Explain.Type == "score_cam" {
		// Full Score-CAM: re-run detect session once per top-K channel with a masked image.
		// detectRunner is a closure that captures the lifecycle-owned detect session —
		// this avoids any circular import between internal/explain and internal/lifecycle.
		featName := man.Explain.Outputs["features"]
		var featTensor engine.Tensor
		for i, n := range outputNames {
			if n == featName && i < len(outputs) {
				featTensor = outputs[i]
				break
			}
		}
		if featTensor.Shape == nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: feature tensor %q not found in explain session outputs", featName)
		}

		topK := man.Explain.EffectiveTopChannels()
		if req.TopChannels > 0 {
			topK = req.TopChannels
		}

		plainMdl := s.model // Score-CAM requires a plain Model (pipeline models not supported)
		if plainMdl == nil {
			return ExplainResult{}, fmt.Errorf("lifecycle: score_cam explain requires a plain Model, not a pipeline")
		}
		detectRunner := func(masked image.Image) (float32, error) {
			in, meta2, err2 := plainMdl.Preprocess(masked)
			if err2 != nil {
				return 0, err2
			}
			outs, err2 := s.engine.Run([]engine.Tensor{in})
			if err2 != nil {
				return 0, err2
			}
			res, err2 := plainMdl.Postprocess(outs, meta2)
			if err2 != nil {
				return 0, err2
			}
			if detectionIdx < len(res.Detections) {
				return float32(res.Detections[detectionIdx].Conf), nil
			}
			return 0, nil
		}

		heatmap, W, H, err = explain.ScoreCAMHeatmap(featTensor, img, detectRunner, topK, origW, origH)
	} else {
		// Attention map: single inference already done, extract from outputs.
		heatmap, W, H, err = exp.Heatmap(outputs, outputNames, meta, detectionIdx, origW, origH)
	}

	if err != nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: heatmap computation failed: %w", err)
	}

	s.touch(time.Now())
	return ExplainResult{Heatmap: heatmap, Width: W, Height: H}, nil
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

// Close stops the reaper and releases all models (called on server shutdown).
func (m *Manager) Close() {
	m.once.Do(func() { close(m.stop) })
	m.mu.Lock()
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

// reaper periodically releases models idle longer than idleTimeout (idleTimeout<=0 = never).
func (m *Manager) reaper() {
	t := time.NewTicker(reaperInterval)
	defer t.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-t.C:
			now := time.Now()
			var expired []*Session
			m.mu.Lock()
			for name, s := range m.live {
				// A session serving a request is never idle, however old its lastUsed.
				if s.refs == 0 && s.idleTimeout > 0 && s.idleFor(now) > s.idleTimeout {
					if r := m.retireLocked(name); r != nil {
						expired = append(expired, r)
					}
				}
			}
			m.mu.Unlock()
			// Closed outside m.mu: destroying a GPU session takes time and m.mu gates every request.
			for _, s := range expired {
				_ = s.close()
			}
		}
	}
}

func nilIfEmpty(s string) []string {
	if s == "" {
		return nil
	}
	return []string{s}
}

// poolOverride reads VS_POOL_OVERRIDE (an integer >= 1) to force the per-role ONNX
// session-pool size at load, overriding a model's built-in PoolSizes (and giving
// single-session models a pool). It exists for the pool×concurrency evaluation sweep;
// unset or invalid returns 0 (no override — keep model defaults).
func poolOverride() int {
	v := strings.TrimSpace(os.Getenv("VS_POOL_OVERRIDE"))
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return 0
	}
	return n
}

// checkKeepAspectGraph refuses input.keep_aspect on a graph whose spatial input size is fixed.
// Keep-aspect feeds a different H×W for every image shape; a fixed graph would load fine and then
// fail on the first non-square request with an ONNX Runtime shape error naming no manifest field.
// An unreadable graph is not judged here: Load reports it on its own.
func checkKeepAspectGraph(man *registry.Manifest) error {
	if !man.Input.KeepAspect {
		return nil
	}
	ins, _, err := engine.Inspect(man.ModelFilePath())
	if err != nil || len(ins) == 0 {
		return nil
	}
	if shp := ins[0].Shape; len(shp) == 4 && shp[2] > 0 && shp[3] > 0 {
		return fmt.Errorf("lifecycle: %q declares input.keep_aspect, but %s has a fixed %dx%d input — "+
			"keep_aspect needs an export with dynamic height/width (or remove keep_aspect to squash)",
			man.Name, man.ModelFilePath(), shp[3], shp[2])
	}
	return nil
}

// rescanAllowed rate-limits on-demand registry rescans to one per second.
func (m *Manager) rescanAllowed() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	if now.Sub(m.lastRescan) < time.Second {
		return false
	}
	m.lastRescan = now
	return true
}
