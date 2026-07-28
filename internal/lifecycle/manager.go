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
	m.mu.Lock()
	if _, ok := m.live[name]; ok {
		m.mu.Unlock()
		return nil
	}
	m.mu.Unlock()

	entry, ok := m.reg.Get(name)
	if !ok {
		// Model not in registry — it may have been pulled while the server was
		// running. Re-scan once before giving up.
		m.reg.Scan() //nolint:errcheck — scan errors are non-fatal (logged at startup)
		entry, ok = m.reg.Get(name)
	}
	if !ok {
		return fmt.Errorf("lifecycle: model %q not found in registry", name)
	}
	man := entry.Manifest

	if !man.WeightsExist() {
		return fmt.Errorf("lifecycle: no weights for %q at %s — download them per the README in the model directory", name, man.ModelFilePath())
	}

	// License-policy hardening: when the manifest pins a sha256 (and/or a verified
	// source allowlist is configured), bind the declared license to the audited
	// bytes/origin. A no-op for manifests that declare neither (backward compatible).
	if err := man.VerifyWeights(); err != nil {
		return fmt.Errorf("lifecycle: weight verification failed for %q: %w", name, err)
	}

	labels, err := man.LoadLabels()
	if err != nil {
		return err
	}
	providers, err := man.Providers()
	if err != nil {
		return err
	}

	cfg := models.Config{
		Name:       man.Name,
		Width:      man.Input.Width,
		Height:     man.Input.Height,
		Layout:     man.Input.Layout,
		Mean:       man.Input.Normalize.Mean,
		Std:        man.Input.Normalize.Std,
		Letterbox:  man.Input.Letterbox,
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
		poolSizes := map[string]int{}
		if ps, ok := mdl.(models.PoolSizer); ok {
			poolSizes = ps.PoolSizes()
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

// Unload releases a model from memory. Not an error if the model is not loaded.
func (m *Manager) Unload(name string) error {
	m.mu.Lock()
	s, ok := m.live[name]
	if ok {
		delete(m.live, name)
	}
	m.mu.Unlock()
	if !ok {
		return nil
	}
	return s.close()
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
	m.mu.Lock()
	s := m.live[name]
	m.mu.Unlock()
	if s == nil {
		return api.Result{}, fmt.Errorf("lifecycle: model %q was just unloaded", name)
	}
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
	m.mu.Lock()
	s := m.live[name]
	m.mu.Unlock()
	if s == nil {
		return api.Result{}, fmt.Errorf("lifecycle: model %q was just unloaded", name)
	}
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
	m.mu.Lock()
	s := m.live[name]
	m.mu.Unlock()
	if s == nil {
		return fmt.Errorf("lifecycle: model %q not loaded", name)
	}
	if s.ExplainEngine() != nil {
		return nil // already created
	}

	entry, ok := m.reg.Get(name)
	if !ok {
		return fmt.Errorf("lifecycle: model %q not in registry", name)
	}
	man := entry.Manifest

	// Pipeline models do not support explain sessions at this time.
	if s.pipeline != nil {
		return fmt.Errorf("model %q is a pipeline model — explain session is not supported for pipeline models", name)
	}

	if man.Explain == nil {
		return fmt.Errorf("model %q does not support explain (no explain block in manifest)", name)
	}

	providers, err := man.Providers()
	if err != nil {
		return err
	}

	// Create session with ALL outputs (detect + explain).
	explainEng, err := engine.NewSession(man.ModelFilePath(), nil, nil, providers)
	if err != nil {
		return fmt.Errorf("lifecycle: failed to create explain session for %q: %w", name, err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	// Re-check under lock (another goroutine may have won the race).
	if s2 := m.live[name]; s2 != nil && s2.ExplainEngine() == nil {
		s2.SetExplainEngine(explainEng)
	} else {
		_ = explainEng.Close() // lost the race, discard
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

	m.mu.Lock()
	s := m.live[name]
	m.mu.Unlock()
	if s == nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: model %q was just unloaded", name)
	}

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

	// Simple models only (pipeline models are rejected by loadExplainSession above).
	mdl := s.model
	if mdl == nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: model %q has no simple Model — explain requires a plain Model", name)
	}

	// Preprocess the image using the model's own preprocess logic.
	inputTensor, meta, err := mdl.Preprocess(img)
	if err != nil {
		return ExplainResult{}, fmt.Errorf("lifecycle: preprocess for explain failed: %w", err)
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

	// Resolve the detection index: if Class is given, run a detect inference to
	// find the first detection matching that class, then use its index.
	detectionIdx := req.DetectionIdx
	if req.Class != "" {
		detectOuts, derr := s.engine.Run([]engine.Tensor{inputTensor})
		if derr == nil {
			res, derr2 := mdl.Postprocess(detectOuts, meta)
			if derr2 == nil {
				for i, d := range res.Detections {
					if d.Class == req.Class {
						detectionIdx = i
						break
					}
				}
			}
		}
		// If class lookup fails we fall back to req.DetectionIdx (already set above).
	}

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

		detectRunner := func(masked image.Image) (float32, error) {
			in, meta2, err2 := mdl.Preprocess(masked)
			if err2 != nil {
				return 0, err2
			}
			outs, err2 := s.engine.Run([]engine.Tensor{in})
			if err2 != nil {
				return 0, err2
			}
			res, err2 := mdl.Postprocess(outs, meta2)
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
	defer m.mu.Unlock()
	for name, s := range m.live {
		_ = s.close()
		delete(m.live, name)
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
			var expired []string
			m.mu.Lock()
			for name, s := range m.live {
				if s.idleTimeout > 0 && s.idleFor(now) > s.idleTimeout {
					expired = append(expired, name)
				}
			}
			for _, name := range expired {
				if s := m.live[name]; s != nil {
					_ = s.close()
					delete(m.live, name)
				}
			}
			m.mu.Unlock()
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
