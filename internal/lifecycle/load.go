package lifecycle

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/pkg/api"
)

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
			run, err := newRunnable(path, nil, nil, poolSizes[role], providers)
			if err != nil {
				closeEngines(engines)
				return err
			}
			engines[role] = run
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

		// VS_POOL_OVERRIDE>1 wraps N identical sessions in a pool so a single-session
		// (classification/detection) model can serve inferences concurrently (eval sweep).
		run, err := newRunnable(man.ModelFilePath(), inName, detectOutNames, poolOverride(), providers)
		if err != nil {
			return err
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

// newRunnable creates the ONNX session(s) for one weights file: ONE session when n <= 1, otherwise
// a SessionPool of n identical sessions (n concurrent inferences instead of one at a time). If any
// session fails, every session it already created is closed before the error is returned, so a
// failed load never strands VRAM.
func newRunnable(path string, inputNames, outputNames []string, n int, providers []engine.Provider) (engine.Runnable, error) {
	if n <= 1 {
		s, err := engine.NewSession(path, inputNames, outputNames, providers)
		if err != nil {
			return nil, err
		}
		return s, nil
	}
	sessions := make([]*engine.Session, 0, n)
	for i := 0; i < n; i++ {
		s, err := engine.NewSession(path, inputNames, outputNames, providers)
		if err != nil {
			for _, c := range sessions {
				_ = c.Close()
			}
			return nil, err
		}
		sessions = append(sessions, s)
	}
	return engine.NewSessionPool(sessions), nil
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
