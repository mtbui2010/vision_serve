package lifecycle

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/pkg/api"
)

// loadCall is one in-progress load of a model (see Load). done is closed when the load has
// finished, successfully or not; the other fields are guarded by Manager.mu.
type loadCall struct {
	done chan struct{}
	// cancelled is set by Unload (or Close) while the load runs: its session must not go live.
	cancelled bool
	// err is the load's result, readable once done is closed.
	err error
	// waiters counts the requests waiting on this load (diagnostics and tests).
	waiters int
}

// Load loads a model into memory (idempotent: returns immediately if already loaded).
// This is where models.Model is built from the manifest and engine.Session is created.
//
// ctx bounds only this caller's WAIT. The load itself runs on its own goroutine, owned by no
// request: when ctx ends first, Load returns an error wrapping ctx.Err() and the load carries on —
// the model goes live for the requests still waiting and for the next one (the idle reaper unloads
// it if nobody comes). Cancelling one waiter, the one that started the load included, never
// cancels the load for the others; only Unload and Close do.
func (m *Manager) Load(ctx context.Context, name string) error {
	// One load per name at a time (singleflight). Two first requests used to both hash the
	// weights and build every ONNX session — twice the VRAM at peak — and throw one copy away.
	// Waiters re-check after the load finishes; if it failed, the next one retries the load.
	for {
		if err := ctx.Err(); err != nil {
			return gaveUp(name, err)
		}
		m.mu.Lock()
		if m.closed {
			m.mu.Unlock()
			return fmt.Errorf("lifecycle: cannot load %q: the server is shutting down", name)
		}
		if _, ok := m.live[name]; ok {
			m.mu.Unlock()
			return nil
		}
		call, busy := m.loading[name]
		if !busy {
			call = &loadCall{done: make(chan struct{})}
			if m.loading == nil {
				m.loading = map[string]*loadCall{}
			}
			m.loading[name] = call
			m.mu.Unlock()
			go func() { _ = m.lead(name, call) }() // lead publishes its result in call.err
			if err := m.waitLoad(ctx, name, call); err != nil {
				return err
			}
			return call.err // this caller started the load: its result, success or failure
		}
		// A request that arrives after an Unload already cancelled this load did not ask for it:
		// it waits for the cancelled load to finish (two builds of one model must not overlap)
		// and then loads the model again.
		joinedCancelled := call.cancelled
		call.waiters++
		m.mu.Unlock()
		if err := m.waitLoad(ctx, name, call); err != nil {
			m.mu.Lock()
			call.waiters--
			m.mu.Unlock()
			return err
		}
		// Nothing writes call.err or call.cancelled after done is closed (it has left m.loading).
		if call.cancelled && !joinedCancelled {
			// Unloaded while it was loading. This request asked for THAT load; loading the model
			// again for it would undo the unload the moment it returned.
			return call.err
		}
	}
}

// waitLoad waits for call to finish, or for ctx to end (then it returns gaveUp, and the load goes
// on without this caller).
func (m *Manager) waitLoad(ctx context.Context, name string, call *loadCall) error {
	select {
	case <-call.done:
		return nil
	case <-ctx.Done():
		return gaveUp(name, ctx.Err())
	}
}

// lead runs a load, on its own goroutine (see Load), and publishes the result in call: the session
// goes live, unless an Unload or Close arrived meanwhile — then it is closed and the load fails.
// A panic while building (a model factory, a binding) becomes an error: it used to leave the
// name in m.loading forever, so every later request for that model hung.
func (m *Manager) lead(name string, call *loadCall) (err error) {
	var sess *Session
	defer func() {
		if r := recover(); r != nil {
			log.Printf("lifecycle: loading %q panicked: %v\n%s", name, r, debug.Stack())
			sess, err = nil, fmt.Errorf("lifecycle: loading %q panicked: %v", name, r)
		}
		var drop *Session
		m.mu.Lock()
		delete(m.loading, name)
		switch {
		case err != nil:
		case call.cancelled:
			drop = sess
			err = fmt.Errorf("lifecycle: model %q was unloaded while it was loading", name)
		case m.live[name] != nil: // cannot happen under singleflight; never replace a live session
			drop = sess
		default:
			m.live[name] = sess
		}
		call.err = err
		close(call.done)
		m.mu.Unlock()
		if drop != nil {
			_ = drop.close() // outside m.mu: closing a GPU session takes a while
		}
	}()
	sess, err = m.load(name)
	return err
}

// load builds the model and its sessions without publishing them; only lead calls it.
func (m *Manager) load(name string) (*Session, error) {
	base, man, err := m.buildChecked(name)
	if err != nil {
		return nil, err
	}
	providers, err := man.Providers()
	if err != nil {
		return nil, err
	}

	idleSec := man.Runtime.IdleUnloadSeconds
	if m.idleOverrideSec >= 0 {
		idleSec = m.idleOverrideSec // global --idle-unload-seconds override (0 = never)
	}
	idle := time.Duration(idleSec) * time.Second
	now := time.Now()
	task := api.Task(man.Task)

	open := m.openRunnable
	if open == nil {
		open = newRunnable
	}

	// Build the appropriate session kind (heavy ONNX sessions all created here so
	// lifecycle owns them — CLAUDE.md: sessions must go through the Manager).
	var sess *Session
	switch mdl := base.(type) {
	case models.PipelineModel:
		filesAbs := man.FilesAbs()
		if len(filesAbs) == 0 {
			return nil, fmt.Errorf("lifecycle: model %q is multi-session but its manifest has no 'files' map", name)
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
		built := false
		defer func() { // on an error or a panic, release the roles already opened
			if !built {
				closeEngines(engines)
			}
		}()
		for _, role := range mdl.Roles() {
			path, ok := filesAbs[role]
			if !ok {
				return nil, fmt.Errorf("lifecycle: role %q of model %q not found in manifest 'files'", role, name)
			}
			// runtime.threads overrides the default intra-op threads for this role's session(s).
			threads := -1
			if n, ok := man.IntraOpThreads(role); ok {
				threads = manifestThreads(man.Name, role, n, numCPU())
			}
			run, err := open(path, nil, nil, poolSizes[role], threads, providers)
			if err != nil {
				return nil, err
			}
			engines[role] = run
		}
		sess = newPipelineSession(man.Name, task, mdl, engines, idle, now)
		built = true
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
		run, err := open(man.ModelFilePath(), inName, detectOutNames, poolOverride(), -1, providers)
		if err != nil {
			return nil, err
		}
		sess = newSimpleSession(man.Name, task, mdl, run, idle, now)
	default:
		return nil, fmt.Errorf("lifecycle: model %q implements neither Model nor PipelineModel", name)
	}
	sess.man = man // snapshot: explain must describe what was loaded, not today's manifest on disk
	return sess, nil
}

// newRunnable creates the ONNX session(s) for one weights file: ONE session when n <= 1, otherwise
// a SessionPool of n identical sessions (n concurrent inferences instead of one at a time). If any
// session fails, every session it already created is closed before the error is returned, so a
// failed load never strands VRAM.
//
// Each pooled session gets a capped intra-op thread pool (poolIntraOpThreads); a lone session
// keeps ORT's default. threads >= 0 (the manifest's runtime.threads for this role) replaces both
// for every session created here; threads < 0 means the manifest sets nothing.
func newRunnable(path string, inputNames, outputNames []string, n, threads int, providers []engine.Provider) (engine.Runnable, error) {
	if n <= 1 {
		so := engine.SessionOptions{}
		if threads >= 0 {
			so.IntraOpThreads = threads
		}
		s, err := newEngineSession(path, inputNames, outputNames, providers, so)
		if err != nil {
			return nil, err
		}
		return s, nil
	}
	so := engine.SessionOptions{}
	if threads >= 0 {
		so.IntraOpThreads = threads
	} else {
		env := os.Getenv("VISIONSERVE_POOL_THREADS")
		var warn string
		so.IntraOpThreads, warn = poolIntraOpThreads(n, numCPU(), env)
		if warn != "" {
			warnOnce("VISIONSERVE_POOL_THREADS="+env, warn)
		}
	}
	sessions := make([]*engine.Session, 0, n)
	for i := 0; i < n; i++ {
		s, err := newEngineSession(path, inputNames, outputNames, providers, so)
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

// newEngineSession creates one ONNX session; tests replace it.
var newEngineSession = engine.NewSessionWith

// maxPoolThreads caps poolIntraOpThreads' default (see there for the measurements).
const maxPoolThreads = 3

// poolIntraOpThreads sizes the intra-op thread pool of EACH session in an n-session pool:
// NumCPU/(2n) threads, at least 1 and at most maxPoolThreads.
//
// ORT's default gives every session one spinning thread per physical core, and a pool exists to
// run its sessions at once: MobileSAM's automask drives its 4 decoder copies together, so the
// default put 4×cores busy threads on the machine. The lone sessions next to the pool (the SAM
// encoder, a detector) keep that default, so they already spin on one thread per physical core —
// half the logical CPUs with 2-way SMT. The pool gets the other half, split over its n sessions.
// Past 2–3 threads the small SAM decoder gains nothing and its extra threads only spin, hence
// the cap.
//
// Measured on CPU, served end to end (2×12-core Xeon, shared and busy; hosts emulated with
// taskset; lone sessions given one thread per emulated physical core, which is what ORT's default
// picks on such a host — under taskset ORT counts and pins to all 24 cores of the machine
// instead). Medians of 3 interleaved rounds, pool of 4 decoders, k = threads per decoder:
// box = mobile-sam, one box (s); auto = mobile-sam automask, 640×480 (s); grasp = grasp-rfdetr
// (s); conc4 = mobile-sam box, 4 clients (req/s). * = this rule, d = ORT's default.
//
//	host            k     box   auto  grasp  conc4
//	4 cores         1*   0.47   6.6   2.10   2.35
//	                2    0.83  11.2   3.17   1.30
//	                4d   1.19  22.1   5.21   0.88
//	4c/8t           1*   0.50   6.9   1.44   2.56
//	                2    0.48   6.0   1.47   2.40
//	                4d   0.61  12.4   2.64   1.80
//	8 cores         1*   0.39   6.4   2.00   3.30
//	                2    0.60   5.0   2.51   1.89
//	                8d   1.21  21.9   5.65   0.89
//	8c/16t          1    0.40   6.7   1.48   3.69
//	                2*   0.37   4.3   1.23   3.26
//	                4    0.38   4.4   1.67   3.25
//	                8d   0.52  10.6   2.74   2.00
//	12c/24t         1    0.36   6.6   1.23   4.08
//	                2+   0.33   3.9   1.15   4.02
//	                3*   0.31   3.4   1.21   3.98
//	                6+   0.32   3.9   1.72   3.46
//	                12d  0.52  10.3   2.82   1.98
//	24c/48t         2+   0.34   4.1   1.45   4.33
//	                3*   0.32   3.3   1.51   4.21
//	                6    0.32   3.5   1.80   3.70
//	                24d  0.87  15.9   4.16   1.28
//
// (+ = a second 3-round run, against 3 threads at 0.31/3.4/1.25/3.99 and 0.31/3.3/1.49/4.25.)
//
// Outputs are identical whatever the count. ORT's default is the slowest at every size, a single
// prompted request included: its spinning threads outnumber the CPUs. The previous rule,
// NumCPU/(4n), gave 1 thread up to 31 CPUs, which made automask 1.6–2× slower at 16 and 24.
//
// VISIONSERVE_POOL_THREADS (env) overrides it: an integer >= 1 is used as is (at most ncpu), 0
// restores ORT's default. A value that is not an integer >= 0 is ignored. Either correction comes
// back as warn, which the caller logs once (warnOnce) rather than on every load.
func poolIntraOpThreads(n, ncpu int, env string) (threads int, warn string) {
	heuristic := min(maxPoolThreads, max(1, ncpu/(2*max(n, 1))))
	v := strings.TrimSpace(env)
	if v == "" {
		return heuristic, ""
	}
	k, err := strconv.Atoi(v)
	switch {
	case err != nil || k < 0:
		return heuristic, fmt.Sprintf("lifecycle: ignoring VISIONSERVE_POOL_THREADS=%q (want an integer >= 0; "+
			"0 = ONNX Runtime's default) — pooled sessions get NumCPU/(2n) intra-op threads, 1 to %d", env, maxPoolThreads)
	case k > ncpu:
		return ncpu, fmt.Sprintf("lifecycle: VISIONSERVE_POOL_THREADS=%d is more than the %d logical CPUs — "+
			"capped at %d intra-op threads per pooled session", k, ncpu, ncpu)
	}
	return k, ""
}

// manifestThreads caps a manifest's runtime.threads value for one role at the machine's logical
// CPUs. The registry only checks it is >= 0: a manifest is portable, and the CPU count belongs to
// the host it is served on. A cap is logged once per model and role.
func manifestThreads(model, role string, n, ncpu int) int {
	if n <= ncpu {
		return n
	}
	warnOnce("threads:"+model+"/"+role, fmt.Sprintf("lifecycle: %s: runtime.threads.%s = %d is more than the %d "+
		"logical CPUs — capped at %d", model, role, n, ncpu, ncpu))
	return ncpu
}

// numCPU is runtime.NumCPU; tests replace it.
var numCPU = runtime.NumCPU

// warned holds the keys warnOnce has already logged.
var warned sync.Map

// warnOnce logs msg the first time key is seen in this process. A misconfiguration read at every
// load (an env var, a manifest field) is reported once, not once per model load.
func warnOnce(key, msg string) {
	if _, dup := warned.LoadOrStore(key, struct{}{}); !dup {
		log.Print(msg)
	}
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
		return nil, nil, fmt.Errorf("lifecycle: %w: %q is not in the registry", ErrModelNotFound, name)
	}
	man := entry.Manifest

	if !man.WeightsExist() {
		return nil, nil, fmt.Errorf("lifecycle: %w: no weights for %q at %s — download them per the README in the model directory",
			ErrModelNotFound, name, man.ModelFilePath())
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
	// The manifest's preprocessing, resolved once here for every model: the preprocess: block
	// with the legacy input.* fields as aliases (registry.Manifest.PreprocessSpec). The legacy
	// fields below stay filled — the registry keeps them consistent with the block.
	spec, err := man.PreprocessSpec()
	if err != nil {
		return nil, nil, fmt.Errorf("lifecycle: %q: %w", name, err)
	}

	cfg := models.Config{
		Name:       man.Name,
		Preprocess: &spec,
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
