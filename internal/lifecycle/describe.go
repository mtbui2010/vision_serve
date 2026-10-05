package lifecycle

import (
	"os"
	"sort"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/registry"
)

// LoadPlan is what Load would do for a model, decided WITHOUT creating any ONNX session: the
// checks Load runs first (buildModel + the input-shape check), the execution-provider chain, and
// the sessions it would open per role with their pool sizes and thread counts. It exists for
// `visionserve inspect`, so the card shows what serving really decides rather than a
// re-derivation that could drift.
type LoadPlan struct {
	// Manifest is the registry entry (nil when the name is not in the registry); Model the built
	// model object (nil when BuildErr is set before it was built).
	Manifest *registry.Manifest
	Model    models.Base
	// BuildErr is the error Load would fail with before opening any session: weights missing, a
	// pin that does not match, unreadable labels, a preprocessing the architecture refuses, or the
	// input-shape check (Input.Err). nil = only session creation (ONNX Runtime) remains.
	BuildErr error
	// Input is the input-shape check with what it compared (zero when the model was not built).
	Input InputFit
	// Providers is the EP chain every session tries in order (VISIONSERVE_EP, --tensorrt applied).
	Providers []engine.Provider
	// IdleUnloadSeconds is the idle time after which the model is unloaded (0 = never), with the
	// manager's --idle-unload-seconds override applied.
	IdleUnloadSeconds int
	// Sessions lists, per role (sorted; "model" for a plain model), what Load would open.
	Sessions []SessionPlan
}

// SessionPlan is one role's sessions.
type SessionPlan struct {
	Role, Path string
	// Pool is the number of identical sessions opened for the role (1 = a single session).
	Pool int
	// Threads is each session's intra-op thread count; 0 = ONNX Runtime's default (one per
	// physical core). ThreadsFrom says which rule chose it.
	Threads     int
	ThreadsFrom string
}

// Thread-count sources reported in SessionPlan.ThreadsFrom.
const (
	ThreadsORTDefault = "ONNX Runtime default"
	ThreadsManifest   = "manifest runtime.threads"
	ThreadsPoolRule   = "pool rule NumCPU/(2n), 1 to 3"
	ThreadsPoolEnv    = "VISIONSERVE_POOL_THREADS"
)

// Describe returns the LoadPlan of a registered model. It runs the same build and checks Load
// runs, and opens nothing; the model need not be (and is not) loaded.
func (m *Manager) Describe(name string) LoadPlan {
	var plan LoadPlan
	if e, ok := m.reg.Get(name); ok {
		plan.Manifest = e.Manifest
	}
	base, man, err := m.buildModel(name)
	if err != nil {
		plan.BuildErr = err
		return plan
	}
	plan.Manifest, plan.Model = man, base
	plan.Input = JudgeInputShape(man, base)
	if plan.Input.Err != nil {
		plan.BuildErr = plan.Input.Err
	}
	plan.Providers, _ = man.Providers() // validated by buildModel
	plan.IdleUnloadSeconds = man.Runtime.IdleUnloadSeconds
	if m.idleOverrideSec >= 0 {
		plan.IdleUnloadSeconds = m.idleOverrideSec
	}

	switch mdl := base.(type) {
	case models.PipelineModel:
		pools := map[string]int{}
		if ps, ok := mdl.(models.PoolSizer); ok {
			for role, n := range ps.PoolSizes() {
				pools[role] = n
			}
		}
		files := man.FilesAbs()
		for _, role := range mdl.Roles() {
			n := pools[role]
			if ov := poolOverride(); ov > 0 {
				n = ov
			}
			mt, set := man.IntraOpThreads(role)
			threads, from := sessionThreads(n, mt, set)
			plan.Sessions = append(plan.Sessions, SessionPlan{Role: role, Path: files[role], Pool: max(n, 1), Threads: threads, ThreadsFrom: from})
		}
	case models.Model:
		n := poolOverride()
		threads, from := sessionThreads(n, 0, false)
		plan.Sessions = []SessionPlan{{Role: "model", Path: man.ModelFilePath(), Pool: max(n, 1), Threads: threads, ThreadsFrom: from}}
	}
	sort.Slice(plan.Sessions, func(i, j int) bool { return plan.Sessions[i].Role < plan.Sessions[j].Role })
	return plan
}

// sessionThreads is the intra-op thread count newRunnable gives each of a role's n sessions, and
// the rule that chose it: the manifest's runtime.threads (capped at the CPUs, manifestThreads),
// else ORT's default for a lone session, else the pool rule (poolIntraOpThreads) or its env
// override. It never logs; Load warns about the same values when it applies them.
func sessionThreads(n, manifestN int, manifestSet bool) (threads int, from string) {
	if manifestSet {
		return min(manifestN, numCPU()), ThreadsManifest
	}
	if n <= 1 {
		return 0, ThreadsORTDefault
	}
	env := os.Getenv("VISIONSERVE_POOL_THREADS")
	threads, warn := poolIntraOpThreads(n, numCPU(), env)
	if env != "" && warn == "" {
		return threads, ThreadsPoolEnv
	}
	return threads, ThreadsPoolRule
}
