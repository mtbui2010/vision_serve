package lifecycle

import (
	"image"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/registry"
)

// testPipe is a session-less PipelineModel for runtime tests: its roles are the manifest's
// files: keys (so Load opens one fake engine per role) and Infer runs the hook installed for the
// model name, if any.
type testPipe struct {
	name  string
	roles []string
}

func (p *testPipe) Name() string      { return p.name }
func (p *testPipe) Task() models.Task { return models.TaskEmbed }
func (p *testPipe) Roles() []string   { return p.roles }
func (p *testPipe) Infer(img image.Image, pr models.Prompt, r models.Runner) (models.Result, error) {
	if h := testHooks.infer(p.name); h != nil {
		return h(r)
	}
	return models.Result{}, nil
}

// exclusivePipe is a testPipe that asks the runtime to serialize its Infer (models.Exclusive).
type exclusivePipe struct{ testPipe }

func (p *exclusivePipe) Exclusive() bool { return true }

// notExclusivePipe implements models.Exclusive but says no: it must run concurrently.
type notExclusivePipe struct{ testPipe }

func (p *notExclusivePipe) Exclusive() bool { return false }

// buildGate blocks a test-pipe factory: started is closed when the build begins, and the build
// returns once proceed is closed.
type buildGate struct{ started, proceed chan struct{} }

// hooks holds per-model-name behaviour for the test architectures (guarded by mu).
type hooks struct {
	mu     sync.Mutex
	inferF map[string]func(models.Runner) (models.Result, error)
	gates  map[string]*buildGate
}

var testHooks = &hooks{
	inferF: map[string]func(models.Runner) (models.Result, error){},
	gates:  map[string]*buildGate{},
}

func (h *hooks) infer(name string) func(models.Runner) (models.Result, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.inferF[name]
}

func (h *hooks) setInfer(t *testing.T, name string, f func(models.Runner) (models.Result, error)) {
	h.mu.Lock()
	h.inferF[name] = f
	h.mu.Unlock()
	t.Cleanup(func() { h.mu.Lock(); delete(h.inferF, name); h.mu.Unlock() })
}

// gate installs a build gate for name and returns it.
func (h *hooks) gate(t *testing.T, name string) *buildGate {
	g := &buildGate{started: make(chan struct{}), proceed: make(chan struct{})}
	h.mu.Lock()
	h.gates[name] = g
	h.mu.Unlock()
	t.Cleanup(func() { h.mu.Lock(); delete(h.gates, name); h.mu.Unlock() })
	return g
}

// waitGate runs inside a factory: it blocks on name's gate, once (the gate is consumed).
func (h *hooks) waitGate(name string) {
	h.mu.Lock()
	g := h.gates[name]
	delete(h.gates, name)
	h.mu.Unlock()
	if g != nil {
		close(g.started)
		<-g.proceed
	}
}

func init() {
	pipe := func(cfg models.Config) testPipe {
		testHooks.waitGate(cfg.Name)
		roles := make([]string, 0, len(cfg.Files))
		for r := range cfg.Files {
			roles = append(roles, r)
		}
		sort.Strings(roles)
		return testPipe{name: cfg.Name, roles: roles}
	}
	models.Register("test-pipe", func(cfg models.Config) (models.Base, error) {
		p := pipe(cfg)
		return &p, nil
	})
	models.Register("test-pipe-exclusive", func(cfg models.Config) (models.Base, error) {
		return &exclusivePipe{pipe(cfg)}, nil
	})
	models.Register("test-pipe-not-exclusive", func(cfg models.Config) (models.Base, error) {
		return &notExclusivePipe{pipe(cfg)}, nil
	})
}

// writeTestModel writes <root>/<name>/manifest.yaml for architecture arch with one or more roles
// (files: map, default just "x"), creating an empty weights file per role. extra is appended to
// the manifest verbatim.
func writeTestModel(t *testing.T, root, name, arch, extra string, roles ...string) {
	t.Helper()
	if len(roles) == 0 {
		roles = []string{"x"}
	}
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	y := "name: " + name + "\ntask: embed\nlicense: MIT\narchitecture: " + arch + "\nfiles:\n"
	for _, r := range roles {
		if err := os.WriteFile(filepath.Join(dir, r+".onnx"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		y += "  " + r + ": " + r + ".onnx\n"
	}
	y += "input:\n  width: 8\n  height: 8\nruntime:\n  prefer: [cpu]\n" + extra
	if err := os.WriteFile(filepath.Join(dir, "manifest.yaml"), []byte(y), 0o644); err != nil {
		t.Fatal(err)
	}
}

func scanRegistry(t *testing.T, root string) *registry.Registry {
	t.Helper()
	reg := registry.New(root)
	warns, err := reg.Scan()
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range warns {
		t.Fatalf("registry warning: %v", w)
	}
	return reg
}

// fakeOpener stands in for ONNX Runtime: every "session" it opens is a fakeEngine it remembers.
type fakeOpener struct {
	mu   sync.Mutex
	made []*fakeEngine
}

func (o *fakeOpener) open(_ string, _, _ []string, _ int, _ []engine.Provider) (engine.Runnable, error) {
	fe := &fakeEngine{}
	o.mu.Lock()
	o.made = append(o.made, fe)
	o.mu.Unlock()
	return fe, nil
}

func (o *fakeOpener) engines() []*fakeEngine {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]*fakeEngine(nil), o.made...)
}

// newFakeManager is a real Manager over reg whose sessions are fakeEngines.
func newFakeManager(t *testing.T, reg *registry.Registry) (*Manager, *fakeOpener) {
	t.Helper()
	op := &fakeOpener{}
	m := NewManager(reg)
	m.openRunnable = op.open
	t.Cleanup(m.Close)
	return m, op
}
