package lifecycle

import (
	"testing"
	"time"

	"visionserve/internal/engine"
)

// waitFor polls cond until it holds or the deadline passes.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// waiters reports how many requests are waiting on name's in-progress load.
func (m *Manager) waitersFor(name string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if c := m.loading[name]; c != nil {
		return c.waiters
	}
	return 0
}

// L7: an Unload that arrives while the model is still loading used to find nothing in the live
// map, report success, and then watch the load finish and make the model resident — "unloaded"
// but holding its memory. The freshly built session must instead be closed when the load ends,
// and the requests that were waiting on that load must not quietly load it again.
func TestUnloadDuringLoadRetiresTheNewSession(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "race", "test-pipe", "")
	m, op := newFakeManager(t, scanRegistry(t, root))
	gate := testHooks.gate(t, "race")

	leader := make(chan error, 1)
	go func() { leader <- m.Load("race") }()
	<-gate.started // the leader is building the model

	waiter := make(chan error, 1)
	go func() { waiter <- m.Load("race") }()
	waitFor(t, "the second request to wait on the load", func() bool { return m.waitersFor("race") == 1 })

	if err := m.Unload("race"); err != nil {
		t.Fatalf("Unload during load: %v", err)
	}
	close(gate.proceed)

	if err := <-leader; err == nil {
		t.Error("the load that was unloaded under it reported success")
	}
	if err := <-waiter; err == nil {
		t.Error("a request waiting on the unloaded load reported success")
	}
	if m.IsLoaded("race") {
		t.Fatal("model resident after Unload returned: the in-progress load went live anyway")
	}
	engs := op.engines()
	if len(engs) != 1 {
		t.Fatalf("built %d engines, want exactly 1 (the waiter must not reload)", len(engs))
	}
	if !engs[0].closed.Load() {
		t.Fatal("the session built by the unloaded load was never closed (VRAM leak)")
	}

	// A request that arrives AFTER the unload is a new request: it loads normally.
	if err := m.Load("race"); err != nil {
		t.Fatalf("load after unload: %v", err)
	}
	if !m.IsLoaded("race") {
		t.Fatal("a fresh load after the unload did not go live")
	}
}

// Shutdown during a load (a --preload still building when SIGTERM arrives): the session must be
// closed when the load finishes, and nothing may load after Close.
func TestCloseDuringLoadDropsTheNewSession(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "late", "test-pipe", "")
	m, op := newFakeManager(t, scanRegistry(t, root))
	gate := testHooks.gate(t, "late")

	leader := make(chan error, 1)
	go func() { leader <- m.Load("late") }()
	<-gate.started
	m.Close()
	close(gate.proceed)

	if err := <-leader; err == nil {
		t.Error("a load finishing after Close reported success")
	}
	if m.IsLoaded("late") {
		t.Fatal("a load finishing after Close went live")
	}
	if engs := op.engines(); len(engs) != 1 || !engs[0].closed.Load() {
		t.Fatalf("the session built during shutdown was not closed: %d engines", len(engs))
	}
	if err := m.Load("late"); err == nil {
		t.Fatal("Load after Close succeeded")
	}
}

// A model factory that panics must not wedge its name: the in-progress entry used to stay in the
// loading map forever, so every later request for that model hung on it.
func TestPanickingBuildDoesNotWedgeTheModel(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "boom", "test-pipe", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	calls := 0
	m.openRunnable = func(string, []string, []string, int, []engine.Provider) (engine.Runnable, error) {
		calls++
		if calls == 1 {
			panic("simulated crash while creating a session")
		}
		return &fakeEngine{}, nil
	}

	if err := m.Load("boom"); err == nil {
		t.Fatal("a load that panicked reported success")
	}
	done := make(chan error, 1)
	go func() { done <- m.Load("boom") }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("retry after a panicked load: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the model is wedged: a load after a panicked load hangs")
	}
}
