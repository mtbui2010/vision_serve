package lifecycle

import (
	"context"
	"errors"
	"fmt"
	"image"
	"sync/atomic"
	"testing"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// slotEngine is a one-slot session that waits for its slot the way engine.Session and
// engine.SessionPool do: until the slot frees or ctx ends. hold, when set, keeps the first run
// inside the slot until it is closed; runs counts the runs that got the slot.
type slotEngine struct {
	fakeEngine
	slot    chan struct{}
	hold    chan struct{}
	entered chan struct{} // closed when the first run is inside the slot
	runs    atomic.Int32
}

func newSlotEngine() *slotEngine {
	return &slotEngine{slot: make(chan struct{}, 1), hold: make(chan struct{}), entered: make(chan struct{})}
}

func (e *slotEngine) RunNamed(ctx context.Context, _ map[string]engine.Tensor) ([]engine.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case e.slot <- struct{}{}:
	case <-ctx.Done():
		return nil, fmt.Errorf("slot: %w", ctx.Err())
	}
	defer func() { <-e.slot }()
	if e.runs.Add(1) == 1 {
		close(e.entered)
		<-e.hold
	}
	return nil, nil
}

// request runs one request as the server does — admission, then PredictPrompt — and reports its
// error on the returned channel.
func request(m *Manager, ctx context.Context, name string) <-chan error {
	errc := make(chan error, 1)
	go func() {
		release, err := m.Admit(context.Background(), name) // admitted before the client leaves
		if err != nil {
			errc <- err
			return
		}
		defer release()
		_, err = m.PredictPrompt(ctx, name, image.NewRGBA(image.Rect(0, 0, 2, 2)), models.Prompt{})
		errc <- err
	}()
	return errc
}

// promptly returns the error from errc, failing the test if it takes more than a second.
func promptly(t *testing.T, what string, errc <-chan error) error {
	t.Helper()
	select {
	case err := <-errc:
		return err
	case <-time.After(time.Second):
		t.Fatalf("%s: still waiting a second after its context was cancelled", what)
		return nil
	}
}

// assertIdle checks that every request let go of what it held: no lease on name's session and no
// admission slot.
func assertIdle(t *testing.T, m *Manager, name string) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if s := m.live[name]; s == nil || s.refs != 0 {
		t.Fatalf("session %q: live=%v, want live with no lease left", name, s != nil)
	}
	if n := m.admitted[name]; n != 0 {
		t.Fatalf("%d admission slots still held on %q", n, name)
	}
}

// A request queued behind the only session slot while its client leaves stops waiting at once,
// runs nothing, and leaves the slot, its lease and its admission slot as they were.
func TestCancelledWaiterOnSessionSlot(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "one-slot", "test-pipe", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	eng := newSlotEngine()
	m.openRunnable = func(string, []string, []string, int, int, []engine.Provider) (engine.Runnable, error) {
		return eng, nil
	}
	var infers atomic.Int32
	testHooks.setInfer(t, "one-slot", func(r models.Runner) (models.Result, error) {
		infers.Add(1)
		_, err := r.Run("x", nil)
		return models.Result{}, err
	})

	first := request(m, context.Background(), "one-slot")
	<-eng.entered // the first request holds the slot

	ctx, cancel := context.WithCancel(context.Background())
	queued := request(m, ctx, "one-slot")
	waitFor(t, "the second request to reach the session", func() bool { return infers.Load() == 2 })
	time.Sleep(10 * time.Millisecond) // and to block on the slot
	cancel()
	if err := promptly(t, "queued request", queued); !errors.Is(err, context.Canceled) {
		t.Fatalf("queued request: err = %v, want context.Canceled", err)
	}

	close(eng.hold)
	if err := <-first; err != nil {
		t.Fatalf("the request holding the slot: %v", err)
	}
	if n := eng.runs.Load(); n != 1 {
		t.Fatalf("%d runs got the slot, want 1: the cancelled request ran", n)
	}
	if err := <-request(m, context.Background(), "one-slot"); err != nil {
		t.Fatalf("a request after the cancelled one: %v", err)
	}
	if n := eng.runs.Load(); n != 2 {
		t.Fatalf("%d runs, want 2", n)
	}
	assertIdle(t, m, "one-slot")
}

// The same for the per-model lock of an Exclusive pipeline: the waiter gives up, its Infer never
// starts, and the lock is free for the next request.
func TestCancelledWaiterOnExclusiveLock(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "excl", "test-pipe-exclusive", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	hold, entered := make(chan struct{}), make(chan struct{})
	var infers atomic.Int32
	testHooks.setInfer(t, "excl", func(models.Runner) (models.Result, error) {
		if infers.Add(1) == 1 {
			close(entered)
			<-hold
		}
		return models.Result{}, nil
	})

	first := request(m, context.Background(), "excl")
	<-entered

	ctx, cancel := context.WithCancel(context.Background())
	queued := request(m, ctx, "excl")
	time.Sleep(20 * time.Millisecond) // let it block on the lock
	cancel()
	if err := promptly(t, "request waiting for the Exclusive lock", queued); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}

	close(hold)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
	if n := infers.Load(); n != 1 {
		t.Fatalf("Infer ran %d times, want 1: the cancelled request ran", n)
	}
	if err := <-request(m, context.Background(), "excl"); err != nil {
		t.Fatalf("a request after the cancelled one (is the lock still held?): %v", err)
	}
	if n := infers.Load(); n != 2 {
		t.Fatalf("Infer ran %d times, want 2", n)
	}
	assertIdle(t, m, "excl")
}

// A request whose client is already gone runs nothing: no load, no inference.
func TestDoneContextLoadsAndRunsNothing(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "lazy", "test-pipe", "")
	m, op := newFakeManager(t, scanRegistry(t, root))
	var infers atomic.Int32
	testHooks.setInfer(t, "lazy", func(models.Runner) (models.Result, error) {
		infers.Add(1)
		return models.Result{}, nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	if _, err := m.PredictPrompt(ctx, "lazy", img, models.Prompt{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("PredictPrompt: err = %v, want context.Canceled", err)
	}
	if _, err := m.InferTensor(ctx, "lazy", engine.F32([]float32{1}, 1)); !errors.Is(err, context.Canceled) {
		t.Fatalf("InferTensor: err = %v, want context.Canceled", err)
	}
	if _, err := m.Explain(ctx, "lazy", img, ExplainRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Explain: err = %v, want context.Canceled", err)
	}
	if _, err := m.Preprocess(ctx, "lazy", img, models.Prompt{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Preprocess: err = %v, want context.Canceled", err)
	}
	if m.IsLoaded("lazy") || len(op.engines()) != 0 || infers.Load() != 0 {
		t.Fatalf("loaded=%v sessions=%d infers=%d for a request that was already gone",
			m.IsLoaded("lazy"), len(op.engines()), infers.Load())
	}
}

// Requests waiting on a model's load give up one by one when their clients leave — the one that
// started the load too — and the load is not cancelled for the request still waiting: it finishes
// once, the model goes live, and nobody builds it again.
func TestCancelledLoadWaitersDoNotCancelTheLoad(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "slow", "test-pipe", "")
	m, op := newFakeManager(t, scanRegistry(t, root))
	gate := testHooks.gate(t, "slow")

	leaderCtx, cancelLeader := context.WithCancel(context.Background())
	leader := make(chan error, 1)
	go func() { leader <- m.Load(leaderCtx, "slow") }()
	<-gate.started // the load is building the model

	waiterCtx, cancelWaiter := context.WithCancel(context.Background())
	waiter := make(chan error, 1)
	go func() { waiter <- m.Load(waiterCtx, "slow") }()
	stays := make(chan error, 1)
	go func() { stays <- m.Load(context.Background(), "slow") }()
	waitFor(t, "two requests to wait on the load", func() bool { return m.waitersFor("slow") == 2 })

	cancelWaiter()
	if err := promptly(t, "cancelled load waiter", waiter); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled waiter: err = %v, want context.Canceled", err)
	}
	if n := m.waitersFor("slow"); n != 1 {
		t.Fatalf("%d waiters after one left, want 1", n)
	}
	cancelLeader()
	if err := promptly(t, "cancelled request that started the load", leader); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled leader: err = %v, want context.Canceled", err)
	}
	if m.IsLoaded("slow") {
		t.Fatal("loaded before the build finished")
	}

	close(gate.proceed)
	if err := <-stays; err != nil {
		t.Fatalf("the request that stayed: %v (the load was cancelled for it)", err)
	}
	if !m.IsLoaded("slow") {
		t.Fatal("model not live after its load finished")
	}
	if err := m.Load(context.Background(), "slow"); err != nil {
		t.Fatal(err)
	}
	if engs := op.engines(); len(engs) != 1 || engs[0].closed.Load() {
		t.Fatalf("built %d sessions (closed: %v), want exactly 1, live", len(engs), len(engs) == 1 && engs[0].closed.Load())
	}
}

// When every request waiting on a load leaves, the load still completes and the model is live for
// the next request, which does not build it again.
func TestLoadFinishesAfterEveryWaiterLeft(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "orphan", "test-pipe", "")
	m, op := newFakeManager(t, scanRegistry(t, root))
	gate := testHooks.gate(t, "orphan")

	ctx, cancel := context.WithCancel(context.Background())
	leader := make(chan error, 1)
	go func() { leader <- m.Load(ctx, "orphan") }()
	<-gate.started
	cancel()
	if err := promptly(t, "the only request on the load", leader); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	close(gate.proceed)
	waitFor(t, "the orphaned load to go live", func() bool { return m.IsLoaded("orphan") })
	if err := m.Load(context.Background(), "orphan"); err != nil {
		t.Fatal(err)
	}
	if n := len(op.engines()); n != 1 {
		t.Fatalf("built %d sessions, want 1", n)
	}
}
