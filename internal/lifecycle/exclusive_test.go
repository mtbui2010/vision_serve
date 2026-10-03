package lifecycle

import (
	"context"
	"errors"
	"image"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// rendezvous makes n Infer calls wait for each other: each returns an error unless all n were
// inside at the same time (within the timeout). It proves concurrency without relying on timing.
type rendezvous struct {
	mu      sync.Mutex
	n, seen int
	all     chan struct{}
}

func newRendezvous(n int) *rendezvous { return &rendezvous{n: n, all: make(chan struct{})} }

func (r *rendezvous) arrive() error {
	r.mu.Lock()
	r.seen++
	if r.seen == r.n {
		close(r.all)
	}
	r.mu.Unlock()
	select {
	case <-r.all:
		return nil
	case <-time.After(3 * time.Second):
		return errors.New("calls did not overlap: Infer was serialized")
	}
}

// An Exclusive pipeline's Infer never runs concurrently with itself — the runtime holds the lock,
// replacing the package-level groundingdino.PipelineMu.
func TestExclusivePipelineInferIsSerialized(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "excl", "test-pipe-exclusive", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))

	var inside, peak atomic.Int32
	testHooks.setInfer(t, "excl", func(models.Runner) (models.Result, error) {
		n := inside.Add(1)
		defer inside.Add(-1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		time.Sleep(2 * time.Millisecond)
		return models.Result{}, nil
	})

	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := m.Predict(context.Background(), "excl", img); err != nil {
				t.Errorf("Predict: %v", err)
			}
		}()
	}
	wg.Wait()
	if p := peak.Load(); p != 1 {
		t.Fatalf("%d concurrent Infer calls on an Exclusive model, want 1", p)
	}
}

// Models that do not ask for it (no Exclusive method, or Exclusive() == false) keep running
// concurrently: the lock is opt-in, not a new global bottleneck.
func TestNonExclusivePipelinesRunConcurrently(t *testing.T) {
	for _, arch := range []string{"test-pipe", "test-pipe-not-exclusive"} {
		t.Run(arch, func(t *testing.T) {
			root := t.TempDir()
			writeTestModel(t, root, "free", arch, "")
			m, _ := newFakeManager(t, scanRegistry(t, root))
			rv := newRendezvous(2)
			testHooks.setInfer(t, "free", func(models.Runner) (models.Result, error) {
				return models.Result{}, rv.arrive()
			})
			runConcurrently(t, m, "free", "free")
		})
	}
}

// The lock is per loaded SESSION, not global: two Exclusive models run side by side (the global
// PipelineMu made the default router wait on an unrelated GroundingDINO pipeline).
func TestExclusiveLockIsPerSession(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "ex-a", "test-pipe-exclusive", "")
	writeTestModel(t, root, "ex-b", "test-pipe-exclusive", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	rv := newRendezvous(2)
	hook := func(models.Runner) (models.Result, error) { return models.Result{}, rv.arrive() }
	testHooks.setInfer(t, "ex-a", hook)
	testHooks.setInfer(t, "ex-b", hook)
	runConcurrently(t, m, "ex-a", "ex-b")
}

func runConcurrently(t *testing.T, m *Manager, names ...string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	errs := make(chan error, len(names))
	for _, n := range names {
		go func(n string) { _, err := m.Predict(context.Background(), n, img); errs <- err }(n)
	}
	for range names {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
}

// An Exclusive model runs one request at a time whatever its pools, so admission counts it as
// one slot (bound = the floor), not as its largest pool.
func TestExclusiveCountsAsOneSlot(t *testing.T) {
	t.Setenv("VS_POOL_OVERRIDE", "6")
	t.Setenv("VISIONSERVE_MAX_QUEUE", "")
	root := t.TempDir()
	writeTestModel(t, root, "excl", "test-pipe-exclusive", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	m.openRunnable = func(_ string, _, _ []string, n, _ int, _ []engine.Provider) (engine.Runnable, error) {
		return &fakePool{n: n}, nil
	}
	if err := m.Load(context.Background(), "excl"); err != nil {
		t.Fatal(err)
	}
	if got := admitCapacity(t, m, "excl"); got != defaultMinQueue {
		t.Fatalf("Exclusive model with pools of 6: bound %d, want %d (one slot)", got, defaultMinQueue)
	}
}
