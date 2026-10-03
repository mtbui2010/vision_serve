package lifecycle

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"visionserve/internal/engine"
)

// fakePool is a fakeEngine that reports n concurrent slots, like engine.SessionPool.
type fakePool struct {
	fakeEngine
	n int
}

func (p *fakePool) Size() int { return p.n }

// A full model must be refused IMMEDIATELY with ErrOverloaded — the whole point is that a request
// does not wait (holding its decoded upload) behind an unbounded queue — and only that model.
func TestAdmitBoundsRequestsPerModel(t *testing.T) {
	m := &Manager{live: map[string]*Session{}, stop: make(chan struct{})}
	m.SetMaxQueue(3)

	var releases []func()
	for i := 0; i < 3; i++ {
		r, err := m.Admit(context.Background(), "a")
		if err != nil {
			t.Fatalf("request %d refused below the bound: %v", i, err)
		}
		releases = append(releases, r)
	}
	start := time.Now()
	r, err := m.Admit(context.Background(), "a")
	if !errors.Is(err, ErrOverloaded) {
		t.Fatalf("4th request on a model bounded at 3: err = %v, want ErrOverloaded", err)
	}
	if time.Since(start) > 100*time.Millisecond {
		t.Fatal("a refused request waited instead of failing fast")
	}
	r() // the release handed back with an error must be safe to call

	other, err := m.Admit(context.Background(), "b")
	if err != nil {
		t.Fatalf("the bound is per model, but another model was refused: %v", err)
	}
	other()

	releases[0]()
	again, err := m.Admit(context.Background(), "a")
	if err != nil {
		t.Fatalf("a slot freed by release was not reusable: %v", err)
	}
	again()
	for _, r := range releases[1:] {
		r()
	}
}

// Calling release twice (a defer plus an explicit call on an error path) must free ONE slot, not
// two: the second call would otherwise hand a slot that belongs to another request to a third.
func TestAdmitReleaseIsIdempotent(t *testing.T) {
	m := &Manager{live: map[string]*Session{}, stop: make(chan struct{})}
	m.SetMaxQueue(1)

	r1, err := m.Admit(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	r1()
	r1()
	r2, err := m.Admit(context.Background(), "a")
	if err != nil {
		t.Fatalf("slot not freed: %v", err)
	}
	if _, err := m.Admit(context.Background(), "a"); !errors.Is(err, ErrOverloaded) {
		t.Fatalf("a double release freed a second slot: err = %v, want ErrOverloaded", err)
	}
	r2()
	m.mu.Lock()
	n := len(m.admitted)
	m.mu.Unlock()
	if n != 0 {
		t.Fatalf("%d admission entries left after every release: names would accumulate", n)
	}
}

// The automatic bound follows the model's concurrency: twice its inference slots (the largest
// pool among its roles), never below defaultMinQueue, and the floor alone before it is loaded.
func TestAdmitDefaultFollowsPoolSize(t *testing.T) {
	t.Setenv("VS_POOL_OVERRIDE", "20") // 2x20 = 40 > the floor, so the pool decides
	t.Setenv("VISIONSERVE_MAX_QUEUE", "")
	root := t.TempDir()
	writeTestModel(t, root, "pooled", "test-pipe", "")
	m, _ := newFakeManager(t, scanRegistry(t, root))
	m.openRunnable = func(_ string, _, _ []string, n int, _ []engine.Provider) (engine.Runnable, error) {
		if n > 1 {
			return &fakePool{n: n}, nil
		}
		return &fakeEngine{}, nil
	}

	if got := admitCapacity(t, m, "pooled"); got != defaultMinQueue {
		t.Fatalf("not loaded: bound %d, want the floor %d", got, defaultMinQueue)
	}
	if err := m.Load("pooled"); err != nil {
		t.Fatal(err)
	}
	if got := admitCapacity(t, m, "pooled"); got != 40 {
		t.Fatalf("pool of 20: bound %d, want 2x20 = 40", got)
	}
}

// admitCapacity admits until refused and returns how many were admitted (then releases them).
func admitCapacity(t *testing.T, m *Manager, name string) int {
	t.Helper()
	var rs []func()
	defer func() {
		for _, r := range rs {
			r()
		}
	}()
	for i := 0; i < 10_000; i++ {
		r, err := m.Admit(context.Background(), name)
		if err != nil {
			if !errors.Is(err, ErrOverloaded) {
				t.Fatalf("Admit: %v", err)
			}
			return len(rs)
		}
		rs = append(rs, r)
	}
	return -1 // unbounded
}

func TestAdmitMaxQueueEnv(t *testing.T) {
	for _, c := range []struct {
		env  string
		want int
	}{
		{"2", 2},
		{" 5 ", 5},
		{"0", -1}, // 0 = admission control off
		{"lots", defaultMinQueue},
		{"-3", defaultMinQueue},
		{"", defaultMinQueue},
	} {
		t.Setenv("VISIONSERVE_MAX_QUEUE", c.env)
		m := NewManager(scanRegistry(t, t.TempDir()))
		if got := admitCapacity(t, m, "x"); got != c.want {
			t.Errorf("VISIONSERVE_MAX_QUEUE=%q: bound %d, want %d", c.env, got, c.want)
		}
		m.Close()
	}
}

// Under contention the bound must hold exactly: never more admitted at once than the limit.
func TestAdmitBoundHoldsUnderContention(t *testing.T) {
	const limit = 5
	m := &Manager{live: map[string]*Session{}, stop: make(chan struct{})}
	m.SetMaxQueue(limit)
	var inside, peak, ok, refused atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				r, err := m.Admit(context.Background(), "m")
				if err != nil {
					refused.Add(1)
					continue
				}
				ok.Add(1)
				n := inside.Add(1)
				for {
					p := peak.Load()
					if n <= p || peak.CompareAndSwap(p, n) {
						break
					}
				}
				time.Sleep(10 * time.Microsecond)
				inside.Add(-1)
				r()
				r() // idempotent under contention too
			}
		}()
	}
	wg.Wait()
	if p := peak.Load(); p > limit {
		t.Fatalf("%d requests admitted at once, bound is %d", p, limit)
	}
	if ok.Load() == 0 || refused.Load() == 0 {
		t.Fatalf("expected both admissions and refusals under contention: ok=%d refused=%d", ok.Load(), refused.Load())
	}
}

// A request whose client has left is not admitted: the error carries the context's, it is not an
// overload, and no slot is taken — the bound is still fully available to live requests.
func TestAdmitRefusesADoneContext(t *testing.T) {
	m := &Manager{live: map[string]*Session{}, stop: make(chan struct{})}
	m.SetMaxQueue(1)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, err := m.Admit(ctx, "a")
	if !errors.Is(err, context.Canceled) || errors.Is(err, ErrOverloaded) {
		t.Fatalf("canceled request: err = %v, want context.Canceled (not ErrOverloaded)", err)
	}
	r() // the release handed back with an error must be safe to call
	if n := m.admitted["a"]; n != 0 {
		t.Fatalf("a canceled request holds %d slot(s)", n)
	}
	live, err := m.Admit(context.Background(), "a")
	if err != nil {
		t.Fatalf("the only slot was taken by a request that was refused: %v", err)
	}
	live()
}
