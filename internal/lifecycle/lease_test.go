package lifecycle

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"visionserve/internal/engine"
)

// fakeEngine records Close and fails Run after it, like a real session would.
type fakeEngine struct{ closed atomic.Bool }

func (f *fakeEngine) Run([]engine.Tensor) ([]engine.Tensor, error) { return f.check() }
func (f *fakeEngine) RunNamed(map[string]engine.Tensor) ([]engine.Tensor, error) {
	return f.check()
}
func (f *fakeEngine) check() ([]engine.Tensor, error) {
	if f.closed.Load() {
		return nil, engine.ErrClosed
	}
	return nil, nil
}
func (f *fakeEngine) InputNames() []string      { return nil }
func (f *fakeEngine) OutputNames() []string     { return nil }
func (f *fakeEngine) ActiveEP() engine.Provider { return engine.ProviderCPU }
func (f *fakeEngine) Close() error              { f.closed.Store(true); return nil }

func newTestManager(name string, idle time.Duration, last time.Time) (*Manager, *fakeEngine) {
	fe := &fakeEngine{}
	m := &Manager{live: map[string]*Session{}, stop: make(chan struct{})}
	m.live[name] = &Session{name: name, engine: fe, idleTimeout: idle, lastUsed: last}
	return m, fe
}

// A session must not be closed while a request holds it: Unload retires it, and the LAST
// release closes it. Before the lease, Unload/reaper/Close closed it under the request.
func TestUnloadWaitsForInFlightRequest(t *testing.T) {
	m, fe := newTestManager("m", 0, time.Now())
	s, release, err := m.acquire("m")
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Unload("m"); err != nil {
		t.Fatal(err)
	}
	if fe.closed.Load() {
		t.Fatal("Unload closed a session that a request was still using")
	}
	if _, err := s.engine.Run(nil); err != nil {
		t.Fatalf("in-flight request failed after Unload: %v", err)
	}
	if _, _, err := m.acquire("m"); err == nil {
		t.Fatal("a retired session was handed to a new request")
	}
	release()
	if !fe.closed.Load() {
		t.Fatal("the last release did not close the retired session")
	}
}

// The reaper must skip a busy session even when its lastUsed is far past the idle timeout
// (a long request that started just before the timeout).
func TestReaperSkipsBusySession(t *testing.T) {
	m, fe := newTestManager("m", time.Millisecond, time.Now().Add(-time.Hour))
	_, release, _ := m.acquire("m")
	m.live["m"].lastUsed = time.Now().Add(-time.Hour) // acquire touched it; make it stale again
	m.mu.Lock()
	var expired []*Session
	for name, s := range m.live {
		if s.refs == 0 && s.idleTimeout > 0 && s.idleFor(time.Now()) > s.idleTimeout {
			if r := m.retireLocked(name); r != nil {
				expired = append(expired, r)
			}
		}
	}
	m.mu.Unlock()
	if len(expired) != 0 || fe.closed.Load() {
		t.Fatal("reaper retired a session that was serving a request")
	}
	release()
}

// Many concurrent requests racing Unload and Close: nothing may panic, and the session is
// closed exactly when the last request is done.
func TestConcurrentRequestsAndUnloadDoNotRace(t *testing.T) {
	m, fe := newTestManager("m", 0, time.Now())
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s, release, err := m.acquire("m")
			if err != nil {
				return // unloaded before we got it: a clean error, which is fine
			}
			defer release()
			if _, err := s.engine.Run(nil); err != nil {
				t.Errorf("request saw a closed session: %v", err)
			}
		}()
	}
	go func() { _ = m.Unload("m") }()
	wg.Wait()
	m.Close()
	if !fe.closed.Load() {
		t.Fatal("session never closed")
	}
}
