package lifecycle

import (
	"fmt"
	"time"
)

// Unload releases a model from memory. Not an error if the model is not loaded.
//
// A model still LOADING is unloaded too: Unload does not wait for it (a TensorRT build takes
// minutes), it marks the load cancelled, and the load closes what it built instead of making it
// live. Without that, the load finished after Unload had reported success and the model stayed
// resident.
func (m *Manager) Unload(name string) error {
	m.mu.Lock()
	if call := m.loading[name]; call != nil {
		call.cancelled = true
	}
	s := m.retireLocked(name)
	m.mu.Unlock()
	if s == nil {
		return nil // not loaded, or still in use: the last request closes it
	}
	return s.close()
}

// acquire leases the live session for name: it cannot be closed (by Unload, the idle reaper or
// Close) until the returned release runs. Without the lease a request could keep using a
// session that was closed under it — a "send on closed channel" panic, or a hang on a pool.
// It also marks the session used NOW, so a request arriving at the edge of the idle timeout is
// not reaped mid-flight.
func (m *Manager) acquire(name string) (*Session, func(), error) {
	m.mu.Lock()
	s := m.live[name]
	if s == nil {
		m.mu.Unlock()
		return nil, nil, fmt.Errorf("lifecycle: model %q was just unloaded", name)
	}
	s.refs++
	m.mu.Unlock()
	s.touch(time.Now())
	return s, func() { m.release(s) }, nil
}

func (m *Manager) release(s *Session) {
	m.mu.Lock()
	s.refs--
	closeNow := s.retired && s.refs == 0
	m.mu.Unlock()
	if closeNow {
		_ = s.close()
	}
}

// retireLocked removes name from the live map (caller holds m.mu). It returns the session when
// nobody is using it — the caller must then close it, OUTSIDE m.mu, since closing a GPU session
// can take a while and m.mu gates every request. A session still in use is left to the last
// release to close.
func (m *Manager) retireLocked(name string) *Session {
	s := m.live[name]
	if s == nil {
		return nil
	}
	delete(m.live, name)
	s.retired = true
	if s.refs > 0 {
		return nil
	}
	return s
}
