package lifecycle

import "time"

// reaperInterval is the interval for scanning idle models to auto-unload.
const reaperInterval = 30 * time.Second

// reaper periodically releases models idle longer than idleTimeout (idleTimeout<=0 = never).
func (m *Manager) reaper() {
	t := time.NewTicker(reaperInterval)
	defer t.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-t.C:
			// Closed outside m.mu: destroying a GPU session takes time and m.mu gates every request.
			for _, s := range m.expireIdle(time.Now()) {
				_ = s.close()
			}
		}
	}
}

// expireIdle retires every live model that has been idle longer than its idle timeout and returns
// the retired sessions; the caller closes them outside m.mu. A session serving a request is never
// idle, however old its lastUsed.
func (m *Manager) expireIdle(now time.Time) []*Session {
	var expired []*Session
	m.mu.Lock()
	defer m.mu.Unlock()
	for name, s := range m.live {
		if s.refs == 0 && s.idleTimeout > 0 && s.idleFor(now) > s.idleTimeout {
			if r := m.retireLocked(name); r != nil {
				expired = append(expired, r)
			}
		}
	}
	return expired
}
