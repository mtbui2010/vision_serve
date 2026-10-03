package lifecycle

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
)

// Admission control bounds how many requests one model may hold at a time — running plus waiting
// for a session — so memory stays bounded under load. Without it every request queued behind a
// busy model keeps its decoded upload (up to 40 MP, ~160 MB) alive while it waits.
//
// The bound per model is, in order of precedence:
//
//   - SetMaxQueue(n), or the environment variable VISIONSERVE_MAX_QUEUE=n read by NewManager:
//     n >= 1 admits at most n requests per model; n = 0 turns admission control off (unbounded,
//     the behaviour before it existed — e.g. for a pool×concurrency benchmark sweep). Anything
//     else is ignored with a warning.
//   - otherwise automatic: 2 × the model's inference slots (the largest session pool among its
//     roles; 1 for a single session), and never below defaultMinQueue.
//     A model that is not loaded yet counts 1 slot.
//
// A refused request fails at once with ErrOverloaded (HTTP 503) — it never waits for a slot.
const defaultMinQueue = 4

// maxQueue values other than a positive bound (see Manager.maxQueue).
const (
	queueAuto      = 0  // automatic bound (the zero value)
	queueUnbounded = -1 // admission control off
)

// SetMaxQueue sets the per-model admission bound, overriding VISIONSERVE_MAX_QUEUE: n >= 1 bounds
// every model at n requests (running + waiting), 0 disables admission control, and n < 0 restores
// the automatic bound. Call once at startup, before serving requests.
func (m *Manager) SetMaxQueue(n int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	switch {
	case n > 0:
		m.maxQueue = n
	case n == 0:
		m.maxQueue = queueUnbounded
	default:
		m.maxQueue = queueAuto
	}
}

// maxQueueFromEnv reads VISIONSERVE_MAX_QUEUE into a Manager.maxQueue value.
func maxQueueFromEnv() int {
	v := strings.TrimSpace(os.Getenv("VISIONSERVE_MAX_QUEUE"))
	if v == "" {
		return queueAuto
	}
	n, err := strconv.Atoi(v)
	switch {
	case err != nil || n < 0:
		log.Printf("lifecycle: ignoring VISIONSERVE_MAX_QUEUE=%q (want an integer >= 0; 0 = unbounded) — using the automatic bound", v)
		return queueAuto
	case n == 0:
		return queueUnbounded
	default:
		return n
	}
}

// Admit reserves a slot for one request on model name BEFORE the server decodes the upload, so
// memory does not grow with the number of requests queued behind a busy model. The caller must
// call release when the request is done; release is idempotent (a second call does nothing), and
// it is a no-op func — never nil — when Admit fails. It returns an error wrapping ErrOverloaded,
// immediately, when the model already holds its bound (see defaultMinQueue for the bound).
//
// Admit does not load the model and does not check the name: an unknown model is admitted and
// then fails with ErrModelNotFound on Load. The bookkeeping entry is dropped with the last
// release, so made-up names do not accumulate.
func (m *Manager) Admit(name string) (release func(), err error) {
	m.mu.Lock()
	limit := m.admitLimitLocked(name)
	n := m.admitted[name]
	if limit > 0 && n >= limit {
		m.mu.Unlock()
		return func() {}, fmt.Errorf("lifecycle: %w: %q already has %d requests running or waiting (limit %d per model; "+
			"set VISIONSERVE_MAX_QUEUE to change it)", ErrOverloaded, name, n, limit)
	}
	if m.admitted == nil {
		m.admitted = map[string]int{}
	}
	m.admitted[name] = n + 1
	m.mu.Unlock()

	var once sync.Once
	return func() { once.Do(func() { m.unadmit(name) }) }, nil
}

func (m *Manager) unadmit(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if n := m.admitted[name] - 1; n > 0 {
		m.admitted[name] = n
	} else {
		delete(m.admitted, name)
	}
}

// admitLimitLocked returns the bound for name (<= 0 = unbounded). Caller holds m.mu.
func (m *Manager) admitLimitLocked(name string) int {
	switch {
	case m.maxQueue > 0:
		return m.maxQueue
	case m.maxQueue == queueUnbounded:
		return 0
	}
	slots := 1
	if s := m.live[name]; s != nil {
		slots = s.slots()
	}
	return max(defaultMinQueue, 2*slots)
}
