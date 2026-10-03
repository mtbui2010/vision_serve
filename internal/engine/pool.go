package engine

import (
	"context"
	"sync"
)

// Runnable is satisfied by both *Session and *SessionPool, so lifecycle can hold
// either behind the same interface without knowing which is which.
//
// ctx bounds the WAIT for the session, not the inference: a call whose ctx is done before it gets
// the session (the worker thread of a single session, a free member of a pool) returns an error
// wrapping ctx.Err() and runs nothing. Once ONNX Runtime has the job it runs to the end (a Run
// cannot be interrupted) and the call returns its result.
type Runnable interface {
	Run(ctx context.Context, inputs []Tensor) ([]Tensor, error)
	RunNamed(ctx context.Context, inputs map[string]Tensor) ([]Tensor, error)
	InputNames() []string
	OutputNames() []string
	ActiveEP() Provider
	Close() error
}

// IntoRunnable is a Runnable that can write chosen outputs into caller-owned buffers
// (Session.RunNamedInto), under RunNamed's ctx contract. Both *Session and *SessionPool
// implement it.
type IntoRunnable interface {
	RunNamedInto(ctx context.Context, inputs, into map[string]Tensor) ([]Tensor, error)
}

var (
	_ IntoRunnable = (*Session)(nil)
	_ IntoRunnable = (*SessionPool)(nil)
)

// SessionPool wraps N identical ONNX sessions as a bounded concurrency pool.
// A caller that calls RunNamed blocks only when all N sessions are busy, then
// runs as soon as one becomes free. The pool is goroutine-safe.
//
// Used by lifecycle to give AMG-style models (MobileSAM decoder, Grounded-SAM
// decoder) multiple concurrent inference slots instead of serializing all
// goroutines on a single session mutex.
type SessionPool struct {
	ch          chan *Session
	done        chan struct{} // closed by Close: waiting Run calls return ErrClosed
	closeOnce   sync.Once
	inputNames  []string
	outputNames []string
	// eps is each member's active EP. Members are created from the same graph and EP chain, but
	// each creation falls back on its own: one copy can land on the CPU (a CUDA out-of-memory on
	// the 4th decoder) while the others run on the GPU.
	eps []Provider
}

// NewSessionPool builds a pool from pre-created sessions. All sessions must use
// the same ONNX graph (identical input/output names).
func NewSessionPool(sessions []*Session) *SessionPool {
	ch := make(chan *Session, len(sessions))
	for _, s := range sessions {
		ch <- s
	}
	p := &SessionPool{ch: ch, done: make(chan struct{})}
	if len(sessions) > 0 {
		p.inputNames = sessions[0].InputNames()
		p.outputNames = sessions[0].OutputNames()
	}
	for _, s := range sessions {
		p.eps = append(p.eps, s.ActiveEP())
	}
	return p
}

func (p *SessionPool) Run(ctx context.Context, inputs []Tensor) ([]Tensor, error) {
	s, err := p.take(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { p.ch <- s }()
	return s.Run(ctx, inputs)
}

func (p *SessionPool) RunNamed(ctx context.Context, inputs map[string]Tensor) ([]Tensor, error) {
	s, err := p.take(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { p.ch <- s }()
	return s.RunNamed(ctx, inputs)
}

// RunNamedInto is Session.RunNamedInto on a free member of the pool, with the same ctx contract
// as RunNamed: a call whose ctx ends before it gets a member runs nothing.
func (p *SessionPool) RunNamedInto(ctx context.Context, inputs, into map[string]Tensor) ([]Tensor, error) {
	s, err := p.take(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { p.ch <- s }()
	return s.RunNamedInto(ctx, inputs, into)
}

// take borrows a free session, or fails once the pool is closed or ctx is done. Without the done
// case a call waiting for a slot while Close drained the pool blocked forever; without the ctx case
// a request whose client had left kept its place in the queue and then ran.
func (p *SessionPool) take(ctx context.Context) (*Session, error) {
	select {
	case <-p.done:
		return nil, ErrClosed
	default:
	}
	if err := ctx.Err(); err != nil { // a free member must not win over a ctx already done
		return nil, gaveUp(err)
	}
	select {
	case s := <-p.ch:
		return s, nil
	case <-p.done:
		return nil, ErrClosed
	case <-ctx.Done():
		return nil, gaveUp(ctx.Err())
	}
}

// Size is the number of sessions in the pool: how many inferences it runs at once.
func (p *SessionPool) Size() int { return cap(p.ch) }

func (p *SessionPool) InputNames() []string  { return p.inputNames }
func (p *SessionPool) OutputNames() []string { return p.outputNames }

// ActiveEP reports the EP every request on the pool is guaranteed: the members' common EP, or,
// when they differ, the slowest one among them (a request may be served by that member). Use
// ActiveEPs or RunnableDevice for the full picture.
func (p *SessionPool) ActiveEP() Provider { return slowestEP(p.eps) }

// ActiveEPs returns each member's active EP, in creation order.
func (p *SessionPool) ActiveEPs() []Provider { return append([]Provider(nil), p.eps...) }

// Close drains the pool and destroys every session. Blocks until all in-flight
// sessions have been returned — call only after all requests are done (e.g. from
// lifecycle.Manager.Close after the server shuts down).
func (p *SessionPool) Close() error {
	p.closeOnce.Do(func() { close(p.done) })
	n := cap(p.ch)
	var firstErr error
	for i := 0; i < n; i++ {
		s := <-p.ch
		if err := s.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
