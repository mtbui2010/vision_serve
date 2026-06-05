package engine

// Runnable is satisfied by both *Session and *SessionPool, so lifecycle can hold
// either behind the same interface without knowing which is which.
type Runnable interface {
	Run(inputs []Tensor) ([]Tensor, error)
	RunNamed(inputs map[string]Tensor) ([]Tensor, error)
	InputNames() []string
	OutputNames() []string
	ActiveEP() Provider
	Close() error
}

// SessionPool wraps N identical ONNX sessions as a bounded concurrency pool.
// A caller that calls RunNamed blocks only when all N sessions are busy, then
// runs as soon as one becomes free. The pool is goroutine-safe.
//
// Used by lifecycle to give AMG-style models (MobileSAM decoder, Grounded-SAM
// decoder) multiple concurrent inference slots instead of serializing all
// goroutines on a single session mutex.
type SessionPool struct {
	ch          chan *Session
	inputNames  []string
	outputNames []string
	activeEP    Provider
}

// NewSessionPool builds a pool from pre-created sessions. All sessions must use
// the same ONNX graph (identical input/output names and execution provider).
func NewSessionPool(sessions []*Session) *SessionPool {
	ch := make(chan *Session, len(sessions))
	for _, s := range sessions {
		ch <- s
	}
	p := &SessionPool{ch: ch}
	if len(sessions) > 0 {
		p.inputNames = sessions[0].InputNames()
		p.outputNames = sessions[0].OutputNames()
		p.activeEP = sessions[0].ActiveEP()
	}
	return p
}

func (p *SessionPool) Run(inputs []Tensor) ([]Tensor, error) {
	s := <-p.ch
	defer func() { p.ch <- s }()
	return s.Run(inputs)
}

func (p *SessionPool) RunNamed(inputs map[string]Tensor) ([]Tensor, error) {
	s := <-p.ch
	defer func() { p.ch <- s }()
	return s.RunNamed(inputs)
}

func (p *SessionPool) InputNames() []string  { return p.inputNames }
func (p *SessionPool) OutputNames() []string { return p.outputNames }
func (p *SessionPool) ActiveEP() Provider    { return p.activeEP }

// Close drains the pool and destroys every session. Blocks until all in-flight
// sessions have been returned — call only after all requests are done (e.g. from
// lifecycle.Manager.Close after the server shuts down).
func (p *SessionPool) Close() error {
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
