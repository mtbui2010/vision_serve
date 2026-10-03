package lifecycle

import "errors"

// Typed errors: the contract between the runtime and the HTTP layer. The runtime wraps them
// (fmt.Errorf("...: %w", ErrModelNotFound)); the server maps them with errors.Is to a status code,
// so a missing model is a 404, a bad prompt a 400 and a full queue a 503 — not all 500.
var (
	// ErrModelNotFound: the name is not in the registry, or its weights are missing.
	ErrModelNotFound = errors.New("model not found")
	// ErrInvalidRequest: the request itself is wrong (prompt, template, option values).
	ErrInvalidRequest = errors.New("invalid request")
	// ErrOverloaded: admission control refused the request (too many waiting for this model).
	ErrOverloaded = errors.New("model overloaded")
)

// Admit reserves a slot for one request on model name BEFORE the server decodes the upload, so
// memory does not grow with the number of requests queued behind a busy model. The caller must
// call release when the request is done. It returns an error wrapping ErrOverloaded when the
// model's queue is full.
//
// CONTRACT STUB (refactor wave 1): always admits. The runtime stream replaces the body with a
// real per-model bound; the HTTP stream already calls it. Keep the signature.
func (m *Manager) Admit(name string) (release func(), err error) {
	return func() {}, nil
}
