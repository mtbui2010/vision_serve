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
