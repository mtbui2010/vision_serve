//go:build !linux

package engine

// captureStderr on non-linux: does not redirect fd 2 (only linux needs to swallow ORT's
// EP noise). Runs fn directly.
func captureStderr(fn func() error) (string, error) {
	return "", fn()
}
