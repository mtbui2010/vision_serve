//go:build linux

package engine

import (
	"bytes"
	"io"
	"os"
	"sync"

	"golang.org/x/sys/unix"
)

// stderrMu serializes the (process-wide) fd 2 redirection across session creations.
var stderrMu sync.Mutex

// captureStderr runs fn while temporarily redirecting fd 2 (including ORT's C/C++ side) into
// a pipe, collected into a buffer. It returns the captured stderr content + fn's error.
//
// Used to swallow ORT's "red" logs on GPU EP fallback (but still reprints them if fn truly fails).
// If setting up the redirection fails, fn runs normally (without capture) — which is safe.
func captureStderr(fn func() error) (string, error) {
	stderrMu.Lock()
	defer stderrMu.Unlock()

	r, w, err := os.Pipe()
	if err != nil {
		return "", fn()
	}
	saved, err := unix.Dup(2) // save the original fd 2
	if err != nil {
		r.Close()
		w.Close()
		return "", fn()
	}
	if err := unix.Dup3(int(w.Fd()), 2, 0); err != nil { // fd 2 -> pipe write end
		unix.Close(saved)
		r.Close()
		w.Close()
		return "", fn()
	}

	// continuously drain in a goroutine to avoid filling the pipe buffer and blocking.
	var buf bytes.Buffer
	done := make(chan struct{})
	go func() {
		_, _ = io.Copy(&buf, r)
		close(done)
	}()

	runErr := fn()

	_ = unix.Dup3(saved, 2, 0) // restore the original fd 2
	unix.Close(saved)
	w.Close()
	<-done
	r.Close()
	return buf.String(), runErr
}
