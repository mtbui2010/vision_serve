//go:build linux

package engine

import (
	"bufio"
	"io"
	"os"
	"strings"
	"sync"

	"golang.org/x/sys/unix"
)

// stderrMu serializes the (process-wide) fd 2 redirection across session creations. See stderr.go
// for why the redirection exists and why it cannot be shared between overlapping creations.
var stderrMu sync.Mutex

// captureStderr runs fn while fd 2 (including ORT's C/C++ side) points into a pipe. Lines in ORT's
// log format are collected and returned with fn's error; every other line is written through to
// the original stderr as soon as it arrives. fd 2 is restored even if fn panics.
//
// If setting up the redirection fails, fn runs normally (without capture) — which is safe.
func captureStderr(fn func() error) (captured string, err error) {
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
	original := os.NewFile(uintptr(saved), "stderr")

	// Drain continuously (a full pipe would block ORT), routing line by line.
	var kept strings.Builder // ORT's lines
	done := make(chan struct{})
	go func() {
		defer close(done)
		route(r, original, &kept)
	}()

	// Restore in a defer: fn may panic (the panic report must reach the real stderr) and the
	// lock must be released either way. captured is read only once the drain has finished.
	defer func() {
		_ = unix.Dup3(saved, 2, 0) // fd 2 -> the original stderr again
		w.Close()                  // with fd 2 restored this was the last write end: the drain sees EOF
		<-done
		r.Close()
		original.Close()
		captured = kept.String()
	}()
	return "", fn()
}

// route copies src line by line: ORT log lines into kept, everything else to passthrough.
func route(src io.Reader, passthrough io.Writer, kept *strings.Builder) {
	br := bufio.NewReader(src)
	for {
		line, err := br.ReadString('\n')
		if line != "" {
			if isORTLogLine(line) {
				kept.WriteString(line)
			} else {
				_, _ = io.WriteString(passthrough, line)
			}
		}
		if err != nil {
			return
		}
	}
}
