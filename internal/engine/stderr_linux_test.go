//go:build linux

package engine

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// redirectTestStderr points fd 2 at a file for the rest of the test (restored on cleanup) and
// returns a function reading what reached it. It stands for the terminal: what the capture
// re-emits must land here, what it keeps must not.
func redirectTestStderr(t *testing.T) func() string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "stderr.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	saved, err := unix.Dup(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := unix.Dup3(int(f.Fd()), 2, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = unix.Dup3(saved, 2, 0)
		unix.Close(saved)
		f.Close()
	})
	return func() string {
		b, _ := os.ReadFile(path)
		return string(b)
	}
}

const (
	ortLine   = "2026-10-03 09:15:02.123456789 [E:onnxruntime:Default, provider_bridge_ort.cc:1848 TryGetProviderInfo_CUDA] Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file\n"
	otherLine = "2026/10/03 09:15:02 preloaded: rf-detr\n"
)

// The capture exists to hold ORT's own EP-fallback messages; it used to swallow EVERYTHING written
// to fd 2 while a session was created (minutes, for a TensorRT build), including other
// goroutines' logs. Those must reach the real stderr, and promptly — not after the build.
func TestCaptureStderrKeepsORTLinesAndPassesTheRestThrough(t *testing.T) {
	read := redirectTestStderr(t)

	captured, err := captureStderr(func() error {
		os.Stderr.WriteString(ortLine)
		os.Stderr.WriteString(otherLine)
		// Passed through while fn is still running (another goroutine's log during a long build).
		deadline := time.Now().Add(3 * time.Second)
		for !strings.Contains(read(), otherLine) {
			if time.Now().After(deadline) {
				return errors.New("a non-ORT line was held back until the capture ended")
			}
			time.Sleep(time.Millisecond)
		}
		os.Stderr.WriteString("partial line without a newline")
		return errors.New("fn's own error")
	})
	if err == nil || err.Error() != "fn's own error" {
		t.Fatalf("captureStderr returned err %v, want fn's error", err)
	}
	if captured != ortLine {
		t.Errorf("captured %q, want only the ORT line", captured)
	}
	out := read()
	if !strings.Contains(out, otherLine) || !strings.Contains(out, "partial line without a newline") {
		t.Errorf("non-ORT output did not reach stderr: %q", out)
	}
	if strings.Contains(out, "onnxruntime") {
		t.Errorf("ORT's line leaked to stderr although it was captured: %q", out)
	}
}

// fd 2 used to be restored only on the normal return: a panic in fn left the process writing its
// stderr — including the panic report itself — into a dead pipe, and kept the global lock.
func TestCaptureStderrRestoresStderrAfterAPanic(t *testing.T) {
	read := redirectTestStderr(t)

	func() {
		defer func() { _ = recover() }()
		_, _ = captureStderr(func() error { panic("boom inside session creation") })
	}()

	os.Stderr.WriteString("after the panic\n")
	if out := read(); !strings.Contains(out, "after the panic") {
		t.Fatalf("stderr was not restored after a panic: %q", out)
	}

	done := make(chan struct{})
	go func() {
		_, _ = captureStderr(func() error { return nil })
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the capture lock was never released after a panic")
	}
}
