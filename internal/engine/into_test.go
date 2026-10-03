package engine

import (
	"context"
	"errors"
	"testing"
	"time"
)

// RunNamedInto writes a named output straight into the caller's buffer (the returned tensor
// aliases it, values as RunNamed), on a lone session and on a pool, and refuses a buffer it
// cannot honour instead of silently allocating.
func TestRealRunNamedInto(t *testing.T) {
	path := identityModel(t)
	s, err := NewSession(path, []string{"x"}, []string{"y"}, []Provider{ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	s2, err := NewSession(path, []string{"x"}, []string{"y"}, []Provider{ProviderCPU})
	if err != nil {
		s.Close()
		t.Fatal(err)
	}
	pool := NewSessionPool([]*Session{s2})
	defer s.Close()
	defer pool.Close()

	in := map[string]Tensor{"x": F32([]float32{4, 5, 6}, 1, 3)}
	for name, r := range map[string]IntoRunnable{"session": s, "pool": pool} {
		buf := []float32{-1, -1, -1}
		outs, err := r.RunNamedInto(context.Background(), in, map[string]Tensor{"y": F32(buf, 1, 3)})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(outs) != 1 || &outs[0].Data[0] != &buf[0] {
			t.Fatalf("%s: output does not alias the caller buffer", name)
		}
		if buf[0] != 4 || buf[1] != 5 || buf[2] != 6 || len(outs[0].Shape) != 2 || outs[0].Shape[1] != 3 {
			t.Fatalf("%s: buffer %v shape %v, want [4 5 6] [1 3]", name, buf, outs[0].Shape)
		}

		// nil / empty into is RunNamed: a fresh copy.
		outs, err = r.RunNamedInto(context.Background(), in, nil)
		if err != nil || len(outs) != 1 || outs[0].Data[2] != 6 {
			t.Fatalf("%s: nil into: %v %v", name, outs, err)
		}

		for what, into := range map[string]map[string]Tensor{
			"wrong shape":    {"y": F32(make([]float32, 4), 1, 4)},
			"unknown output": {"z": F32(make([]float32, 3), 1, 3)},
			"int64 buffer":   {"y": I64(make([]int64, 3), 1, 3)},
			"empty buffer":   {"y": F32(nil, 1, 3)},
		} {
			if _, err := r.RunNamedInto(context.Background(), in, into); err == nil {
				t.Errorf("%s: %s accepted", name, what)
			}
		}
	}
}

// RunNamedInto follows RunNamed's ctx contract: a done ctx never starts the run (even on an idle
// worker or a free pool member), and a caller queued behind a running job gives up on cancel
// without its job ever running — so nothing writes into its buffer after it returns.
func TestRunNamedIntoCancel(t *testing.T) {
	named := func(s *Session) *Session {
		s.inputNames, s.outputNames = []string{"x"}, []string{"y"}
		return s
	}
	in := map[string]Tensor{"x": F32([]float32{1, 2, 3}, 1, 3)}
	poison := func() []float32 { return []float32{-7, -7, -7} }
	untouched := func(b []float32) bool { return b[0] == -7 && b[1] == -7 && b[2] == -7 }

	done, cancel := context.WithCancel(context.Background())
	cancel()
	s := named(fakeWorkerSession(t))
	pool := NewSessionPool([]*Session{named(fakeWorkerSession(t))})
	for name, r := range map[string]IntoRunnable{"session": s, "pool": pool} {
		for i := 0; i < 50; i++ { // the select would pick an idle worker half of the time
			buf := poison()
			if _, err := r.RunNamedInto(done, in, map[string]Tensor{"y": F32(buf, 1, 3)}); !errors.Is(err, context.Canceled) {
				t.Fatalf("%s, done ctx: err = %v, want context.Canceled", name, err)
			}
			if !untouched(buf) {
				t.Fatalf("%s, done ctx: buffer written %v", name, buf)
			}
		}
	}

	// Queued behind a running job on the session.
	release, firstDone := blockWorker(t, s)
	ctx, cancelQ := context.WithCancel(context.Background())
	buf := poison()
	waiter := make(chan error, 1)
	go func() {
		_, err := s.RunNamedInto(ctx, in, map[string]Tensor{"y": F32(buf, 1, 3)})
		waiter <- err
	}()
	time.Sleep(20 * time.Millisecond)
	cancelQ()
	select {
	case err := <-waiter:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("queued caller: err = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled RunNamedInto still waiting for the worker")
	}
	release()
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
	if _, err := s.submit(context.Background(), func() ([]Tensor, error) { return nil, nil }); err != nil {
		t.Fatal(err) // the worker has drained everything queued before this job
	}
	if !untouched(buf) {
		t.Fatalf("the cancelled caller's run wrote its buffer: %v", buf)
	}

	// Waiting for a pool member.
	held, err := pool.take(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancelP := context.WithCancel(context.Background())
	go func() {
		_, err := pool.RunNamedInto(ctx, in, map[string]Tensor{"y": F32(poison(), 1, 3)})
		waiter <- err
	}()
	time.Sleep(20 * time.Millisecond)
	cancelP()
	select {
	case err := <-waiter:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("pool waiter: err = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled pool RunNamedInto still waiting")
	}
	pool.ch <- held
	if n := len(pool.ch); n != pool.Size() {
		t.Fatalf("pool has %d free members after a cancelled RunNamedInto, want %d", n, pool.Size())
	}
}
