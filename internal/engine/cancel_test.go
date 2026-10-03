package engine

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// fakeWorkerSession is a Session whose worker runs jobs without ORT (as in
// TestJobPanicBecomesErrorAndWorkerSurvives). Close stops the worker.
func fakeWorkerSession(t *testing.T) *Session {
	t.Helper()
	s := &Session{jobs: make(chan func()), closeErr: make(chan error, 1)}
	go func() {
		for fn := range s.jobs {
			fn()
		}
		s.closeErr <- nil
	}()
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// blockWorker submits a job that holds s's worker until the returned release is called. It returns
// once the job is running.
func blockWorker(t *testing.T, s *Session) (release func(), done <-chan error) {
	t.Helper()
	running, proceed := make(chan struct{}), make(chan struct{})
	errc := make(chan error, 1)
	go func() {
		_, err := s.submit(context.Background(), func() ([]Tensor, error) {
			close(running)
			<-proceed
			return nil, nil
		})
		errc <- err
	}()
	<-running
	return func() { close(proceed) }, errc
}

// A caller queued behind a running job gives up when its ctx ends: it returns promptly with an
// error wrapping the ctx error, and its job never runs. The session keeps serving afterwards.
func TestSubmitWaiterGivesUpOnCancel(t *testing.T) {
	s := fakeWorkerSession(t)
	release, firstDone := blockWorker(t, s)

	ctx, cancel := context.WithCancel(context.Background())
	var ran atomic.Bool
	waiter := make(chan error, 1)
	go func() {
		_, err := s.submit(ctx, func() ([]Tensor, error) { ran.Store(true); return nil, nil })
		waiter <- err
	}()
	time.Sleep(20 * time.Millisecond) // let it queue behind the running job
	cancel()
	select {
	case err := <-waiter:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled waiter: err = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled waiter still waiting for the worker")
	}

	release()
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
	if ran.Load() {
		t.Fatal("the cancelled caller's job ran")
	}
	if _, err := s.submit(context.Background(), func() ([]Tensor, error) { return nil, nil }); err != nil {
		t.Fatalf("session after a cancelled waiter: %v", err)
	}
}

// A ctx already done never starts a job, even on an idle worker.
func TestSubmitWithDoneContextRunsNothing(t *testing.T) {
	s := fakeWorkerSession(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var ran atomic.Bool
	for i := 0; i < 50; i++ { // the select would pick an idle worker half of the time
		if _, err := s.submit(ctx, func() ([]Tensor, error) { ran.Store(true); return nil, nil }); !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	}
	if ran.Load() {
		t.Fatal("a job ran for a context that was already done")
	}
}

// A pool caller waiting for a free member gives up on cancel, and the member it did not get is not
// lost: the pool still has all its members afterwards.
func TestPoolWaiterGivesUpOnCancel(t *testing.T) {
	member := fakeWorkerSession(t)
	pool := NewSessionPool([]*Session{member})

	// Hold the only member: take it as Run would, and give it back on release.
	held, err := pool.take(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	waiter := make(chan error, 1)
	go func() { _, err := pool.take(ctx); waiter <- err }()
	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case err := <-waiter:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled pool waiter: err = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled pool waiter still waiting")
	}
	pool.ch <- held

	if n := len(pool.ch); n != pool.Size() {
		t.Fatalf("pool has %d free members after a cancelled waiter, want %d", n, pool.Size())
	}
	if _, err := pool.take(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("take with a done ctx and a free member: err = %v, want context.Canceled", err)
	}
	if n := len(pool.ch); n != pool.Size() {
		t.Fatalf("a refused take kept a member: %d free, want %d", n, pool.Size())
	}
}
