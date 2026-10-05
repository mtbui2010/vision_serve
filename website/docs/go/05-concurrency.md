# 5. Concurrency

!!! abstract "What you'll learn"
    - Goroutines, channels and `select`, Go's tools for doing many things at once.
    - `sync.Mutex`, `sync.Once` and `context.Context`.
    - How they are used in the four places that make VisionServe safe under load: the
      engine's per-session worker thread, single-flight model loading, admission control,
      and leases with the idle reaper.
    - What a data race is, and how `go test -race` finds one.

The server answers many requests at the same time. Go's `net/http` runs **every request in
its own goroutine**, so every handler, and everything it calls, may run in parallel with
itself. CLAUDE.md warns: "access to a model session must be thread-safe. Sessions run on
their own OS-locked worker goroutine; never call one from an arbitrary goroutine under a
mutex." This chapter shows how the code gets it right, and why a mutex is not enough.

## Goroutines

`go f(x)` starts `f(x)` running concurrently and returns immediately. A goroutine is very
cheap (a few KB of stack), so the server can have thousands. Go's runtime spreads them over
all CPU cores, moving them between OS threads as it likes.

=== "Go"

    ```go
    go m.reaper()            // runs in the background until the program ends
    ```

=== "Python"

    ```python
    threading.Thread(target=m.reaper, daemon=True).start()
    # but the GIL lets only one thread run Python code at a time
    ```

The lifecycle manager starts its idle reaper this way
([manager.go#L62-L75](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/manager.go#L62-L75)).

## Channels and `select`

A **channel** is a typed pipe between goroutines: `ch <- v` sends, `v := <-ch` receives.
On an *unbuffered* channel (`make(chan T)`) the sender waits until a receiver takes the
value: it is a hand-off. A *buffered* channel (`make(chan T, n)`) holds up to `n` values.
`close(ch)` says "no more values"; a `for v := range ch` loop then ends, and every receive
on a closed channel returns at once. Closing is therefore a cheap way to **broadcast** "done"
to any number of waiters.

MobileSAM's automatic mask generator uses a channel to hand indices to a fixed number of
workers, a classic worker pool:

```go title="internal/models/mobilesam/automask.go (lines 291-313)"
// parallelForW is parallelFor that also tells fn which worker (0..workers-1) runs item i, so a
// worker can reuse state of its own across its items.
func parallelForW(n, workers int, fn func(w, i int)) {
	if workers > n {
		workers = n
	}
	var wg sync.WaitGroup
	next := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range next {
				fn(w, i)
			}
		}(w)
	}
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Wait()
}
```

[automask.go#L291-L313 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/mobilesam/automask.go#L291-L313)

`sync.WaitGroup` counts running goroutines; `wg.Wait()` blocks until all called `Done`. In
Python this is `ThreadPoolExecutor(max_workers).map(fn, range(n))`. The worker number `w`
lets each worker keep buffers of its own across the items it handles: the full-resolution
pass gives every worker one mask buffer that the decoder writes into
([automask.go#L238-L248](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/mobilesam/automask.go#L238-L248)),
instead of allocating a new one (30 MB at 3200×2400) per mask. Since only one goroutine ever uses
`logits[w]`, no lock is needed. The plain `parallelFor(n, workers, fn)` just drops `w`.

`select` waits on several channel operations and runs the first one that is ready. The
idle reaper wakes up every 30 seconds, or stops when the manager closes `m.stop`:

```go title="internal/lifecycle/reaper.go (lines 8-23)"
// reaper periodically releases models idle longer than idleTimeout (idleTimeout<=0 = never).
func (m *Manager) reaper() {
	t := time.NewTicker(reaperInterval)
	defer t.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-t.C:
			// Closed outside m.mu: destroying a GPU session takes time and m.mu gates every request.
			for _, s := range m.expireIdle(time.Now()) {
				_ = s.close()
			}
		}
	}
}
```

[reaper.go#L8-L23 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/reaper.go#L8-L23)

## Mutexes and `sync.Once`

When several goroutines read and write the same map or counter, they must take turns. A
`sync.Mutex` is a lock: `Lock()` waits until no one else holds it. The manager guards all
its bookkeeping with one:

```go title="internal/lifecycle/manager.go (lines 30-60, trimmed)"
// Manager holds the live models and coordinates thread-safe load/unload.
type Manager struct {
	reg  *registry.Registry
	tmpl *templates.Store // nil = no template support

	mu   sync.Mutex
	live map[string]*Session
	// loading holds the in-progress load of each model being loaded (see Load).
	loading    map[string]*loadCall
	// ...
	admitted map[string]int
	maxQueue int
	// ...
}
```

[manager.go#L30-L60 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/manager.go#L30-L60)

The convention is that the fields declared after `mu` are the ones it protects. The rule the
code follows everywhere: **hold the lock briefly, never across slow work**. Closing a GPU
session takes a while, so the reaper above collects the sessions under the lock
(`expireIdle`) and closes them *after* releasing it.

`sync.Once` runs something exactly once, however many goroutines call it at the same time.
ONNX Runtime must be initialised once per process:

```go title="internal/engine/ort.go (lines 42-56)"
// ensureORT initializes the ORT environment exactly once per process.
func ensureORT() error {
	initOnce.Do(func() {
		if path := os.Getenv("ORT_DYLIB_PATH"); path != "" {
			ort.SetSharedLibraryPath(path)
			ortLibPath = path
		}
		// If ORT_DYLIB_PATH is not set, the binding locates the library via the OS
		// default mechanism (LD_LIBRARY_PATH). Report a clear error if init fails.
		if err := ort.InitializeEnvironment(); err != nil {
			initErr = fmt.Errorf("engine: failed to initialize ONNX Runtime (set ORT_DYLIB_PATH to libonnxruntime.so?): %w", err)
		}
	})
	return initErr
}
```

[ort.go#L42-L56 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/ort.go#L42-L56)

## The engine: one OS thread per ONNX session

This is the most important concurrency design in the project, and the one with the least
obvious reason.

**The problem.** ONNX Runtime's CUDA provider creates GPU resources (a cuBLAS/cuDNN handle,
a memory arena) **per OS thread**, the first time a thread runs the session. Go moves
goroutines between OS threads freely. With only a mutex, each inference could run on a
different thread and leak a new CUDA context every time. The comment on the type records
what happened:

```go title="internal/engine/ort.go (lines 139-166, trimmed)"
// Session is a live ONNX session. Thread-safe AND OS-thread-pinned: every call to the
// underlying ORT session (create, Run, Destroy) is funnelled onto ONE dedicated OS thread
// owned by this Session's worker goroutine (see worker / submit).
//
// Why pinned, not just mutex-serialized: ORT's CUDA execution provider allocates GPU
// resources PER OS THREAD (a cublas + cudnn handle and a memory arena, created lazily on the
// first Run seen on each thread). Go freely migrates a goroutine across OS threads between
// calls — and cgo calls in particular spawn fresh threads — so a plain mutex would let each
// inference land on a different thread, leaking a new CUDA context every time until
// `cublasCreate` fails with "CUBLAS failure 3: the resource allocation failed" after a few
// requests. Pinning to one thread means exactly one CUDA per-thread context per session for
// its whole lifetime. (see CLAUDE.md: session access must be thread-safe + VRAM-safe.)
type Session struct {
	sess        *ort.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
	activeEP    Provider // EP that was actually loaded (first one whose libs were available)

	jobs chan func() // work funnelled onto the dedicated OS thread; closed by Close
	// ...
	jobsMu    sync.RWMutex
	closed    bool
	closeOnce sync.Once
	closeErr  chan error // worker sends the Destroy() result here after jobs drains
}
```

[ort.go#L139-L166 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/ort.go#L139-L166)

**The solution.** Each `Session` starts one goroutine, the *worker*, and pins it to its OS
thread with `runtime.LockOSThread()`. The worker creates the ORT session, then runs every
job sent on the `jobs` channel, then destroys the session, all on that one thread:

```go title="internal/engine/ort.go (lines 234-261)"
// worker owns the session's single OS thread for its entire lifetime: it creates the ORT
// session, runs every job serially, and destroys the session — all on the same locked thread.
// This is what keeps ORT's CUDA EP to one per-thread context per session (see Session doc).
func (s *Session) worker(modelPath string, providers []Provider, so SessionOptions, ready chan<- error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	sess, ep, err := func() (sess *ort.DynamicAdvancedSession, ep Provider, err error) {
		defer func() {
			if p := recover(); p != nil {
				sess, err = nil, recoverJob(p)
			}
		}()
		return createORTSession(modelPath, s.inputNames, s.outputNames, providers, so)
	}()
	if err != nil {
		ready <- err
		return
	}
	s.sess = sess
	s.activeEP = ep
	ready <- nil // happens-before NewSession's return: s.sess/s.activeEP are safely published

	for fn := range s.jobs {
		fn()
	}
	s.closeErr <- s.sess.Destroy() // jobs closed by Close: destroy on the same locked thread
}
```

[ort.go#L234-L261 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/ort.go#L234-L261)

`chan<- error` is a *send-only* channel: the worker may only send on `ready`. `NewSession`
waits on `<-ready` so it returns only once the session exists
([ort.go#L226-L231](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/ort.go#L226-L231)).

A request goroutine never touches ORT. `Run` wraps the work in a closure and **submits** it:

```go title="internal/engine/ort.go (lines 543-581)"
func (s *Session) submit(ctx context.Context, work func() ([]Tensor, error)) ([]Tensor, error) {
	type result struct {
		outs []Tensor
		err  error
	}
	if err := ctx.Err(); err != nil { // an idle worker must not win over a ctx already done
		return nil, gaveUp(err)
	}
	ch := make(chan result, 1)
	s.jobsMu.RLock()
	if s.closed {
		s.jobsMu.RUnlock()
		return nil, ErrClosed
	}
	job := func() {
		var r result
		// A panic in the job must not unwind the worker (it would kill the process, and with it
		// every other session). The session stays usable: the panic is in Go code around the ORT
		// call — cgo cannot unwind a Go panic through C, ORT reports its own failures as error
		// statuses — and the job's deferred tensor Destroy calls run during the unwind.
		defer func() {
			if p := recover(); p != nil {
				r = result{nil, recoverJob(p)}
			}
			ch <- r
		}()
		r.outs, r.err = work()
	}
	select {
	case s.jobs <- job:
	case <-ctx.Done():
		s.jobsMu.RUnlock()
		return nil, gaveUp(ctx.Err())
	}
	s.jobsMu.RUnlock()
	// Work accepted before Close still runs: the worker drains jobs before destroying the session.
	r := <-ch
	return r.outs, r.err
}
```

[ort.go#L543-L581 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/ort.go#L543-L581)

```mermaid
sequenceDiagram
    participant R1 as request goroutine 1
    participant R2 as request goroutine 2
    participant J as jobs channel (unbuffered)
    participant W as worker goroutine (LockOSThread)
    participant O as ONNX Runtime (C++)
    R1->>J: send job 1
    J->>W: worker receives job 1
    R2->>J: send job 2, blocks while the worker is busy
    W->>O: Run on the pinned thread
    O-->>W: outputs
    W-->>R1: result on R1's own channel
    J->>W: worker receives job 2
    W->>O: Run on the same thread
    O-->>W: outputs
    W-->>R2: result on R2's own channel
```

Four things to notice:

- **Serialisation for free.** One worker runs jobs one at a time, so the session is never
  used by two goroutines at once. No mutex is needed around `s.sess`.
- **Each caller has its own reply channel** (`ch`, buffered with size 1 so the worker never
  waits for the caller to read).
- **Waiting can be cancelled.** The hand-over is a `select` between sending the job and
  `ctx.Done()`: a request whose client leaves while it queues behind other jobs gives up, and
  its job never runs. Once the worker has the job, `submit` waits for the result regardless,
  because an ORT run cannot be interrupted. (`context.Context` is explained at the end of
  this chapter.)
- **Closing is safe.** `Close` takes the write lock, sets `closed`, and closes `jobs`
  ([ort.go#L651-L667](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/ort.go#L651-L667)).
  A late `submit` sees `closed` under the read lock and returns `ErrClosed` instead of
  panicking with "send on closed channel".

When one session is not enough (MobileSAM's decoder is called ~256 times per image in
automask mode), lifecycle wraps N sessions in a `SessionPool`. The pool is a buffered
channel of free sessions, used as a semaphore: `take` receives one (or gives up when the
pool closes or `ctx` ends, again with a `select`), and the caller sends it back when done
([pool.go#L73-L122](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/pool.go#L73-L122)).

## Loading a model once (single-flight)

Eight requests for a model that is not loaded yet arrive at the same moment. Loading means
hashing hundreds of MB of weights and creating GPU sessions; doing it eight times would use
eight times the VRAM. `Manager.Load` lets the first request build and makes the rest wait
for it:

```go title="internal/lifecycle/load.go (lines 23-31, 41-91, trimmed)"
type loadCall struct {
	done chan struct{}
	// cancelled is set by Unload (or Close) while the load runs: its session must not go live.
	cancelled bool
	// err is the load's result, readable once done is closed.
	err error
	// ...
}

// ...
func (m *Manager) Load(ctx context.Context, name string) error {
	// ...
	for {
		if err := ctx.Err(); err != nil {
			return gaveUp(name, err)
		}
		m.mu.Lock()
		// ...
		if _, ok := m.live[name]; ok {
			m.mu.Unlock()
			return nil
		}
		call, busy := m.loading[name]
		if !busy {
			call = &loadCall{done: make(chan struct{})}
			// ...
			m.loading[name] = call
			m.mu.Unlock()
			go func() { _ = m.lead(name, call) }() // lead publishes its result in call.err
			if err := m.waitLoad(ctx, name, call); err != nil {
				return err
			}
			return call.err // this caller started the load: its result, success or failure
		}
		// ...
		call.waiters++
		m.mu.Unlock()
		if err := m.waitLoad(ctx, name, call); err != nil {
			// ...
			return err
		}
		// ...
	}
}
```

[load.go#L23-L31](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/load.go#L23-L31),
[#L41-L91](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/load.go#L41-L91)

The build itself runs in a goroutine of its own (`go func() { ... m.lead(name, call) }()`),
owned by no request. Every caller, including the one that started the load, then waits in
`waitLoad`:

```go title="internal/lifecycle/load.go (lines 95-102)"
func (m *Manager) waitLoad(ctx context.Context, name string, call *loadCall) error {
	select {
	case <-call.done:
		return nil
	case <-ctx.Done():
		return gaveUp(name, ctx.Err())
	}
}
```

[load.go#L95-L102 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/load.go#L95-L102)

`chan struct{}` carries no data; it exists only to be closed. When the build finishes,
`lead` closes `call.done` and every waiter wakes up at once, loops, and finds the model in
`m.live`. A waiter whose request is cancelled (`ctx.Done()`) returns early, but the load
carries on for everyone else: one client leaving, even the one that started the load,
never cancels it. Only `Unload` and `Close` do. The test `TestConcurrentLoadsBuildOnce`
starts 8 goroutines and checks the factory ran exactly once
([load_test.go#L39-L73](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/load_test.go#L39-L73)).

## Admission control, leases and the reaper

Three smaller mechanisms keep memory bounded and sessions alive while in use.

**Admission.** Before the server decodes an upload, it reserves a slot for the model.
When the model already has its maximum number of requests running or waiting, the request
fails immediately with `ErrOverloaded` (HTTP 503) instead of piling up decoded images in
memory:

```go title="internal/lifecycle/admission.go (lines 86-106)"
func (m *Manager) Admit(ctx context.Context, name string) (release func(), err error) {
	if err := ctx.Err(); err != nil {
		return func() {}, fmt.Errorf("lifecycle: %q not admitted, the request is gone: %w", name, err)
	}
	m.mu.Lock()
	limit := m.admitLimitLocked(name)
	n := m.admitted[name]
	if limit > 0 && n >= limit {
		m.mu.Unlock()
		return func() {}, fmt.Errorf("lifecycle: %w: %q already has %d requests running or waiting (limit %d per model; "+
			"set VISIONSERVE_MAX_QUEUE to change it)", ErrOverloaded, name, n, limit)
	}
	if m.admitted == nil {
		m.admitted = map[string]int{}
	}
	m.admitted[name] = n + 1
	m.mu.Unlock()

	var once sync.Once
	return func() { once.Do(func() { m.unadmit(name) }) }, nil
}
```

[admission.go#L86-L106 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/admission.go#L86-L106)

It returns a `release` **closure**: a function that remembers `name` and `once`. Wrapping
it in `sync.Once` makes calling `release()` twice harmless.

**Leases.** `acquire` increments a reference count on the live session; `release`
decrements it. Unload and the reaper only *retire* a session that is in use; the last
`release` closes it. So a request can never run on a session that was closed under it.

??? example "The lease code: acquire and release"

    ```go title="internal/lifecycle/lease.go (lines 32-53)"
    func (m *Manager) acquire(name string) (*Session, func(), error) {
    	m.mu.Lock()
    	s := m.live[name]
    	if s == nil {
    		m.mu.Unlock()
    		return nil, nil, fmt.Errorf("lifecycle: model %q was just unloaded", name)
    	}
    	s.refs++
    	m.mu.Unlock()
    	s.touch(time.Now())
    	return s, func() { m.release(s) }, nil
    }

    func (m *Manager) release(s *Session) {
    	m.mu.Lock()
    	s.refs--
    	closeNow := s.retired && s.refs == 0
    	m.mu.Unlock()
    	if closeNow {
    		_ = s.close()
    	}
    }
    ```

    [lease.go#L32-L53 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/lease.go#L32-L53)

`PredictPrompt` uses it as `s, release, err := m.loadAndAcquire(ctx, name)` (load if needed,
then `acquire`) followed by `defer release()`
([manager.go#L106-L130](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/manager.go#L106-L130)).

**Reaper.** Shown above: every 30 s it retires models idle longer than their
`idle_unload_seconds`, skipping any with `refs > 0`.

## `context.Context`: "is anyone still waiting?"

A `context.Context` travels with a request and says when it is over: the client
disconnected, a deadline passed, or the server is shutting down. `ctx.Err()` is non-nil
once that happens, and `<-ctx.Done()` is a channel closed at that moment. `net/http` gives
every request one, `r.Context()`.

Once an ONNX run has started it cannot be interrupted, so VisionServe checks the context
before inference and passes it down, so that every *wait* on the way can stop early: the
model's load, an Exclusive model's lock, the session's worker, a free copy in a pool. You
saw the pattern above: a `select` with a `case <-ctx.Done():` next to the real work. The
first check is in `server.Predict`:

```go title="internal/server/predict.go (lines 24-30)"
func Predict(ctx context.Context, p Predictor, model string, img image.Image, prompt models.Prompt) (api.Result, error) {
	fullW, fullH := img.Bounds().Dx(), img.Bounds().Dy()
	img, rect, hasROI := cropROI(img, &prompt)
	if ctx.Err() != nil {
		return api.Result{}, errClientGone
	}
	res, err := p.PredictPrompt(ctx, model, img, prompt)
```

[predict.go#L24-L30 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/server/predict.go#L24-L30)

By convention `ctx` is the first parameter of every function that can wait on behalf of a
request, which is why `PredictPrompt`, `Load`, `Run` and `RunNamed` all take one.

The server's admit helper does the same check around admission
([handlers.go#L95-L112](https://github.com/mtbui2010/vision_serve/blob/main/internal/server/handlers.go#L95-L112)).
On shutdown, `serve` waits for SIGINT/SIGTERM on a channel and gives in-flight requests
10 seconds with `context.WithTimeout`.

??? example "The shutdown code in serve"

    ```go title="internal/cli/serve.go (lines 92-110)"
    	// graceful shutdown on SIGINT/SIGTERM. ListenAndServe returns as soon as Shutdown STARTS, so
    	// wait for it to finish draining requests and releasing models before returning (main exits).
    	stopped := make(chan struct{})
    	go func() {
    		defer close(stopped)
    		sig := make(chan os.Signal, 1)
    		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
    		<-sig
    		log.Println("shutting down server...")
    		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    		defer cancel()
    		_ = srv.Shutdown(ctx)
    	}()

    	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
    		return err
    	}
    	<-stopped
    	return nil
    ```

    [serve.go#L92-L110 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/cli/serve.go#L92-L110)

## Data races and the race detector

A **data race** happens when two goroutines access the same memory at the same time and at
least one writes, without a lock or channel ordering them. The result is undefined: a map
can be corrupted, a counter can lose updates, and the bug may show once in a million
requests. Python's GIL hides many of these; Go does not.

Go ships a race detector. Add `-race` to `go test` (or `go build`) and it reports every
race that actually happens during the run, with both stack traces. It makes programs
slower (several times), so it is for tests, not production.

!!! tip "Run `-race` on anything you change in `lifecycle`, `engine`, `server` or a model with goroutines"
    ```bash
    go test -race ./internal/lifecycle/ ./internal/server/ ./internal/engine/
    ```

## Try it

1. Watch single-flight loading work, under the race detector:
   ```bash
   go test -race ./internal/lifecycle -run TestConcurrentLoadsBuildOnce -v
   ```
2. Create a race on purpose. In `internal/lifecycle/admission.go`, comment out the two lock
   lines at the start of `unadmit`:
   ```go
   func (m *Manager) unadmit(name string) {
   	// m.mu.Lock()
   	// defer m.mu.Unlock()
   ```
   Then run the contention test (64 goroutines admitting and releasing):
   ```console
   $ go test -race -run TestAdmitBoundHoldsUnderContention ./internal/lifecycle/
   ==================
   WARNING: DATA RACE
   Read at 0x00c000116000 by goroutine 12:
     runtime.mapaccess1_faststr()
     visionserve/internal/lifecycle.(*Manager).Admit()
         .../internal/lifecycle/admission.go:92 +0x1fc
   ...
   Previous write at 0x00c000116000 by goroutine 9:
     runtime.mapassign_faststr()
     visionserve/internal/lifecycle.(*Manager).unadmit()
         .../internal/lifecycle/admission.go:112 +0xb5
   ...
   ```
   `Admit` reads `m.admitted` under the lock while `unadmit` writes it without one. Without
   `-race` the test may well pass. Restore the file: `git checkout internal/lifecycle/admission.go`.
3. Read the engine's own tests for the worker: a panic inside a job must become an error and
   leave the session usable.
   ```bash
   go test ./internal/engine -run Panic -v
   ```
   `TestJobPanicBecomesErrorAndWorkerSurvives` runs everywhere. The two tests that need a
   real ONNX session are skipped (`--- SKIP`) unless `ORT_DYLIB_PATH` is set.

## Recap

- `go f()` starts a goroutine; `net/http` already runs each request in its own.
- Channels hand values between goroutines; closing one broadcasts "done"; `select` waits on
  several. `sync.WaitGroup` waits for a group; `sync.Once` runs something exactly once.
- Guard shared maps and counters with a `sync.Mutex`; hold it briefly, never across slow work.
- Each `engine.Session` owns one goroutine locked to one OS thread, because ORT's CUDA
  provider keeps per-thread GPU state; requests submit closures to it over a channel.
- `Load` is single-flight (a `done` channel), `Admit` bounds queued requests, leases stop
  sessions from closing under a request, and the reaper unloads idle models.
- `context.Context` tells the server when nobody is waiting any more.
- `go test -race` finds data races; use it whenever you touch concurrent code.
