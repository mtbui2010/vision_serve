# Lifecycle manager

A model on disk is just files: an `.onnx` graph, a `manifest.yaml`, maybe a label list. Before it
can answer a request it has to be turned into one or more ONNX Runtime **sessions** (a session is
a graph loaded into memory, on the CPU or in GPU memory, ready to run). Sessions are heavy: a
GroundingDINO session holds hundreds of megabytes and can take seconds to build. The lifecycle
manager (`internal/lifecycle`) is the one place that creates, shares and frees them. It loads a
model the first time someone asks for it, lets many requests use it at once, refuses work when a
model is already swamped, and unloads it again after it has sat idle. Everything else in
VisionServe (the HTTP server, the `run` CLI) goes through it, so the rules that keep memory and
GPU usage under control live in exactly one package.

## The picture

```mermaid
sequenceDiagram
    participant H as HTTP handler
    participant M as lifecycle.Manager
    participant R as registry
    participant E as engine
    participant S as Session

    H->>M: Admit(ctx, model)
    M-->>H: release func, or ErrOverloaded
    H->>H: decode image
    H->>M: PredictPrompt(model, img, prompt)
    alt not loaded yet
        M->>R: manifest, verify sha256, labels
        M->>E: open session or pool per role
        E-->>M: engine.Runnable
        M->>M: store as live Session
    end
    M->>S: acquire lease, refs++
    S->>S: Predict or pipeline Infer
    S-->>M: api.Result
    M->>S: release lease, refs--
    M-->>H: api.Result
    H->>M: release admission slot
    Note over M: reaper every 30 s closes models idle past their timeout
```

## Key ideas

### One Manager, one live Session per model

`Manager` keeps a map from model name to a live `Session`. A `Session` bundles the model's Go code
(pre/postprocessing, see [Models and manifests](models.md)) with the ONNX sessions it needs. A
plain model has one engine session; a pipeline model (MobileSAM, GroundingDINO, Grounded-SAM, ...)
has one per **role**, such as `encoder` and `decoder`.

```go title="internal/lifecycle/manager.go"
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

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/manager.go#L30-L59)

The request entry point is `PredictPrompt`: load if needed, take a lease, run, give the lease back.

```go title="internal/lifecycle/manager.go"
func (m *Manager) PredictPrompt(name string, img image.Image, prompt models.Prompt) (api.Result, error) {
	if err := m.Load(name); err != nil {
		return api.Result{}, err
	}
	s, release, err := m.acquire(name)
	if err != nil {
		return api.Result{}, err
	}
	defer release()
	if err := m.resolveTemplates(&prompt); err != nil {
		return api.Result{}, err
	}
	return s.Predict(img, prompt, time.Now())
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/manager.go#L100-L113)

### Lazy loading, one load at a time

Nothing is loaded at startup unless you pass `visionserve serve --preload a,b`. The first request
for a model builds it. If ten requests for a cold model arrive together, only the first one (the
"leader") builds; the other nine wait for it and then reuse its result. This pattern is often
called *singleflight*. Without it, every request would build its own copy, briefly using ten times
the GPU memory, and nine copies would be thrown away.

```go title="internal/lifecycle/load.go"
	for {
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
			return m.lead(name, call)
		}
		// ...
		call.waiters++
		m.mu.Unlock()
		<-call.done
		// ...
	}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/load.go#L38-L71)

Building a model (`buildModel`) runs the load-time checks in order: the name is in the registry
(rescanning the models directory at most once per second, so a model pulled while the server runs
is found), the weights exist, their SHA-256 matches the manifest when it pins one, the labels
load, the execution-provider chain is valid, and the `preprocess:` block resolves. Only then are
ONNX sessions opened. If anything fails half way, the sessions already opened are closed again, so
a failed load never leaves GPU memory behind. A panic during the build is turned into an error,
because a stuck "loading" entry would make every later request for that model hang.

### Leases: never close a session under a running request

A model can be unloaded three ways: `POST /api/unload` (or `visionserve rm`), the idle reaper,
and server shutdown. Any of them can happen while a request is still running on the model. To make
that safe, every request holds a **lease**: a reference count on the `Session`.

```go title="internal/lifecycle/lease.go"
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

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/lease.go#L32-L53)

Unloading only *retires* a session: it disappears from the live map at once (new requests load a
fresh copy), but the actual close happens when the last lease is released. Before leases existed,
an unload during a request could crash the process with a "send on closed channel" panic.

An unload that arrives while the model is still *loading* marks the load cancelled; the loader
then closes what it built instead of publishing it.

### Idle unload

The reaper wakes every 30 seconds and retires every model that has no running request and has been
idle longer than its timeout.

```go title="internal/lifecycle/reaper.go"
func (m *Manager) expireIdle(now time.Time) []*Session {
	var expired []*Session
	m.mu.Lock()
	defer m.mu.Unlock()
	for name, s := range m.live {
		if s.refs == 0 && s.idleTimeout > 0 && s.idleFor(now) > s.idleTimeout {
			if r := m.retireLocked(name); r != nil {
				expired = append(expired, r)
			}
		}
	}
	return expired
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/reaper.go#L28-L40)

The timeout comes from the manifest (`runtime.idle_unload_seconds`, 300 in almost every shipped manifest;
`0` means never). `visionserve serve --idle-unload-seconds N` overrides it for every model: `-1`
(the default) keeps each manifest's value, `0` keeps all models resident, and `N` sets them all to
N seconds. Sessions are closed outside the manager's lock, because destroying a GPU session takes
time and that lock gates every request.

### Session pools and `PoolSizes`

One engine session runs one inference at a time (see [Engine](engine.md)). For most models that is
fine. MobileSAM's automatic mask generator, however, calls its small decoder a few hundred times
per image, and those calls are independent. A pipeline model can therefore ask for several
identical copies of a role by implementing `models.PoolSizer`; MobileSAM, Grounded-SAM, the hybrid
router with SAM, grasp and background all ask for 4 decoder copies. Lifecycle wraps them in an
`engine.SessionPool`, which hands each call to whichever copy is free.

```go title="internal/lifecycle/load.go"
func newRunnable(path string, inputNames, outputNames []string, n, threads int, providers []engine.Provider) (engine.Runnable, error) {
	if n <= 1 {
		// ...
		s, err := newEngineSession(path, inputNames, outputNames, providers, so)
		if err != nil {
			return nil, err
		}
		return s, nil
	}
	// ...
	sessions := make([]*engine.Session, 0, n)
	for i := 0; i < n; i++ {
		s, err := newEngineSession(path, inputNames, outputNames, providers, so)
		if err != nil {
			for _, c := range sessions {
				_ = c.Close()
			}
			return nil, err
		}
		sessions = append(sessions, s)
	}
	return engine.NewSessionPool(sessions), nil
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/load.go#L227-L262)

`VS_POOL_OVERRIDE=n` forces every role (and plain models too) to a pool of `n`. It exists for
benchmark sweeps, not for normal use.

### The `Runner`: how pipeline models reach their sessions

A pipeline model never holds an ONNX session itself. Lifecycle passes it a `Runner`, a small
gateway that runs a session by role name and reports its input and output names. The model decides
the order (encoder, then decoder; detector, then segmenter), but it cannot create, keep or close a
session.

```go title="internal/lifecycle/session.go"
type runner struct {
	engines map[string]engine.Runnable
}

func (r runner) Run(role string, inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
	s, ok := r.engines[role]
	if !ok {
		return nil, fmt.Errorf("lifecycle: no ONNX session for role %q", role)
	}
	return s.RunNamed(inputs)
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/session.go#L227-L237)

### Exclusive models

Some pipelines must not run two requests at once on the same loaded model. GroundingDINO,
Grounded-SAM and the GroundingDINO variant of grasp implement `models.Exclusive`; lifecycle then
holds a per-model lock around the whole `Infer` call. The lock belongs to the loaded model, so two
*different* models still run side by side. (This replaced an older process-wide mutex inside the
GroundingDINO package that made unrelated models wait for each other.)

```go title="internal/lifecycle/session.go"
func (s *Session) inferPipeline(img image.Image, prompt models.Prompt) (api.Result, error) {
	if s.exclusive {
		s.inferMu.Lock()
		defer s.inferMu.Unlock()
	}
	return s.pipeline.Infer(img, prompt, runner{s.engines})
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/session.go#L116-L122)

### Admission control

When a busy model has a queue of requests waiting, each waiting request keeps its decoded image in
memory (up to about 160 MB for a 40-megapixel upload). Admission control caps how many requests a
model may hold, running plus waiting. The server calls `Admit` *before* it decodes the image, and a
refused request fails immediately with `ErrOverloaded`, which the server turns into HTTP 503 with a
`Retry-After` header (see [HTTP server](server.md)). A request whose client has already
disconnected is not admitted either.

```go title="internal/lifecycle/admission.go"
func (m *Manager) admitLimitLocked(name string) int {
	switch {
	case m.maxQueue > 0:
		return m.maxQueue
	case m.maxQueue == queueUnbounded:
		return 0
	}
	slots := 1
	if s := m.live[name]; s != nil {
		slots = s.slots()
	}
	return max(defaultMinQueue, 2*slots)
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/admission.go#L119-L131)

The automatic bound is `max(32, 2 × slots)`, where *slots* is the model's largest session pool (1
for a single session, and 1 for an Exclusive model whatever its pools). `VISIONSERVE_MAX_QUEUE=n`
sets a fixed bound of `n` for every model; `VISIONSERVE_MAX_QUEUE=0` turns admission control off.
Anything else is ignored with a warning. The floor of 32 is deliberately generous: an earlier bound
of 4 refused an ordinary client that sent 16 requests in parallel.

### Intra-op threads

ONNX Runtime gives every CPU session its own pool of worker threads, one per physical core by
default, and idle threads in that pool keep spinning for a while. One session per model is fine.
A pool of 4 decoder copies running at once, each with a full set of spinning threads, puts four
times the core count of busy threads on the machine. Measured on a 48-thread server, MobileSAM's
automask went from 17–22 s to 3.7–5.7 s per image when each pooled session got fewer threads, with
identical output.

```go title="internal/lifecycle/load.go"
func poolIntraOpThreads(n, ncpu int, env string) (threads int, warn string) {
	heuristic := max(1, ncpu/(4*max(n, 1)))
	v := strings.TrimSpace(env)
	if v == "" {
		return heuristic, ""
	}
	k, err := strconv.Atoi(v)
	switch {
	case err != nil || k < 0:
		return heuristic, fmt.Sprintf("lifecycle: ignoring VISIONSERVE_POOL_THREADS=%q (want an integer >= 0; "+
			"0 = ONNX Runtime's default) — pooled sessions get NumCPU/(4n) intra-op threads", env)
	case k > ncpu:
		return ncpu, fmt.Sprintf("lifecycle: VISIONSERVE_POOL_THREADS=%d is more than the %d logical CPUs — "+
			"capped at %d intra-op threads per pooled session", k, ncpu, ncpu)
	}
	return k, ""
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/load.go#L284-L300)

The rules, in order of precedence:

1. The manifest's `runtime.threads` map (role → thread count, e.g. `threads: {head: 1}`) sets the
   count for that role's sessions. It is for pipeline roles only (it needs a `files:` map) and is
   capped at the host's logical CPU count. `0` means ONNX Runtime's default. Use it for a tiny
   session that runs between a big one's calls: a 1 ms score head with its default spinning pool
   slowed a whole CPU request about 3×.
2. Otherwise each session of an n-session pool gets `NumCPU / (4n)` threads, at least 1.
   `VISIONSERVE_POOL_THREADS=k` replaces that with `k` (at most `NumCPU`); `0` restores ORT's
   default.
3. A lone session keeps ORT's default.

### Weight verification without re-hashing

When a manifest pins `sha256:` digests, every load hashes the weights and refuses a mismatch,
because the declared license is bound to the audited bytes (see
[Catalog](catalog.md)). Hashing a 695 MB file takes 3–4 s, and a model is reloaded after every idle
unload, so the registry keeps a process-wide digest cache. An entry is reused only while the file's
size, modification time, change time, device and inode are all unchanged; a file modified within
2 seconds of being hashed is not cached, since file timestamps are too coarse to tell two quick
rewrites apart.

```go title="internal/registry/verify.go"
func (c *digestCache) sha256(path string) (string, error) {
	// ...
	before, err := statFileID(abs)
	// ...
	c.mu.Lock()
	e, ok := c.entries[abs]
	c.mu.Unlock()
	if ok && e.id == before {
		return e.digest, nil
	}
	// ...
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/verify.go#L478-L510)

### Preprocess without inference

`Manager.Preprocess` backs `POST /api/preprocess`. It answers "what tensor would this model
receive for this image?" without loading any session, so you can compare it against the
preprocessing a model was trained with. A mismatch there is silent and expensive: letterboxing
instead of stretching cost RF-DETR 7.35 mAP.

It must report what serving really does, not a second implementation that could drift. A plain
model's own `Preprocess` is called. A pipeline model's own `Infer` is run against a *recording*
`Runner` that captures the inputs of the first session call and then stops the pipeline:

```go title="internal/lifecycle/preprocess.go"
func (r *recordingRunner) Run(role string, inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
	if r.inputs == nil {
		r.role, r.inputs = role, inputs
	}
	return nil, errCaptured
}
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/preprocess.go#L104-L109)

### Explain (heatmaps)

`Manager.Explain` backs `POST /api/explain`. Models whose manifest has an `explain:` block get a
second session of the same ONNX file that also exposes internal feature tensors. It is created
lazily on the first explain call, attached to the leased `Session`, and closed with it. The
detection session itself is opened with those extra outputs filtered out, so ordinary predictions
never pay for them.

### Typed errors

Lifecycle does not know about HTTP. It wraps three sentinel errors, and the server maps them to
status codes with `errors.Is`:

```go title="internal/lifecycle/errors.go"
var (
	// ErrModelNotFound: the name is not in the registry, or its weights are missing.
	ErrModelNotFound = errors.New("model not found")
	// ErrInvalidRequest: the request itself is wrong (prompt, template, option values).
	ErrInvalidRequest = errors.New("invalid request")
	// ErrOverloaded: admission control refused the request (too many waiting for this model).
	ErrOverloaded = errors.New("model overloaded")
)
```

[View on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/errors.go#L8-L15)

## Where in the code

!!! code "Where in the code"
    | File | Responsibility |
    |---|---|
    | `internal/lifecycle/manager.go` | `Manager`, `NewManager`, request entry points (`PredictPrompt`, `InferTensor`), `Close` |
    | `internal/lifecycle/load.go` | `Load` (singleflight), `buildModel` load-time checks, `newRunnable` (session or pool), thread sizing |
    | `internal/lifecycle/lease.go` | `acquire`/`release` leases, `Unload`, retiring a session |
    | `internal/lifecycle/reaper.go` | idle auto-unload loop (every 30 s) |
    | `internal/lifecycle/admission.go` | `Admit`, per-model bound, `VISIONSERVE_MAX_QUEUE` |
    | `internal/lifecycle/session.go` | one loaded model: simple or pipeline mode, `Runner`, Exclusive lock, device string |
    | `internal/lifecycle/preprocess.go` | `/api/preprocess` path and the recording `Runner` |
    | `internal/lifecycle/explain.go` | Score-CAM / attention heatmaps, lazy explain session |
    | `internal/lifecycle/errors.go` | typed errors the server maps to HTTP status |
    | `internal/registry/verify.go` | `VerifyWeights` and the SHA-256 digest cache |
    | `internal/cli/serve.go` | `--preload` and `--idle-unload-seconds` flags |

## Things to know

!!! warning "Sessions are created only here"
    An ONNX session must never be created in a handler or inside a model package. Lifecycle owns
    every session, so it can account for GPU memory, close everything on unload, and keep a
    request from losing its session mid-flight. A pipeline model only orchestrates calls through
    the `Runner`.

!!! note "The first request after an idle pause is slow"
    After `idle_unload_seconds` the model is closed, and the next request pays the full load again
    (session build, plus a weights hash if the file changed). If that matters, run
    `visionserve serve --idle-unload-seconds 0` or preload the model.

!!! note "Admission happens before decoding, not before loading"
    `Admit` only counts requests. It does not check that the model exists: an unknown name is
    admitted and then fails with 404 when `Load` cannot find it. The bookkeeping for that name is
    dropped with its last release, so made-up names do not accumulate.

!!! tip "Thread counts do not change results"
    `VISIONSERVE_POOL_THREADS` and `runtime.threads` only change speed and CPU usage. If you
    compare timings against older numbers measured before the pool cap existed, set
    `VISIONSERVE_POOL_THREADS=0`. The `NumCPU/(4n)` rule was measured on a 48-thread machine; on a
    small edge CPU (say 8 threads with 4 decoder copies) each copy gets a single thread, which has
    not been measured yet.

!!! note "The manifest is snapshotted at load"
    A loaded `Session` keeps the manifest it was built from. If you edit `manifest.yaml` while the
    model is loaded, explain and other per-session logic keep using the old values until the model
    is unloaded and loaded again.

!!! warning "A waiting request does not follow its context"
    `Admit` rejects a request whose client has already left, but once admitted, a request waiting
    for a free session in a pool does not watch its context. This is a known open item.
