package lifecycle

import (
	"context"
	"errors"
	"testing"

	"visionserve/internal/engine"
)

// intoEngine is a fakeEngine that can write into caller buffers, recording what it was given.
type intoEngine struct {
	fakeEngine
	got    map[string]engine.Tensor
	gotCtx context.Context
}

func (e *intoEngine) RunNamedInto(ctx context.Context, _, into map[string]engine.Tensor) ([]engine.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	e.got, e.gotCtx = into, ctx
	return []engine.Tensor{into["y"]}, nil
}

// ctxEngine is a fakeEngine whose RunNamed reports a done ctx, like a real session's wait.
type ctxEngine struct{ fakeEngine }

func (e *ctxEngine) RunNamed(ctx context.Context, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return e.fakeEngine.RunNamed(ctx, in)
}

// runner.RunInto hands the buffers and the request ctx to a session that can take them, runs a
// session that cannot as Run does (same ctx), and fails an unknown role like Run.
func TestRunnerRunInto(t *testing.T) {
	ie, plain := &intoEngine{}, &ctxEngine{}
	ctx := context.WithValue(context.Background(), struct{}{}, "request")
	r := runner{ctx: ctx, engines: map[string]engine.Runnable{"into": ie, "plain": plain}}
	buf := engine.F32(make([]float32, 3), 1, 3)

	outs, err := r.RunInto("into", nil, map[string]engine.Tensor{"y": buf})
	if err != nil || ie.got == nil || len(outs) != 1 || &outs[0].Data[0] != &buf.Data[0] {
		t.Fatalf("into role: outs %v err %v, engine got %v", outs, err, ie.got)
	}
	if ie.gotCtx != ctx {
		t.Fatal("into role: the session did not get the request ctx")
	}
	if _, err := r.RunInto("plain", nil, map[string]engine.Tensor{"y": buf}); err != nil {
		t.Fatalf("plain role: %v", err)
	}
	plain.closed.Store(true) // the fallback really is RunNamed: a closed session reports it
	if _, err := r.RunInto("plain", nil, map[string]engine.Tensor{"y": buf}); err != engine.ErrClosed {
		t.Fatalf("plain role after close: err %v, want ErrClosed from RunNamed", err)
	}
	if _, err := r.RunInto("missing", nil, nil); err == nil {
		t.Fatal("unknown role accepted")
	}

	// A request whose client left: both paths see the done ctx and run nothing.
	done, cancel := context.WithCancel(context.Background())
	cancel()
	r.ctx = done
	ie.got = nil
	if _, err := r.RunInto("into", nil, map[string]engine.Tensor{"y": buf}); !errors.Is(err, context.Canceled) || ie.got != nil {
		t.Fatalf("into role, done ctx: err %v, ran %v", err, ie.got != nil)
	}
	plain.closed.Store(false)
	if _, err := r.RunInto("plain", nil, map[string]engine.Tensor{"y": buf}); !errors.Is(err, context.Canceled) {
		t.Fatalf("plain role, done ctx: err %v, want context.Canceled", err)
	}
}
