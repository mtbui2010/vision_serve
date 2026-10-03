package mobilesam

import (
	"context"
	"errors"
	"fmt"
	"image"
	"sync"
	"sync/atomic"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// failingInto is a models.IntoRunner over a fake decoder whose failAt-th full-resolution call
// (and every later one) fails with context.Canceled — what lifecycle's runner returns once the
// request's client has left. It counts the decoder calls in flight.
type failingInto struct {
	fake     *fakeDecoder
	failAt   int
	mu       *sync.Mutex
	calls    *int
	inFlight *atomic.Int32
}

func (r failingInto) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	return fakeRunner{r.fake}.Run(role, in)
}

func (r failingInto) RunInto(role string, in, into map[string]engine.Tensor) ([]engine.Tensor, error) {
	r.inFlight.Add(1)
	defer r.inFlight.Add(-1)
	r.mu.Lock()
	*r.calls++
	n := *r.calls
	r.mu.Unlock()
	if r.failAt > 0 && n >= r.failAt {
		return nil, fmt.Errorf("decoder wait: %w", context.Canceled)
	}
	return r.fake.runInto(in, into)
}

func (r failingInto) InputNames(role string) []string  { return fakeRunner{r.fake}.InputNames(role) }
func (r failingInto) OutputNames(role string) []string { return fakeRunner{r.fake}.OutputNames(role) }

// Every final-pass logit buffer is freed exactly once, and only after the last decoder call has
// returned, whether the pass succeeds or decoder calls fail (a cancelled request).
func TestFinalPassFreesScratchOnEveryPath(t *testing.T) {
	const W, H = 1200, 800
	var rects [][4]int
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			rects = append(rects, [4]int{40 + 300*i, 40 + 260*j, 150, 120})
		}
	}
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	sam := &mobileSAM{gridSize: 16}
	prev := allocScratch
	defer func() { allocScratch = prev }()

	for _, failAt := range []int{0, 1, 5} { // 0: never fails
		var inFlight atomic.Int32
		var mu sync.Mutex
		var allocs, frees, calls int
		var freedInFlight atomic.Bool
		allocScratch = func(n int) ([]float32, func()) {
			buf, free := prev(n)
			mu.Lock()
			allocs++
			mu.Unlock()
			return buf, func() {
				if inFlight.Load() != 0 {
					freedInFlight.Store(true)
				}
				mu.Lock()
				frees++
				mu.Unlock()
				free()
			}
		}
		r := failingInto{fake: newFake(W, H, 1, rects), failAt: failAt, mu: &mu, calls: &calls, inFlight: &inFlight}
		_, err := sam.Infer(img, models.Prompt{}, r)

		switch {
		case failAt == 0 && err != nil:
			t.Fatalf("failAt 0: %v", err)
		case failAt > 0 && !errors.Is(err, context.Canceled):
			t.Fatalf("failAt %d: err = %v, want context.Canceled", failAt, err)
		}
		if allocs == 0 || allocs != frees || allocs > autoFinalWorkers {
			t.Fatalf("failAt %d: %d buffers allocated, %d freed (want equal, 1..%d)", failAt, allocs, frees, autoFinalWorkers)
		}
		if freedInFlight.Load() {
			t.Fatalf("failAt %d: a buffer was freed while a decoder call was in flight", failAt)
		}
	}
}
