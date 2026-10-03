package lifecycle

import (
	"errors"
	"testing"

	"visionserve/internal/engine"
)

func TestPoolIntraOpThreads(t *testing.T) {
	cases := []struct {
		n, ncpu int
		env     string
		want    int
	}{
		{4, 48, "", 3},     // a quarter of 48 logical CPUs, split over 4 sessions
		{4, 32, "", 2},     // 16-core workstation
		{4, 8, "", 1},      // never below 1
		{4, 4, "", 1},      // edge board
		{2, 48, "", 6},     // smaller pool, more threads each
		{0, 8, "", 2},      // a non-positive n counts as 1
		{4, 48, "5", 5},    // override
		{4, 48, " 1 ", 1},  // override, whitespace
		{4, 48, "0", 0},    // 0 = ORT default
		{4, 48, "-2", 3},   // invalid → heuristic
		{4, 48, "many", 3}, // invalid → heuristic
	}
	for _, c := range cases {
		if got := poolIntraOpThreads(c.n, c.ncpu, c.env); got != c.want {
			t.Errorf("poolIntraOpThreads(%d, %d, %q) = %d, want %d", c.n, c.ncpu, c.env, got, c.want)
		}
	}
}

// A pool's sessions get the capped thread count; a lone session keeps ORT's default.
func TestNewRunnableThreadsOnlyForPools(t *testing.T) {
	t.Setenv("VS_POOL_THREADS", "5")
	stop := errors.New("captured")
	var got []engine.SessionOptions
	prev := newEngineSession
	newEngineSession = func(_ string, _, _ []string, _ []engine.Provider, so engine.SessionOptions) (*engine.Session, error) {
		got = append(got, so)
		return nil, stop
	}
	defer func() { newEngineSession = prev }()

	for _, n := range []int{1, 4} {
		if _, err := newRunnable("m.onnx", nil, nil, n, nil); !errors.Is(err, stop) {
			t.Fatalf("n=%d: err = %v, want the creation error", n, err)
		}
	}
	if len(got) != 2 || got[0].IntraOpThreads != 0 || got[1].IntraOpThreads != 5 {
		t.Fatalf("sessions created with %+v, want [{0} {5}]", got)
	}
}
