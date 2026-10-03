package lifecycle

import (
	"bytes"
	"context"
	"errors"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"visionserve/internal/engine"
)

func TestPoolIntraOpThreads(t *testing.T) {
	cases := []struct {
		n, ncpu int
		env     string
		want    int
		warn    string // "" = no warning
	}{
		{4, 48, "", 3, ""},             // a quarter of 48 logical CPUs, split over 4 sessions
		{4, 32, "", 2, ""},             // 16-core workstation
		{4, 8, "", 1, ""},              // never below 1
		{4, 4, "", 1, ""},              // edge board
		{2, 48, "", 6, ""},             // smaller pool, more threads each
		{0, 8, "", 2, ""},              // a non-positive n counts as 1
		{4, 48, "5", 5, ""},            // override
		{4, 48, " 1 ", 1, ""},          // override, whitespace
		{4, 48, "0", 0, ""},            // 0 = ORT default
		{4, 48, "48", 48, ""},          // up to the CPU count is taken as is
		{4, 48, "-2", 3, "ignoring"},   // invalid → heuristic, said so
		{4, 48, "many", 3, "ignoring"}, // invalid → heuristic, said so
		{4, 48, "1.5", 3, "ignoring"},  // not an integer
		{4, 48, "4096", 48, "capped"},  // more threads than CPUs → capped, said so
	}
	for _, c := range cases {
		got, warn := poolIntraOpThreads(c.n, c.ncpu, c.env)
		if got != c.want {
			t.Errorf("poolIntraOpThreads(%d, %d, %q) = %d, want %d", c.n, c.ncpu, c.env, got, c.want)
		}
		if (c.warn == "") != (warn == "") || !strings.Contains(warn, c.warn) {
			t.Errorf("poolIntraOpThreads(%d, %d, %q) warned %q, want a warning containing %q", c.n, c.ncpu, c.env, warn, c.warn)
		}
	}
}

// A bad VISIONSERVE_POOL_THREADS is logged once per process, not on every pool it sizes.
func TestPoolThreadsEnvWarnsOnce(t *testing.T) {
	env := "bogus-" + t.Name() // a key no other test has used
	warned.Delete("VISIONSERVE_POOL_THREADS=" + env) // an earlier -count iteration logged it already
	t.Setenv("VISIONSERVE_POOL_THREADS", env)
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	prev := newEngineSession
	newEngineSession = func(_ string, _, _ []string, _ []engine.Provider, so engine.SessionOptions) (*engine.Session, error) {
		return nil, errors.New("stop")
	}
	defer func() { newEngineSession = prev }()
	for i := 0; i < 3; i++ {
		_, _ = newRunnable("m.onnx", nil, nil, 4, -1, nil)
	}
	if n := strings.Count(buf.String(), "ignoring VISIONSERVE_POOL_THREADS"); n != 1 {
		t.Errorf("warned %d times, want once:\n%s", n, buf.String())
	}
}

// A manifest's runtime.threads is capped at the machine's CPUs, with one warning per model/role.
func TestManifestThreadsCapped(t *testing.T) {
	warned.Delete("threads:m-" + t.Name() + "/head") // an earlier -count iteration logged it already
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	if got := manifestThreads("m", "head", 1, 8); got != 1 {
		t.Errorf("1 thread on 8 CPUs = %d, want 1", got)
	}
	if got := manifestThreads("m", "head", 0, 8); got != 0 {
		t.Errorf("0 (ORT default) = %d, want 0", got)
	}
	for i := 0; i < 2; i++ {
		if got := manifestThreads("m-"+t.Name(), "head", 64, 8); got != 8 {
			t.Errorf("64 threads on 8 CPUs = %d, want 8", got)
		}
	}
	if n := strings.Count(buf.String(), "capped at 8"); n != 1 {
		t.Errorf("warned %d times, want once:\n%s", n, buf.String())
	}
}

// A pool's sessions get the capped thread count; a lone session keeps ORT's default.
func TestNewRunnableThreadsOnlyForPools(t *testing.T) {
	t.Setenv("VISIONSERVE_POOL_THREADS", "5")
	stop := errors.New("captured")
	var got []engine.SessionOptions
	prev := newEngineSession
	newEngineSession = func(_ string, _, _ []string, _ []engine.Provider, so engine.SessionOptions) (*engine.Session, error) {
		got = append(got, so)
		return nil, stop
	}
	defer func() { newEngineSession = prev }()

	for _, n := range []int{1, 4} {
		if _, err := newRunnable("m.onnx", nil, nil, n, -1, nil); !errors.Is(err, stop) {
			t.Fatalf("n=%d: err = %v, want the creation error", n, err)
		}
	}
	if len(got) != 2 || got[0].IntraOpThreads != 0 || got[1].IntraOpThreads != 5 {
		t.Fatalf("sessions created with %+v, want [{0} {5}]", got)
	}
}

// runtime.threads replaces the default for every session newRunnable creates: a lone session's ORT
// default and a pool's cap alike. threads < 0 (the manifest sets nothing) changes nothing.
func TestNewRunnableManifestThreadsOverride(t *testing.T) {
	t.Setenv("VISIONSERVE_POOL_THREADS", "5")
	stop := errors.New("captured")
	var got []int
	prev := newEngineSession
	newEngineSession = func(_ string, _, _ []string, _ []engine.Provider, so engine.SessionOptions) (*engine.Session, error) {
		got = append(got, so.IntraOpThreads)
		return nil, stop
	}
	defer func() { newEngineSession = prev }()

	cases := []struct{ n, threads, want int }{
		{1, -1, 0}, // unset, lone session: ORT default
		{1, 1, 1},  // lone session pinned to one thread
		{4, -1, 5}, // unset, pool: the pool cap (env)
		{4, 2, 2},  // the manifest wins over the pool cap
		{4, 0, 0},  // 0 = ORT default, explicitly, even in a pool
	}
	for _, c := range cases {
		got = got[:0]
		if _, err := newRunnable("m.onnx", nil, nil, c.n, c.threads, nil); !errors.Is(err, stop) {
			t.Fatalf("n=%d threads=%d: err = %v, want the creation error", c.n, c.threads, err)
		}
		if len(got) != 1 || got[0] != c.want {
			t.Errorf("n=%d threads=%d: first session got %v intra-op threads, want %d", c.n, c.threads, got, c.want)
		}
	}
}

// Load hands each role its own runtime.threads value, and -1 to a role the manifest leaves alone.
func TestLoadPassesManifestThreadsPerRole(t *testing.T) {
	root := t.TempDir()
	writeTestModel(t, root, "threaded", "test-pipe", "  threads:\n    head: 1\n", "det", "head")
	m := NewManager(scanRegistry(t, root))
	t.Cleanup(m.Close)
	var mu sync.Mutex
	got := map[string]int{}
	m.openRunnable = func(path string, _, _ []string, _, threads int, _ []engine.Provider) (engine.Runnable, error) {
		mu.Lock()
		got[strings.TrimSuffix(filepath.Base(path), ".onnx")] = threads
		mu.Unlock()
		return &fakeEngine{}, nil
	}
	if err := m.Load(context.Background(), "threaded"); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got["head"] != 1 || got["det"] != -1 {
		t.Errorf("threads per role = %v, want map[det:-1 head:1]", got)
	}
}
