package registry

import (
	"strings"
	"testing"
)

// runtime.max_useful_side: absent = derive, 0 = never resize, N = longer side N. It is a known key
// (no unknown-key warning) and refuses what it cannot mean: negative, fractional, not a number.
func TestRuntimeMaxUsefulSide(t *testing.T) {
	m, err := writeThreadsManifest(t, twoRoles, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := m.UsefulSideOverride(); ok {
		t.Error("absent key: an override is set")
	}
	for _, c := range []struct {
		yaml string
		want int
	}{{"0", 0}, {"1600", 1600}} {
		m, err := writeThreadsManifest(t, twoRoles, "  max_useful_side: "+c.yaml+"\n")
		if err != nil {
			t.Fatalf("%s: %v", c.yaml, err)
		}
		if n, ok := m.UsefulSideOverride(); !ok || n != c.want {
			t.Errorf("max_useful_side: %s -> %d, %v; want %d, true", c.yaml, n, ok, c.want)
		}
		if keys := m.UnknownKeys(); len(keys) != 0 {
			t.Errorf("max_useful_side: %s reported as unknown: %v", c.yaml, keys)
		}
	}
	for _, c := range []struct{ yaml, want string }{
		{"-1", "must be >= 0"},
		{"1.5", "parse YAML"},
		{"big", "parse YAML"},
	} {
		_, err := writeThreadsManifest(t, twoRoles, "  max_useful_side: "+c.yaml+"\n")
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("max_useful_side: %s: err = %v, want it to contain %q", c.yaml, err, c.want)
		}
	}
}
