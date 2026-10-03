package registry

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeThreadsManifest(t *testing.T, files, runtimeExtra string) (*Manifest, error) {
	t.Helper()
	dir := t.TempDir()
	y := "name: m\ntask: open_vocab\nlicense: MIT\n" + files +
		"input:\n  width: 8\n  height: 8\nruntime:\n  prefer: [cpu]\n" + runtimeExtra
	path := filepath.Join(dir, "manifest.yaml")
	if err := os.WriteFile(path, []byte(y), 0o644); err != nil {
		t.Fatal(err)
	}
	return LoadManifest(path)
}

const twoRoles = "files:\n  rfdetr: d.onnx\n  head: h.onnx\n"

func TestRuntimeThreadsParsed(t *testing.T) {
	m, err := writeThreadsManifest(t, twoRoles, "  threads:\n    head: 1\n    rfdetr: 0\n")
	if err != nil {
		t.Fatal(err)
	}
	if n, ok := m.IntraOpThreads("head"); !ok || n != 1 {
		t.Errorf("head = %d, %v; want 1, true", n, ok)
	}
	if n, ok := m.IntraOpThreads("rfdetr"); !ok || n != 0 {
		t.Errorf("rfdetr = %d, %v; want 0, true (explicit ORT default)", n, ok)
	}
}

// No runtime.threads: nothing is set for any role, so lifecycle keeps its defaults.
func TestRuntimeThreadsAbsentSetsNothing(t *testing.T) {
	m, err := writeThreadsManifest(t, twoRoles, "")
	if err != nil {
		t.Fatal(err)
	}
	for _, role := range []string{"rfdetr", "head", "text"} {
		if _, ok := m.IntraOpThreads(role); ok {
			t.Errorf("%s: a value is set although the manifest declares none", role)
		}
	}
}

func TestRuntimeThreadsRejected(t *testing.T) {
	cases := []struct{ name, files, extra, want string }{
		{"unknown role", twoRoles, "  threads:\n    haed: 1\n", `"haed" is not a role`},
		{"negative", twoRoles, "  threads:\n    head: -1\n", "must be >= 0"},
		{"not an integer", twoRoles, "  threads:\n    head: one\n", "parse YAML"},
		{"fractional", twoRoles, "  threads:\n    head: 1.5\n", "parse YAML"},
		{"no files map", "model_file: x.onnx\n", "  threads:\n    model: 1\n", "no 'files' map"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := writeThreadsManifest(t, c.files, c.extra)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("err = %v, want one containing %q", err, c.want)
			}
		})
	}
}
