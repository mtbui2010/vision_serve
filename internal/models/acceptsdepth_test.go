package models

import (
	"sync"
	"testing"
)

var acceptsDepthOnce sync.Once

// An architecture that registers nothing never accepts depth; a registered one decides from the
// manifest's files.
func TestAcceptsDepthOf(t *testing.T) {
	if AcceptsDepthOf("no-such-arch-acceptsdepth", map[string]string{"depth": "d.onnx"}) {
		t.Error("unregistered architecture accepts depth")
	}
	acceptsDepthOnce.Do(func() { // once per process: -count=N reruns must not register twice
		RegisterAcceptsDepth("test-acceptsdepth", func(files map[string]string) bool { return files["depth"] != "" })
	})
	if !AcceptsDepthOf("test-acceptsdepth", map[string]string{"depth": "d.onnx"}) {
		t.Error("registered architecture with a depth file does not accept depth")
	}
	if AcceptsDepthOf("test-acceptsdepth", nil) {
		t.Error("registered architecture without a depth file accepts depth")
	}
}
