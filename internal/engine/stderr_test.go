package engine

import "testing"

func TestIsORTLogLine(t *testing.T) {
	for _, c := range []struct {
		line string
		want bool
	}{
		{"2026-10-03 09:15:02.123456789 [E:onnxruntime:Default, provider_bridge_ort.cc:1848 TryGetProviderInfo_CUDA] Failed to load library", true},
		{"2026-10-03 09:15:02.1 [W:onnxruntime:, session_state.cc:1162 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned", true},
		{"[W:onnxruntime:Default, tensorrt_execution_provider.h:86 log] [2026-10-03 09:15:02 WARNING] [TRT] ...", true},
		{"2026/10/03 09:15:02 preloaded: rf-detr", false},                                 // Go's log package
		{"engine: [trace] creating session for model.onnx — EP chain: cuda → cpu", false}, // our own trace
		{"panic: runtime error: index out of range", false},
		{"", false},
	} {
		if got := isORTLogLine(c.line); got != c.want {
			t.Errorf("isORTLogLine(%q) = %v, want %v", c.line, got, c.want)
		}
	}
}
