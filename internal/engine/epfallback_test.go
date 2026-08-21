package engine

import "testing"

// A session can be created successfully while ORT quietly abandons the execution provider we
// registered. runErr is nil, so the only evidence is what ORT printed. This detector is what
// turns that into a warning and an honest `device` field instead of a plausible wrong number:
// a silent CPU fallback invalidated a whole measurement sweep in this project, and was noticed
// only because the process held no VRAM.
func TestEPWasDropped(t *testing.T) {
	cases := []struct {
		name string
		out  string
		want bool
	}{
		{"nothing printed — a healthy GPU session", "", false},
		{
			"provider .so missing a dependency",
			"[E:onnxruntime] Failed to load library libonnxruntime_providers_cuda.so with error: " +
				"libcudnn.so.9: cannot open shared object file: No such file or directory",
			true,
		},
		{
			"explicit fallback notice",
			"Some nodes were not assigned to the preferred execution providers. " +
				"Falling back to CPUExecutionProvider.",
			true,
		},
		{"named CUDA failure", "Failed to create CUDAExecutionProvider", true},
		{"named TensorRT failure", "Failed to create TensorRTExecutionProvider.", true},
		{
			// Matching is case-insensitive because ORT is not consistent about it.
			"case does not matter",
			"FAILED TO LOAD LIBRARY libonnxruntime_providers_tensorrt.so",
			true,
		},
		{
			// A working session prints ordinary chatter; a false alarm here would send people
			// hunting a GPU problem that does not exist.
			"benign ORT chatter is not a fallback",
			"[I:onnxruntime] Serializing optimized model. TensorRT engine cache hit for node_1.",
			false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := epWasDropped(c.out); got != c.want {
				t.Errorf("epWasDropped(%q) = %v, want %v", c.out, got, c.want)
			}
		})
	}
}
