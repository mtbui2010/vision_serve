package engine

import "testing"

// equalProviders reports whether two provider chains are identical in order.
func equalProviders(a, b []Provider) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestResolveProviders(t *testing.T) {
	cases := []struct {
		name   string
		prefer []string
		want   []Provider
	}{
		{
			name:   "empty defaults to cpu only",
			prefer: nil,
			want:   []Provider{ProviderCPU},
		},
		{
			name:   "edge chain tensorrt cuda cpu",
			prefer: []string{"tensorrt", "cuda", "cpu"},
			want:   []Provider{ProviderTensorRT, ProviderCUDA, ProviderCPU},
		},
		{
			name:   "cpu always appended last when missing",
			prefer: []string{"cuda"},
			want:   []Provider{ProviderCUDA, ProviderCPU},
		},
		{
			name:   "coreml then cpu",
			prefer: []string{"coreml"},
			want:   []Provider{ProviderCoreML, ProviderCPU},
		},
		{
			name:   "directml then cpu",
			prefer: []string{"directml"},
			want:   []Provider{ProviderDirectML, ProviderCPU},
		},
		{
			name:   "openvino then cpu",
			prefer: []string{"openvino"},
			want:   []Provider{ProviderOpenVINO, ProviderCPU},
		},
		{
			name:   "normalizes case and whitespace",
			prefer: []string{" CoreML ", "CUDA"},
			want:   []Provider{ProviderCoreML, ProviderCUDA, ProviderCPU},
		},
		{
			name:   "dedupes repeated providers preserving first order",
			prefer: []string{"cuda", "cuda", "cpu", "cpu"},
			want:   []Provider{ProviderCUDA, ProviderCPU},
		},
		{
			name:   "skips empty entries",
			prefer: []string{"", "openvino", "  "},
			want:   []Provider{ProviderOpenVINO, ProviderCPU},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveProviders(tc.prefer)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !equalProviders(got, tc.want) {
				t.Errorf("ResolveProviders(%v) = %v, want %v", tc.prefer, got, tc.want)
			}
		})
	}
}

func TestResolveProvidersInvalid(t *testing.T) {
	for _, bad := range []string{"rocm", "vulkan", "gpu", "metal"} {
		if _, err := ResolveProviders([]string{bad}); err == nil {
			t.Errorf("ResolveProviders([%q]) expected error, got nil", bad)
		}
	}
}

// TestResolveProvidersCPUAlwaysLast guards the invariant that CPU is the final
// fallback even when explicitly placed earlier in the chain.
func TestResolveProvidersCPUAlwaysLast(t *testing.T) {
	got, err := ResolveProviders([]string{"cpu", "cuda"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// "cpu" first means it is consumed first; it is NOT re-appended (already seen).
	want := []Provider{ProviderCPU, ProviderCUDA}
	if !equalProviders(got, want) {
		t.Errorf("ResolveProviders([cpu cuda]) = %v, want %v", got, want)
	}
}
