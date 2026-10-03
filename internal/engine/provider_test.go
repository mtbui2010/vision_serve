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
	clearEPEnv(t)
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
	clearEPEnv(t)
	for _, bad := range []string{"rocm", "vulkan", "gpu", "metal"} {
		if _, err := ResolveProviders([]string{bad}); err == nil {
			t.Errorf("ResolveProviders([%q]) expected error, got nil", bad)
		}
	}
}

// TestResolveProvidersCPUAlwaysLast guards the invariant that CPU is the final
// fallback even when explicitly placed earlier in the chain.
func TestResolveProvidersCPUAlwaysLast(t *testing.T) {
	clearEPEnv(t)
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

// clearEPEnv runs a test with neither EP switch set and the --tensorrt flag cleared, whatever
// the environment the tests were started in.
func clearEPEnv(t *testing.T) {
	t.Helper()
	t.Setenv(epOverrideEnv, "")
	t.Setenv(tensorRTEnv, "")
	SetTensorRT(false)
	t.Cleanup(func() { SetTensorRT(false) })
}

// TestResolveProvidersTensorRTOptIn: the default NVIDIA chain is CUDA → CPU; the opt-in (flag or
// env) puts tensorrt right before cuda, never into a chain without cuda, never twice, and never
// on top of VISIONSERVE_EP, which replaces the chain outright.
func TestResolveProvidersTensorRTOptIn(t *testing.T) {
	trt := []Provider{ProviderTensorRT, ProviderCUDA, ProviderCPU}
	def := []Provider{ProviderCUDA, ProviderCPU}
	cases := []struct {
		name   string
		prefer []string
		flag   bool   // --tensorrt
		env    string // VISIONSERVE_TENSORRT
		ep     string // VISIONSERVE_EP
		want   []Provider
	}{
		{name: "default is cuda then cpu", prefer: []string{"cuda", "cpu"}, want: def},
		{name: "flag inserts tensorrt before cuda", prefer: []string{"cuda", "cpu"}, flag: true, want: trt},
		{name: "env inserts tensorrt before cuda", prefer: []string{"cuda", "cpu"}, env: "1", want: trt},
		{name: "env on accepted", prefer: []string{"cuda", "cpu"}, env: "on", want: trt},
		{name: "env 0 keeps default", prefer: []string{"cuda", "cpu"}, env: "0", want: def},
		{name: "unparseable env keeps default", prefer: []string{"cuda", "cpu"}, env: "maybe", want: def},
		{name: "flag wins over env off", prefer: []string{"cuda", "cpu"}, flag: true, env: "0", want: trt},
		{name: "cpu appended after the inserted pair", prefer: []string{"cuda"}, flag: true, want: trt},
		{name: "cuda not first: tensorrt goes right before it", prefer: []string{"openvino", "cuda"}, flag: true,
			want: []Provider{ProviderOpenVINO, ProviderTensorRT, ProviderCUDA, ProviderCPU}},
		{name: "cpu-only chain untouched", prefer: []string{"cpu"}, flag: true, want: []Provider{ProviderCPU}},
		{name: "empty chain untouched", prefer: nil, flag: true, want: []Provider{ProviderCPU}},
		{name: "coreml chain untouched", prefer: []string{"coreml", "cpu"}, flag: true,
			want: []Provider{ProviderCoreML, ProviderCPU}},
		{name: "already lists tensorrt: unchanged", prefer: []string{"tensorrt", "cuda", "cpu"}, flag: true, want: trt},
		{name: "already lists tensorrt after cuda: unchanged", prefer: []string{"cuda", "tensorrt"}, flag: true,
			want: []Provider{ProviderCUDA, ProviderTensorRT, ProviderCPU}},
		{name: "manifest tensorrt kept without the opt-in", prefer: []string{"tensorrt", "cuda", "cpu"}, want: trt},
		{name: "VISIONSERVE_EP replaces the chain", prefer: []string{"cuda", "cpu"}, ep: "cpu",
			want: []Provider{ProviderCPU}},
		{name: "VISIONSERVE_EP wins over the flag", prefer: []string{"cuda", "cpu"}, flag: true, ep: "cuda", want: def},
		{name: "VISIONSERVE_EP wins over the env", prefer: []string{"cuda", "cpu"}, env: "1", ep: "cuda", want: def},
		{name: "VISIONSERVE_EP=tensorrt drops cuda", prefer: []string{"cuda", "cpu"}, ep: "tensorrt",
			want: []Provider{ProviderTensorRT, ProviderCPU}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clearEPEnv(t)
			t.Setenv(tensorRTEnv, tc.env)
			t.Setenv(epOverrideEnv, tc.ep)
			SetTensorRT(tc.flag)
			got, err := ResolveProviders(tc.prefer)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !equalProviders(got, tc.want) {
				t.Errorf("ResolveProviders(%v) flag=%v %s=%q %s=%q = %v, want %v",
					tc.prefer, tc.flag, tensorRTEnv, tc.env, epOverrideEnv, tc.ep, got, tc.want)
			}
		})
	}
}

func TestTensorRTRequested(t *testing.T) {
	for _, tc := range []struct {
		env  string
		flag bool
		want bool
	}{
		{"", false, false}, // the default: off
		{"1", false, true}, {"true", false, true}, {"on", false, true}, {"YES", false, true},
		{"0", false, false}, {"off", false, false}, {"sometimes", false, false},
		{"", true, true}, {"0", true, true}, // the flag only turns it on
	} {
		clearEPEnv(t)
		t.Setenv(tensorRTEnv, tc.env)
		SetTensorRT(tc.flag)
		if got := TensorRTRequested(); got != tc.want {
			t.Errorf("%s=%q flag=%v: TensorRTRequested() = %v, want %v", tensorRTEnv, tc.env, tc.flag, got, tc.want)
		}
	}
}

// TestTensorRTOptInFallsBackToCUDAWithoutLib: the opt-in must not break a host without
// libnvinfer.so.10 — availableProviders drops tensorrt before any provider is appended (loading
// the TRT provider without it aborts the process), leaving CUDA → CPU. On a host with TensorRT
// the resolved chain is kept.
func TestTensorRTOptInFallsBackToCUDAWithoutLib(t *testing.T) {
	clearEPEnv(t)
	SetTensorRT(true)
	chain, err := ResolveProviders([]string{"cuda", "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	got := availableProviders(chain)
	want := []Provider{ProviderTensorRT, ProviderCUDA, ProviderCPU}
	if !TRTAvailable() {
		want = []Provider{ProviderCUDA, ProviderCPU}
	}
	if !equalProviders(got, want) {
		t.Errorf("availableProviders(%v) with TRTAvailable()=%v = %v, want %v", chain, TRTAvailable(), got, want)
	}
}

// TestTRTHintOnlyWhenRequestedAndMissing: the default (CUDA) needs no hint; the hint appears
// only when TensorRT was asked for and cannot load.
func TestTRTHintOnlyWhenRequestedAndMissing(t *testing.T) {
	clearEPEnv(t)
	if h := TRTHint(); h != "" {
		t.Errorf("TRTHint() without the opt-in = %q, want empty", h)
	}
	SetTensorRT(true)
	h := TRTHint()
	if TRTAvailable() && h != "" {
		t.Errorf("TRTHint() with TensorRT available = %q, want empty", h)
	}
	if !TRTAvailable() && h == "" {
		t.Error("TRTHint() with the opt-in and no libnvinfer = empty, want a hint")
	}
	t.Setenv(epOverrideEnv, "cuda")
	if h := TRTHint(); h != "" {
		t.Errorf("TRTHint() under VISIONSERVE_EP = %q, want empty", h)
	}
}
