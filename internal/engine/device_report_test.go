package engine

import "testing"

// member is a pool member as far as reporting is concerned (no ORT needed: nothing runs).
func member(ep Provider) *Session { return &Session{activeEP: ep} }

// A pool whose members did not all land on the same EP (say the 4th decoder copy hit a CUDA
// out-of-memory and fell back to CPU) used to report its FIRST member's EP — "gpu:0" for a pool
// that serves a share of its requests on the CPU.
func TestPoolReportsEveryMember(t *testing.T) {
	uniform := NewSessionPool([]*Session{member(ProviderCUDA), member(ProviderCUDA)})
	if got := uniform.ActiveEP(); got != ProviderCUDA {
		t.Errorf("uniform pool ActiveEP = %s, want cuda", got)
	}
	if got := RunnableDevice(uniform); got != "gpu:0" {
		t.Errorf("uniform pool device = %q, want gpu:0", got)
	}

	mixed := NewSessionPool([]*Session{member(ProviderCUDA), member(ProviderCPU), member(ProviderCUDA)})
	if got := mixed.ActiveEPs(); len(got) != 3 || got[1] != ProviderCPU {
		t.Errorf("ActiveEPs = %v, want every member's EP in order", got)
	}
	// ActiveEP must not claim the GPU for the whole pool: it reports what EVERY request is
	// guaranteed at least — the slowest member's EP.
	if got := mixed.ActiveEP(); got != ProviderCPU {
		t.Errorf("mixed pool ActiveEP = %s, want cpu (the slowest member)", got)
	}
	if got := RunnableDevice(mixed); got != "mixed(cpu,gpu:0)" {
		t.Errorf("mixed pool device = %q, want mixed(cpu,gpu:0)", got)
	}

	trtCuda := NewSessionPool([]*Session{member(ProviderTensorRT), member(ProviderCUDA)})
	if got := trtCuda.ActiveEP(); got != ProviderCUDA {
		t.Errorf("tensorrt+cuda pool ActiveEP = %s, want cuda", got)
	}
}

func TestRunnableDeviceOfASingleSession(t *testing.T) {
	for ep, want := range map[Provider]string{
		ProviderTensorRT: "gpu:0+trt",
		ProviderCUDA:     "gpu:0",
		ProviderOpenVINO: "openvino:0",
		ProviderCPU:      "cpu",
	} {
		if got := RunnableDevice(member(ep)); got != want {
			t.Errorf("RunnableDevice(%s) = %q, want %q", ep, got, want)
		}
	}
}
