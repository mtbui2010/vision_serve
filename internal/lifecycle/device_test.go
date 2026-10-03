package lifecycle

import (
	"testing"
	"time"

	"visionserve/internal/engine"
)

// epEngine is a fakeEngine running on a chosen EP.
type epEngine struct {
	fakeEngine
	ep engine.Provider
}

func (e *epEngine) ActiveEP() engine.Provider { return e.ep }

// A pipeline's device used to come from whichever role Go's map iteration returned first, so a
// model with its encoder on the GPU and its decoder on the CPU reported either, at random.
func TestPipelineDeviceReportsEveryRole(t *testing.T) {
	on := func(ep engine.Provider) engine.Runnable { return &epEngine{ep: ep} }
	for _, c := range []struct {
		name  string
		roles map[string]engine.Runnable
		want  string
	}{
		{"no session", map[string]engine.Runnable{}, "cpu"},
		{"all on cpu", map[string]engine.Runnable{"encoder": on(engine.ProviderCPU), "decoder": on(engine.ProviderCPU)}, "cpu"},
		{"all on the gpu", map[string]engine.Runnable{"encoder": on(engine.ProviderCUDA), "decoder": on(engine.ProviderCUDA)}, "gpu:0"},
		{"split", map[string]engine.Runnable{"encoder": on(engine.ProviderCUDA), "decoder": on(engine.ProviderCPU)},
			"mixed(decoder=cpu,encoder=gpu:0)"},
		{"tensorrt and cuda", map[string]engine.Runnable{"gdino": on(engine.ProviderTensorRT), "sam": on(engine.ProviderCUDA)},
			"mixed(gdino=gpu:0+trt,sam=gpu:0)"},
	} {
		for i := 0; i < 20; i++ { // map order is randomized: the answer must not depend on it
			s := newPipelineSession("m", "embed", &testPipe{}, c.roles, 0, time.Now())
			if s.device != c.want {
				t.Fatalf("%s: device = %q, want %q", c.name, s.device, c.want)
			}
		}
	}
}
