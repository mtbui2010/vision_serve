package engine

import (
	"fmt"
	"strings"
)

// Provider is an ONNX Runtime execution provider.
type Provider string

const (
	ProviderTensorRT Provider = "tensorrt"
	ProviderCUDA     Provider = "cuda"
	ProviderCoreML   Provider = "coreml"   // Apple Silicon / macOS
	ProviderDirectML Provider = "directml" // Windows GPU (AMD / Intel / NVIDIA)
	ProviderOpenVINO Provider = "openvino" // Intel CPU / iGPU / VPU
	ProviderCPU      Provider = "cpu"
)

// validProviders is the allowlist of supported EPs. CPU is always the final fallback.
var validProviders = map[Provider]bool{
	ProviderTensorRT: true,
	ProviderCUDA:     true,
	ProviderCoreML:   true,
	ProviderDirectML: true,
	ProviderOpenVINO: true,
	ProviderCPU:      true,
}

// ResolveProviders normalizes + validates the fallback chain from the manifest (runtime.prefer).
// It always ensures CPU is present at the end of the chain so both edge and server can run.
// Important for edge: try TensorRT → CUDA → CPU.
func ResolveProviders(prefer []string) ([]Provider, error) {
	seen := map[Provider]bool{}
	out := make([]Provider, 0, len(prefer)+1)
	for _, p := range prefer {
		pv := Provider(strings.ToLower(strings.TrimSpace(p)))
		if pv == "" {
			continue
		}
		if !validProviders[pv] {
			return nil, fmt.Errorf("engine: invalid execution provider %q (valid: tensorrt, cuda, coreml, directml, openvino, cpu)", p)
		}
		if seen[pv] {
			continue
		}
		seen[pv] = true
		out = append(out, pv)
	}
	if !seen[ProviderCPU] {
		out = append(out, ProviderCPU) // final fallback
	}
	return out, nil
}
