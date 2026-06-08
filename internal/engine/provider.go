package engine

import (
	"fmt"
	"os"
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

// DeviceString maps an execution provider to the device string returned in API responses.
//   - "gpu:0+trt" — TensorRT EP (fastest; requires libnvinfer.so.10)
//   - "gpu:0"     — CUDA / CoreML / DirectML EP
//   - "openvino:0"— OpenVINO EP
//   - "cpu"       — CPU fallback
func DeviceString(ep Provider) string {
	switch ep {
	case ProviderTensorRT:
		return "gpu:0+trt"
	case ProviderCUDA, ProviderCoreML, ProviderDirectML:
		return "gpu:0"
	case ProviderOpenVINO:
		return "openvino:0"
	default:
		return "cpu"
	}
}

// ResolveProviders normalizes + validates the fallback chain from the manifest (runtime.prefer).
// It always ensures CPU is present at the end of the chain so both edge and server can run.
// Important for edge: try TensorRT → CUDA → CPU.
func ResolveProviders(prefer []string) ([]Provider, error) {
	// VISIONSERVE_EP overrides the manifest's runtime.prefer chain. Intended for
	// edge EP×device benchmarking (force a single execution provider per run), e.g.
	// VISIONSERVE_EP=cpu, =cuda, or =tensorrt. CPU is still appended as the final
	// fallback below, so an unavailable GPU EP degrades gracefully rather than failing.
	if ov := strings.TrimSpace(os.Getenv("VISIONSERVE_EP")); ov != "" {
		prefer = strings.Split(ov, ",")
	}

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
