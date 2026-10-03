package engine

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
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
//   - "gpu:0+trt" — TensorRT EP (opt-in, see TensorRTRequested; requires libnvinfer.so.10)
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

// RunnableDevice is the device string of one session or pool. A pool whose members run on
// different EPs reports all of them, e.g. "mixed(cpu,gpu:0)", instead of its first member's; a
// uniform pool and a single session report exactly DeviceString of their EP.
func RunnableDevice(r Runnable) string {
	multi, ok := r.(interface{ ActiveEPs() []Provider })
	if !ok {
		return DeviceString(r.ActiveEP())
	}
	seen := map[string]bool{}
	var devs []string
	for _, ep := range multi.ActiveEPs() {
		if d := DeviceString(ep); !seen[d] {
			seen[d] = true
			devs = append(devs, d)
		}
	}
	switch len(devs) {
	case 0:
		return DeviceString(r.ActiveEP())
	case 1:
		return devs[0]
	}
	sort.Strings(devs)
	return "mixed(" + strings.Join(devs, ",") + ")"
}

// epSpeedRank orders EPs from fastest to slowest for slowestEP; CPU is the floor.
var epSpeedRank = map[Provider]int{
	ProviderTensorRT: 0,
	ProviderCUDA:     1,
	ProviderCoreML:   2,
	ProviderDirectML: 3,
	ProviderOpenVINO: 4,
	ProviderCPU:      5,
}

// slowestEP returns the slowest EP among eps (ProviderCPU for an empty list).
func slowestEP(eps []Provider) Provider {
	if len(eps) == 0 {
		return ProviderCPU
	}
	slowest := eps[0]
	for _, ep := range eps[1:] {
		if epSpeedRank[ep] > epSpeedRank[slowest] {
			slowest = ep
		}
	}
	return slowest
}

// TensorRT opt-in
//
// The default NVIDIA chain is CUDA → CPU: every shipped manifest and catalog entry says
// `prefer: [cuda, cpu]`. TensorRT stays off unless asked for: on GroundingDINO it measured ~1.5x
// faster but 6.8 held-out mAP lower, and it rebuilds its engine for every new prompt length
// (10-40 s stalls) — BUGS_TO_FIX.md #3.
//
// `visionserve serve --tensorrt` / `run --tensorrt` (SetTensorRT) or VISIONSERVE_TENSORRT=1
// turn it on for the whole process. Then every resolved chain that contains cuda gets tensorrt
// inserted right before it ([cuda, cpu] → [tensorrt, cuda, cpu]). A chain without cuda (cpu-only,
// coreml, directml, openvino) is left alone, and so is a manifest that already lists tensorrt.
// If libnvinfer.so.10 is missing, availableProviders (ort.go) drops tensorrt before any provider
// is appended, and the session lands on CUDA exactly as without the opt-in.

// tensorRTEnv is the process-wide TensorRT opt-in.
const tensorRTEnv = "VISIONSERVE_TENSORRT"

// epOverrideEnv replaces every manifest's chain outright (see ResolveProviders).
const epOverrideEnv = "VISIONSERVE_EP"

var (
	tensorRTFlag   atomic.Bool
	badTensorRTEnv sync.Once
)

// SetTensorRT records the --tensorrt CLI flag. on=false only clears the flag:
// VISIONSERVE_TENSORRT still applies. Call it before any model is loaded.
func SetTensorRT(on bool) { tensorRTFlag.Store(on) }

// TensorRTRequested reports whether TensorRT was opted into, by --tensorrt or
// VISIONSERVE_TENSORRT. A value of the env var that does not parse counts as off and is reported
// once. VISIONSERVE_EP overrides the opt-in (see EPOverride); this does not look at it.
func TensorRTRequested() bool {
	if tensorRTFlag.Load() {
		return true
	}
	return boolEnv(tensorRTEnv, "TensorRT stays off (default chain CUDA → CPU)", &badTensorRTEnv)
}

// EPOverride returns VISIONSERVE_EP as set (trimmed), or "" when it is unset.
func EPOverride() string { return strings.TrimSpace(os.Getenv(epOverrideEnv)) }

// withTensorRT inserts tensorrt right before the first cuda. A chain without cuda, or one that
// already lists tensorrt anywhere, is returned unchanged.
func withTensorRT(chain []Provider) []Provider {
	at := -1
	for i, p := range chain {
		if p == ProviderTensorRT {
			return chain
		}
		if p == ProviderCUDA && at < 0 {
			at = i
		}
	}
	if at < 0 {
		return chain
	}
	out := make([]Provider, 0, len(chain)+1)
	out = append(out, chain[:at]...)
	out = append(out, ProviderTensorRT)
	return append(out, chain[at:]...)
}

// ResolveProviders normalizes + validates the fallback chain from the manifest (runtime.prefer).
// It always ensures CPU is present at the end of the chain so both edge and server can run.
// The NVIDIA default is CUDA → CPU; the TensorRT opt-in (TensorRTRequested) makes it
// TensorRT → CUDA → CPU. VISIONSERVE_EP replaces the whole chain and wins over both.
func ResolveProviders(prefer []string) ([]Provider, error) {
	// VISIONSERVE_EP REPLACES the manifest's runtime.prefer chain: it is not merged with it, and
	// the TensorRT opt-in is not applied on top of it. Intended for edge EP×device benchmarking
	// (force a single execution provider per run), e.g. VISIONSERVE_EP=cpu, =cuda, or =tensorrt
	// (the last gives [tensorrt, cpu], without CUDA). CPU is still appended as the final
	// fallback below, so an unavailable GPU EP degrades gracefully rather than failing.
	override := EPOverride()
	if override != "" {
		prefer = strings.Split(override, ",")
	}

	seen := map[Provider]bool{}
	out := make([]Provider, 0, len(prefer)+2)
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
	if override == "" && TensorRTRequested() {
		out = withTensorRT(out)
	}
	if !seen[ProviderCPU] {
		out = append(out, ProviderCPU) // final fallback
	}
	return out, nil
}
