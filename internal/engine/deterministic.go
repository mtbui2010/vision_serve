package engine

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

// Deterministic GPU kernels
//
// Some ONNX Runtime CUDA kernels are not run-to-run deterministic. The one that bit this project:
// the MobileSAM encoder's last LayerNorm2d computes a ReduceMean over the channel axis, and on the
// CUDA EP that reduction sums in an order that depends on what else the GPU is running. With other
// work on the GPU (another request, another session, another process) about one encoder run in 60
// to 300 returns an embedding that differs in the last bit of a few hundred values; the decoder's
// logit>0 threshold turns that into a mask that differs by 1–43 boundary pixels for the same
// request. On an otherwise idle GPU it did not show in 2000 runs.
//
// ORT's SessionOptions::SetDeterministicCompute(true) makes such kernels pick their deterministic
// variant. Measured on an RTX A6000 (ORT 1.26, CUDA EP), encoder under concurrent GPU load: 26/1500
// runs differing without it, 0/1500 with it, and no measurable change in latency or throughput of
// MobileSAM, RF-DETR, GroundingDINO or Grounded-SAM. The deterministic
// kernels round differently, so GPU outputs move once (GroundingDINO scores by up to ~0.002) and
// then stay put — see docs/refactor-proposal.md §6.
//
// It is OFF by default, opt-in with VISIONSERVE_DETERMINISTIC=1 for the whole process (it then
// applies to every GPU session). The accuracy numbers quoted in the shipped manifests were measured
// with ORT's default kernels, and turning this on moves them slightly (held-out mAP by up to 0.22),
// so the default keeps served GPU outputs equal to those measurements. Turn it on where the same
// request must give the same answer (e.g. a golden comparison between two builds on a GPU). CPU
// sessions are never touched: ORT's CPU kernels are already run-to-run deterministic.
//
// The binding does not wrap SetDeterministicCompute (yalue/onnxruntime_go v1.13.0, nor any
// release up to v1.36.0), and ORT has no session-config key for it, so setDeterministicCompute calls
// the C API function directly (deterministic_cgo.go). That is best effort: if it cannot be done the
// session is still created, nondeterministic, and the reason is logged once.

// deterministicEnv is the process-wide switch.
const deterministicEnv = "VISIONSERVE_DETERMINISTIC"

// deterministicRequested reports whether GPU sessions should ask for deterministic kernels:
// false unless VISIONSERVE_DETERMINISTIC parses as true. A value that does not parse keeps the
// default (off) and is reported once.
func deterministicRequested() bool {
	v := strings.TrimSpace(os.Getenv(deterministicEnv))
	if v == "" {
		return false
	}
	switch strings.ToLower(v) {
	case "on", "yes":
		return true
	case "off", "no":
		return false
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		badDeterministicEnv.Do(func() {
			fmt.Fprintf(os.Stderr, "engine: ignoring %s=%q (want 1/0, true/false, on/off); deterministic GPU kernels stay off\n",
				deterministicEnv, v)
		})
		return false
	}
	return b
}

var badDeterministicEnv sync.Once

// Indices of the OrtApi members setDeterministicCompute calls, counted in onnxruntime_c_api.h
// (struct OrtApi is a table of function pointers; ORT only ever appends to it, so an index never
// moves). TestOrtAPIIndices recounts them in the header shipped with the binding.
const (
	ortAPIVersionDeterministic = 17  // SetDeterministicCompute was added in ORT 1.17 (C API 17)
	ortAPIGetErrorMessage      = 2   // const char* GetErrorMessage(const OrtStatus*)
	ortAPIReleaseStatus        = 93  // void ReleaseStatus(OrtStatus*)
	ortAPISetDeterministic     = 273 // OrtStatus* SetDeterministicCompute(OrtSessionOptions*, bool)
)

// sessionOptionsHandle returns the OrtSessionOptions* behind the binding's SessionOptions, whose
// only field is that unexported pointer. The layout is checked, not assumed: a binding that
// changes it gets an error here (and a nondeterministic session), never a bad pointer passed to C.
func sessionOptionsHandle(o *ort.SessionOptions) (unsafe.Pointer, error) {
	if o == nil {
		return nil, errors.New("nil SessionOptions")
	}
	v := reflect.ValueOf(o).Elem()
	t := v.Type()
	if t.NumField() != 1 || t.Field(0).Type.Kind() != reflect.Pointer ||
		!strings.Contains(t.Field(0).Type.Elem().Name(), "OrtSessionOptions") {
		return nil, fmt.Errorf("unexpected layout of %s (the binding changed; update sessionOptionsHandle)", t)
	}
	p := *(*unsafe.Pointer)(unsafe.Pointer(v.Field(0).UnsafeAddr()))
	if p == nil {
		return nil, errors.New("SessionOptions already destroyed")
	}
	return p, nil
}

// applyDeterministic asks ORT for deterministic kernels on a GPU session's options when that is
// requested. It never fails the session: a failure is logged once and the session goes on as before.
func applyDeterministic(opts *ort.SessionOptions, ep Provider) {
	if ep == ProviderCPU || !deterministicRequested() {
		return
	}
	err := setDeterministic(opts)
	if err != nil {
		deterministicFailed.Do(func() {
			fmt.Fprintf(os.Stderr, "engine: could not enable deterministic GPU kernels (%v); "+
				"GPU results may differ in the last bits between runs\n", err)
		})
		return
	}
	if Trace {
		fmt.Fprintf(os.Stderr, "engine: [trace] deterministic compute on for EP %s\n", providerNames([]Provider{ep}))
	}
}

var deterministicFailed sync.Once

// setDeterministic is setDeterministicCompute; tests replace it to see when it is called.
var setDeterministic = setDeterministicCompute
