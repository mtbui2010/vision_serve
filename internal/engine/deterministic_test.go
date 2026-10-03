package engine

import (
	"encoding/hex"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestDeterministicRequested(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want bool
	}{
		{"", false}, // the default: off
		{"1", true}, {"true", true}, {"on", true}, {"YES", true},
		{"0", false}, {"false", false}, {"off", false}, {"No", false}, {" 0 ", false},
		{"sometimes", false}, // unparseable: keep the default
	} {
		t.Setenv(deterministicEnv, tc.env)
		if got := deterministicRequested(); got != tc.want {
			t.Errorf("%s=%q: deterministicRequested() = %v, want %v", deterministicEnv, tc.env, got, tc.want)
		}
	}
}

// TestApplyDeterministicOnlyOnGPU: CPU sessions are never touched (their outputs stay
// bit-identical to earlier builds), GPU sessions are when VISIONSERVE_DETERMINISTIC=1, and a
// failure never fails the session.
func TestApplyDeterministicOnlyOnGPU(t *testing.T) {
	orig := setDeterministic
	t.Cleanup(func() { setDeterministic = orig })
	calls := 0
	setDeterministic = func(*ort.SessionOptions) error { calls++; return errors.New("injected") }

	for _, tc := range []struct {
		ep   Provider
		env  string
		want int
	}{
		{ProviderCPU, "", 0},
		{ProviderCPU, "1", 0},
		{ProviderCUDA, "", 0},
		{ProviderCUDA, "1", 1},
		{ProviderTensorRT, "on", 1},
		{ProviderCoreML, "1", 1},
		{ProviderCUDA, "0", 0},
	} {
		t.Setenv(deterministicEnv, tc.env)
		calls = 0
		applyDeterministic(nil, tc.ep) // the injected error must only be logged
		if calls != tc.want {
			t.Errorf("ep=%s env=%q: setDeterministic called %d times, want %d", tc.ep, tc.env, calls, tc.want)
		}
	}
}

// TestOrtAPIIndices recounts, in the onnxruntime_c_api.h shipped with the binding, the OrtApi
// members setDeterministicCompute calls by index. struct OrtApi is a plain table of function
// pointers, one per ';'-terminated member, which ORT only ever appends to.
func TestOrtAPIIndices(t *testing.T) {
	dir, err := bindingModuleDir()
	if err != nil {
		t.Skipf("cannot locate the binding module: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "onnxruntime_c_api.h"))
	if err != nil {
		t.Skipf("binding header not available: %v", err)
	}
	src := string(raw)

	if m := regexp.MustCompile(`#define ORT_API_VERSION (\d+)`).FindStringSubmatch(src); m == nil {
		t.Fatal("ORT_API_VERSION not found")
	} else if v, _ := strconv.Atoi(m[1]); v < ortAPIVersionDeterministic {
		t.Fatalf("binding header is C API %d; SetDeterministicCompute needs %d", v, ortAPIVersionDeterministic)
	}

	start := strings.Index(src, "struct OrtApi {")
	if start < 0 {
		t.Fatal("struct OrtApi not found")
	}
	end := strings.Index(src[start:], "\n};")
	if end < 0 {
		t.Fatal("end of struct OrtApi not found")
	}
	body := src[start+len("struct OrtApi {") : start+end]
	body = regexp.MustCompile(`(?s)/\*.*?\*/`).ReplaceAllString(body, "")
	body = regexp.MustCompile(`//[^\n]*`).ReplaceAllString(body, "")
	var members []string
	for _, m := range strings.Split(body, ";") {
		if m = strings.Join(strings.Fields(m), " "); m != "" {
			members = append(members, m)
		}
	}
	if len(members) < 280 {
		t.Fatalf("parsed only %d OrtApi members; the counting rule no longer fits the header", len(members))
	}
	for _, tc := range []struct {
		idx  int
		want string
	}{
		{0, "(ORT_API_CALL* CreateStatus)"},
		{ortAPIGetErrorMessage, "const char*(ORT_API_CALL* GetErrorMessage)(_In_ const OrtStatus* status)"},
		{ortAPIReleaseStatus, "ORT_CLASS_RELEASE(Status)"},
		{ortAPISetDeterministic, "ORT_API2_STATUS(SetDeterministicCompute, _Inout_ OrtSessionOptions* options, bool value)"},
	} {
		if !strings.Contains(members[tc.idx], tc.want) {
			t.Errorf("OrtApi member #%d is %q, want it to contain %q", tc.idx, members[tc.idx], tc.want)
		}
	}
}

func TestSessionOptionsHandle(t *testing.T) {
	if _, err := sessionOptionsHandle(nil); err == nil {
		t.Error("nil SessionOptions: want an error")
	}
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (SessionOptions come from ONNX Runtime)")
	}
	if err := ensureORT(); err != nil {
		t.Skip(err)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	if h, err := sessionOptionsHandle(opts); err != nil || h == nil {
		t.Fatalf("sessionOptionsHandle = %v, %v; want a pointer", h, err)
	}
	if err := opts.Destroy(); err != nil {
		t.Fatal(err)
	}
	if _, err := sessionOptionsHandle(opts); err == nil {
		t.Error("destroyed SessionOptions: want an error, not a dangling pointer")
	}
}

// TestSetDeterministicCompute calls the real C API on real SessionOptions, then builds a session
// from them, and checks the failure path of a library that is not the loaded one.
func TestSetDeterministicCompute(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("needs ORT_DYLIB_PATH (libonnxruntime.so)")
	}
	if err := ensureORT(); err != nil {
		t.Skip(err)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Destroy()
	if err := setDeterministicCompute(opts); err != nil {
		t.Fatalf("setDeterministicCompute: %v", err)
	}
	raw, err := hex.DecodeString(identityONNX) // device_test.go: x, y float [1,3]
	if err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(t.TempDir(), "id.onnx")
	if err := os.WriteFile(model, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := ort.NewDynamicAdvancedSession(model, []string{"x"}, []string{"y"}, opts)
	if err != nil {
		t.Fatalf("session from deterministic options: %v", err)
	}
	s.Destroy()

	orig := ortLibPath
	t.Cleanup(func() { ortLibPath = orig })
	ortLibPath = filepath.Join(t.TempDir(), "libonnxruntime.so")
	if err := setDeterministicCompute(opts); err == nil || !strings.Contains(err.Error(), "not loaded") {
		t.Errorf("unloaded library: err = %v, want a 'not loaded' error", err)
	}
}

// bindingModuleDir returns the module-cache directory of the onnxruntime_go version go.mod
// requires. It reads go.mod and `go env` only: `go list -m` (with GOFLAGS=-mod=mod, as this repo
// is built) may add lines to go.sum, and a test must not modify the checkout.
func bindingModuleDir() (string, error) {
	const mod = "github.com/yalue/onnxruntime_go"
	out, err := exec.Command("go", "env", "GOMODCACHE", "GOMOD").Output()
	if err != nil {
		return "", err
	}
	env := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(env) != 2 || env[0] == "" || env[1] == "" || env[1] == os.DevNull {
		return "", errors.New("go env GOMODCACHE/GOMOD: no module cache or no go.mod")
	}
	gomod, err := os.ReadFile(strings.TrimSpace(env[1]))
	if err != nil {
		return "", err
	}
	if regexp.MustCompile(`(?m)^\s*replace\b.*` + regexp.QuoteMeta(mod)).Match(gomod) {
		return "", errors.New(mod + " is replaced in go.mod")
	}
	m := regexp.MustCompile(`(?m)^\s*(?:require\s+)?` + regexp.QuoteMeta(mod) + `\s+(v\S+)`).FindSubmatch(gomod)
	if m == nil {
		return "", errors.New(mod + " is not required by go.mod")
	}
	// The module path has no upper-case letters, so its cache path needs no escaping.
	return filepath.Join(strings.TrimSpace(env[0]), filepath.FromSlash(mod)+"@"+string(m[1])), nil
}
