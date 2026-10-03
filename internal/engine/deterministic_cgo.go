//go:build !windows

package engine

// CGO: this is the one place VisionServe's own code calls C. It is needed because the ORT Go
// binding does not wrap OrtApi::SetDeterministicCompute and ORT exposes no other way to set it (no
// session-config key, no environment variable) — see deterministic.go. The binding is cgo already,
// so this adds no build requirement; it only adds dlfcn calls (as the binding's own setup_env.go
// does) and one call through the OrtApi function table.

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
	const void *(*GetApi)(uint32_t version);
	const char *(*GetVersionString)(void);
} vsOrtApiBase;

typedef void *(*vsSetBoolFn)(void *options, bool value);
typedef const char *(*vsErrorMessageFn)(const void *status);
typedef void (*vsReleaseStatusFn)(void *status);

// vs_set_deterministic finds the already-loaded ONNX Runtime library by the name the binding opened
// it with (RTLD_NOLOAD: never loads a second copy), fetches the OrtApi table at api_version and calls
// the member at idx_set. Returns 0 on success; otherwise a code and, for an ORT error, its message.
static int vs_set_deterministic(const char *lib, void *options, bool value, uint32_t api_version,
                                int idx_set, int idx_msg, int idx_release, char *msg, size_t msg_len) {
	void *h = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
	if (!h) return 1;
	const vsOrtApiBase *(*get_base)(void) = (const vsOrtApiBase *(*)(void))dlsym(h, "OrtGetApiBase");
	dlclose(h); // drops only the reference RTLD_NOLOAD took; the binding's handle keeps it loaded
	if (!get_base) return 2;
	const vsOrtApiBase *base = get_base();
	if (!base) return 2;
	void *const *api = (void *const *)base->GetApi(api_version);
	if (!api) return 3;
	void *status = ((vsSetBoolFn)api[idx_set])(options, value);
	if (status) {
		snprintf(msg, msg_len, "%s", ((vsErrorMessageFn)api[idx_msg])(status));
		((vsReleaseStatusFn)api[idx_release])(status);
		return 4;
	}
	return 0;
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

// setDeterministicCompute calls OrtApi::SetDeterministicCompute(options, true).
func setDeterministicCompute(o *ort.SessionOptions) error {
	h, err := sessionOptionsHandle(o)
	if err != nil {
		return err
	}
	lib := C.CString(ortLibPath)
	defer C.free(unsafe.Pointer(lib))
	var msg [256]C.char
	rc := C.vs_set_deterministic(lib, h, C.bool(true), C.uint32_t(ortAPIVersionDeterministic),
		C.int(ortAPISetDeterministic), C.int(ortAPIGetErrorMessage), C.int(ortAPIReleaseStatus),
		&msg[0], C.size_t(len(msg)))
	switch rc {
	case 0:
		return nil
	case 1:
		return fmt.Errorf("ONNX Runtime library %q is not loaded under that name", ortLibPath)
	case 2:
		return fmt.Errorf("OrtGetApiBase not found in %q", ortLibPath)
	case 3:
		return fmt.Errorf("ONNX Runtime older than 1.%d (no C API version %d)", ortAPIVersionDeterministic, ortAPIVersionDeterministic)
	default:
		return fmt.Errorf("SetDeterministicCompute: %s", C.GoString(&msg[0]))
	}
}
