//go:build windows

package engine

import ort "github.com/yalue/onnxruntime_go"

// setDeterministicCompute is not wired on Windows: the binding loads ONNX Runtime with
// LoadLibrary there, and the dlfcn lookup in deterministic_cgo.go has no counterpart here yet.
// GPU sessions (DirectML) are created as before.
func setDeterministicCompute(*ort.SessionOptions) error { return errDeterministicUnsupported }
