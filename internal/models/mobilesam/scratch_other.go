//go:build !unix

package mobilesam

// allocLogits returns n float32s on the Go heap; see scratch_unix.go for why unix maps them
// outside it instead.
func allocLogits(n int) ([]float32, func()) {
	return make([]float32, n), func() {}
}
