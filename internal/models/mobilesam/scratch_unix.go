//go:build unix

package mobilesam

import (
	"syscall"
	"unsafe"
)

// allocLogits returns n float32s outside the Go heap (an anonymous private mapping) and the
// function that unmaps them, or a heap slice and a no-op when the mapping fails.
//
// Why off-heap: the final pass's per-worker logit buffers are 30 MB each at 3200×2400 and live
// for the whole pass. On the Go heap they set the GC goal at twice that live set, and once the
// pass ends they sit as garbage until the heap grows to the goal — into the NEXT request, whose
// encoder allocations then stack on top. Unmapped when the pass ends, they return to the OS at
// once. Only this package's final pass touches them, and it unmaps them after every worker
// (and every ORT run writing into them) has returned.
func allocLogits(n int) ([]float32, func()) {
	if n <= 0 {
		return nil, func() {}
	}
	b, err := syscall.Mmap(-1, 0, n*4, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return make([]float32, n), func() {}
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n), func() { _ = syscall.Munmap(b) }
}
