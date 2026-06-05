package engine

import (
	"os"
	"strings"
	"sync"
)

const trtLibName = "libnvinfer.so.10"

var (
	trtOnce    sync.Once
	trtAvail   bool
	trtLibPath string
)

// TRTAvailable reports whether libnvinfer.so.10 is present on the system.
// The check is performed once and cached; subsequent calls are instant.
// It searches LD_LIBRARY_PATH first, then common system paths.
func TRTAvailable() bool {
	trtOnce.Do(func() { trtLibPath, trtAvail = findTRTLib() })
	return trtAvail
}

// TRTLibPath returns the absolute path to the found libnvinfer.so.10,
// or an empty string if not found. Triggers the check if not yet run.
func TRTLibPath() string {
	TRTAvailable()
	return trtLibPath
}

// TRTHint returns a human-readable recommendation to install TensorRT,
// or an empty string if TRT is already available.
func TRTHint() string {
	if TRTAvailable() {
		return ""
	}
	return "TensorRT not found (libnvinfer.so.10) — install for 10-50× faster inference on transformer models (GroundingDINO, MobileSAM). " +
		"Check LD_LIBRARY_PATH or install TensorRT: https://developer.nvidia.com/tensorrt"
}

func findTRTLib() (string, bool) {
	var dirs []string

	// 1. LD_LIBRARY_PATH — user may have set this to point at a custom TRT install.
	if ldPath := os.Getenv("LD_LIBRARY_PATH"); ldPath != "" {
		for _, d := range strings.Split(ldPath, ":") {
			if d != "" {
				dirs = append(dirs, d)
			}
		}
	}

	// 2. Common system paths (Debian/Ubuntu x86 + CUDA/TRT standard locations).
	dirs = append(dirs,
		"/usr/lib/x86_64-linux-gnu",
		"/usr/local/lib",
		"/usr/lib",
		"/usr/local/cuda/lib64",
		"/opt/tensorrt/lib",
		"/usr/local/tensorrt/lib",
	)

	for _, dir := range dirs {
		p := dir + "/" + trtLibName
		if _, err := os.Stat(p); err == nil {
			return p, true
		}
	}
	return "", false
}
