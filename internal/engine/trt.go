package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
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

// TRTOptions returns the TensorRT EP provider options, keyed as ORT expects them.
//
// Why this matters for serving: TensorRT does not "load" an ONNX graph, it COMPILES one
// (kernel autotuning + fusion). That build is measured in MINUTES for transformer graphs
// (GroundingDINO ≈ 155s, MobileSAM encoder ≈ 80s on an A6000). Without a cache that cost is
// paid again on EVERY session creation — i.e. every server restart and every reload after
// `--idle-unload-seconds` evicts the model, which would make idle-unload actively harmful.
// Persisting the compiled engine turns the second and later loads into a file read.
//
// THE ENGINE CACHE MUST BE NAMESPACED PER WEIGHTS FILE. ORT derives its engine cache filename
// from graph properties, and those COLLIDE between two fine-tunes of the same architecture:
// measured here, a cache built from a 17-class RF-DETR made the 22-class sibling "load" in 14 s
// and then serve the 17-class model's class head. Under `gated`/`dual` that surfaces as a loud
// shape error. Under `exact`/`folded` it would NOT — those paths read only boxes and query_feats,
// whose shapes match across checkpoints, so the server would return plausible numbers computed
// from the wrong weights. A silent wrong answer is the worst failure this cache can produce, so
// each weights file gets its own directory.
//
// The TIMING cache stays shared on purpose: it holds kernel-autotuning measurements for the GPU,
// is graph-independent by design, and is what makes the FIRST build of an unseen model faster.
//
// Returns nil (no options -> ORT defaults, no caching) if no cache directory can be resolved, so
// a read-only or HOME-less environment degrades to the previous behavior instead of failing.
func TRTOptions(modelPath string) map[string]string {
	dir := TRTCacheDir()
	if dir == "" {
		return nil
	}
	engineDir := filepath.Join(dir, "engines", engineKey(modelPath))
	if err := os.MkdirAll(engineDir, 0o755); err != nil {
		return nil // not fatal: fall back to uncached TRT
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil
	}
	return map[string]string{
		"trt_engine_cache_enable": "1",
		"trt_engine_cache_path":   engineDir,
		"trt_timing_cache_enable": "1",
		"trt_timing_cache_path":   dir,
	}
}

// engineKey identifies a weights FILE, not a model name: two registry entries may share one file
// (and should share its engine), and one path may be rewritten in place by a re-export (and must
// not). Size and modification time catch the rewrite without hashing hundreds of megabytes on
// every session creation. The absolute path is resolved first so that a symlink and its target —
// several models here are symlinks into the NAS — do not build the same engine twice.
func engineKey(modelPath string) string {
	key := modelPath
	if abs, err := filepath.Abs(modelPath); err == nil {
		if resolved, err := filepath.EvalSymlinks(abs); err == nil {
			key = resolved
		} else {
			key = abs
		}
	}
	if st, err := os.Stat(key); err == nil {
		key = fmt.Sprintf("%s|%d|%d", key, st.Size(), st.ModTime().UnixNano())
	}
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:16])
}

// TRTCacheDir resolves where compiled TensorRT engines are persisted:
//
//	$VISIONSERVE_TRT_CACHE  >  ~/.visionserve/trt-cache
//
// Mirrors the models-dir convention in internal/cli. Empty string if $HOME is unresolvable.
func TRTCacheDir() string {
	if d := strings.TrimSpace(os.Getenv("VISIONSERVE_TRT_CACHE")); d != "" {
		return d
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".visionserve", "trt-cache")
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

	// 2. Common system paths (Debian/Ubuntu multiarch + CUDA/TRT standard locations).
	//    Both x86_64 and aarch64 multiarch dirs are listed so TRT is auto-detected on
	//    Jetson (the primary edge target), where libnvinfer.so.10 lives under aarch64.
	dirs = append(dirs,
		"/usr/lib/x86_64-linux-gnu",
		"/usr/lib/aarch64-linux-gnu",
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
