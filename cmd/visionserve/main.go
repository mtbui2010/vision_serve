// Command visionserve is the lightweight binary (server + CLI) for VisionServe.
package main

import (
	"fmt"
	"os"
	"strings"
	"syscall"

	"visionserve/internal/cli"

	// Blank-import the model packages so their init() registers a factory in the registry.
	// Adding a new model = add one import line here (do NOT modify other core code).
	_ "visionserve/internal/models/background"
	_ "visionserve/internal/models/classification"
	_ "visionserve/internal/models/clip"
	_ "visionserve/internal/models/depth"
	_ "visionserve/internal/models/efficientsam"
	_ "visionserve/internal/models/grasp"
	_ "visionserve/internal/models/groundedsam"
	_ "visionserve/internal/models/groundingdino"
	_ "visionserve/internal/models/hybrid"
	_ "visionserve/internal/models/mobilesam"
	_ "visionserve/internal/models/nanosam"
	_ "visionserve/internal/models/paddleocr"
	_ "visionserve/internal/models/rfdetr"
	_ "visionserve/internal/models/rtdetr"
	_ "visionserve/internal/models/sam2"
	_ "visionserve/internal/models/scrfd"
)

func main() {
	ensureSignalSafe()
	if err := cli.Execute(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// ensureSignalSafe re-execs the process once with GODEBUG=asyncpreemptoff=1 if it is not
// already set. ONNX Runtime / CUDA native code installs signal handlers without SA_ONSTACK;
// Go's async-preemption signal (SIGURG) arriving during a cgo inference then aborts the
// process with a fatal "non-Go code set up signal handler without SA_ONSTACK" — observed
// under concurrent load on the GroundingDINO + MobileSAM pipelines. Disabling async
// preemption removes that interaction. The runtime reads GODEBUG only at startup, so we
// re-exec (guarded against an infinite loop by VS_SIGSAFE) rather than os.Setenv.
func ensureSignalSafe() {
	if os.Getenv("VS_SIGSAFE") == "1" || strings.Contains(os.Getenv("GODEBUG"), "asyncpreemptoff=1") {
		return
	}
	gd := os.Getenv("GODEBUG")
	if gd == "" {
		gd = "asyncpreemptoff=1"
	} else {
		gd += ",asyncpreemptoff=1"
	}
	exe, err := os.Executable()
	if err != nil {
		return // best-effort: continue without re-exec
	}
	env := append(os.Environ(), "GODEBUG="+gd, "VS_SIGSAFE=1")
	_ = syscall.Exec(exe, os.Args, env) // replaces the image; falls through only on failure
}
