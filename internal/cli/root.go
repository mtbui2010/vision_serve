// Package cli defines the visionserve subcommands.
// It uses the standard flag package (no heavy CLI dependency) to keep the binary lightweight.
package cli

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"visionserve/internal/engine"
)

// Version is the binary version (overridden at build time via -ldflags).
var Version = "0.1.18-dev"

const usage = `visionserve — Ollama for Computer Vision (local-first, edge-GPU)

Usage:
  visionserve serve                 start the HTTP server on 127.0.0.1:11435 (this machine only)
  visionserve run <model> <image>   load model + predict + print JSON to stdout (alias: predict)
  visionserve list                  list models in the registry
  visionserve ps                    show models loaded in memory (requires a running server)
  visionserve rm <model>            unload a model from memory (requires a running server)
  visionserve pull <model>          download a curated model from HuggingFace into the registry (no arg = list)
  visionserve pull <folder>         validate a local model folder (manifest.yaml + .onnx) + install it into the registry
  visionserve convert <fmt> <ckpt>  convert a PyTorch/RF-DETR/HuggingFace/TensorFlow checkpoint to ONNX + install it
                                    (runs the converter image via Docker; see: visionserve convert --help)
  visionserve check <model> --images DIR
                                    does the served model behave like your training pipeline? verdict +
                                    likely causes (needs a running server; see: visionserve check --help)
  visionserve version               print the version

Common flags:
  --models <dir>   model registry directory (default ~/.visionserve/models, or $VISIONSERVE_MODELS)
  --addr <host:port>  server address (default 127.0.0.1:11435). serve listens on loopback only
                   by default (no authentication, so no LAN exposure); --addr :11435 listens
                   on all interfaces (what the Docker images pass)
  --tensorrt       (serve, run) opt into the TensorRT EP. The NVIDIA default is CUDA → CPU;
                   with --tensorrt (or VISIONSERVE_TENSORRT=1) it is TensorRT → CUDA → CPU,
                   falling back to CUDA when libnvinfer.so.10 is missing. TensorRT measured
                   6.8 mAP lower on GroundingDINO and rebuilds per prompt length
                   (BUGS_TO_FIX.md #3). VISIONSERVE_EP=<ep,...> replaces the whole chain.
  --save           (run) save an annotated image, auto-named <stem>.go.<model>.<task>.png
  --save-as <file> (run) save the annotated image to this exact path (.png/.jpg; alias: --out)
  --prompt <text>  (run) text prompt for open-vocab models, e.g. "cat. remote."
  --box <x,y,w,h>  (run) box prompt for SAM (multiple separated by ';')
  --point <x,y[,l]> (run) point prompt for SAM (label 1=fg 0=bg)
  --min-size <pct> (run) drop objects whose bbox area is below pct% of the image
  --max-size <pct> (run) drop objects whose bbox area is above pct% of the image

run prints the result JSON to stdout and a one-line summary (model/task/device,
client + server timings) to stderr. Timings exclude image draw/save.
`

// Execute is the entrypoint for the CLI. args is the full os.Args.
func Execute(args []string) error {
	if len(args) < 2 {
		fmt.Print(usage)
		return nil
	}
	switch args[1] {
	case "serve":
		return runServe(args[2:])
	case "run", "predict":
		return runRun(args[2:])
	case "list", "ls":
		return runList(args[2:])
	case "ps":
		return runPs(args[2:])
	case "rm":
		return runRm(args[2:])
	case "pull":
		return runPull(args[2:])
	case "convert":
		return runConvert(args[2:])
	case "check":
		return runCheck(args[2:])
	case "version", "--version", "-v":
		fmt.Printf("visionserve %s\n", Version)
		for _, line := range epStatus() {
			fmt.Println(line)
		}
		return nil
	case "help", "--help", "-h":
		fmt.Print(usage)
		return nil
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %q\n\n", args[1])
		fmt.Print(usage)
		return fmt.Errorf("unknown command: %s", args[1])
	}
}

// tensorRTUsage is the help text of the --tensorrt flag (serve and run).
const tensorRTUsage = "opt into the TensorRT EP: every chain with cuda becomes tensorrt → cuda → cpu " +
	"(default cuda → cpu; same as VISIONSERVE_TENSORRT=1). TensorRT measured 6.8 mAP lower on " +
	"GroundingDINO and rebuilds its engine per prompt length (BUGS_TO_FIX.md #3)"

// addTensorRTFlag defines --tensorrt on fs; applyTensorRTFlag hands its value to the engine.
// Both serve and run load models in this process, so the flag must be applied before any load.
func addTensorRTFlag(fs *flag.FlagSet) *bool { return fs.Bool("tensorrt", false, tensorRTUsage) }

func applyTensorRTFlag(on *bool) {
	if on != nil && *on {
		engine.SetTensorRT(true)
	}
}

// trtCaveat is the one-line reason TensorRT is not the default.
const trtCaveat = "TensorRT caveat: measured 6.8 held-out mAP lower on GroundingDINO, and it rebuilds " +
	"its engine for every new prompt length (BUGS_TO_FIX.md #3)"

// epStatus describes the NVIDIA execution-provider chain this process uses, for the serve
// startup log and `visionserve version`. It may scan the filesystem for libnvinfer.so.10.
func epStatus() []string {
	if ov := engine.EPOverride(); ov != "" {
		return []string{fmt.Sprintf("EP chain: VISIONSERVE_EP=%s replaces every model's runtime.prefer "+
			"(cpu appended last); --tensorrt / VISIONSERVE_TENSORRT are ignored", ov)}
	}
	lib := "libnvinfer.so.10 not found"
	if engine.TRTAvailable() {
		lib = "libnvinfer: " + engine.TRTLibPath()
	}
	if !engine.TensorRTRequested() {
		return []string{
			"EP chain: CUDA → CPU (default). TensorRT: off; enable with --tensorrt or VISIONSERVE_TENSORRT=1 (" + lib + ")",
			trtCaveat,
		}
	}
	if !engine.TRTAvailable() {
		return []string{
			"EP chain: CUDA → CPU. TensorRT was requested (--tensorrt / VISIONSERVE_TENSORRT) but " + lib +
				"; falling back to CUDA (https://developer.nvidia.com/tensorrt)",
		}
	}
	return []string{
		"EP chain: TensorRT → CUDA → CPU (TensorRT ON via --tensorrt / VISIONSERVE_TENSORRT; " + lib + ")",
		trtCaveat,
	}
}

// defaultModelsDir returns the default model registry path:
// ~/.visionserve/models (resolved from $HOME at runtime).
// In a Docker container running as root this is /root/.visionserve/models.
func defaultModelsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "/root/.visionserve/models"
	}
	return filepath.Join(home, ".visionserve", "models")
}

// modelsDir resolves the registry directory:
//
//	--models flag  >  $VISIONSERVE_MODELS env  >  ~/.visionserve/models
func modelsDir(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	if env := os.Getenv("VISIONSERVE_MODELS"); env != "" {
		return env
	}
	return defaultModelsDir()
}
