// Package cli defines the visionserve subcommands.
// It uses the standard flag package (no heavy CLI dependency) to keep the binary lightweight.
package cli

import (
	"fmt"
	"os"
	"path/filepath"

	"visionserve/internal/engine"
)

// Version is the binary version (overridden at build time via -ldflags).
var Version = "0.1.4-dev"

const usage = `visionserve — Ollama for Computer Vision (local-first, edge-GPU)

Usage:
  visionserve serve                 start the HTTP server (port 11435)
  visionserve run <model> <image>   load model + predict + print JSON to stdout
  visionserve list                  list models in the registry
  visionserve ps                    show models loaded in memory (requires a running server)
  visionserve rm <model>            unload a model from memory (requires a running server)
  visionserve pull <model>          download a curated model from HuggingFace into the registry (no arg = list)
  visionserve pull <folder>         validate a local model folder (manifest.yaml + .onnx) + install it into the registry
  visionserve version               print the version

Common flags:
  --models <dir>   model registry directory (default ~/.visionserve/models, or $VISIONSERVE_MODELS)
  --addr <host:port>  server address (default :11435)
  --out <file>     (run) save the image with drawn bboxes/masks to a .png/.jpg file
  --prompt <text>  (run) text prompt for open-vocab models, e.g. "cat. remote."
  --box <x,y,w,h>  (run) box prompt for SAM (multiple separated by ';')
  --point <x,y[,l]> (run) point prompt for SAM (label 1=fg 0=bg)
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
	case "run":
		return runRun(args[2:])
	case "list", "ls":
		return runList(args[2:])
	case "ps":
		return runPs(args[2:])
	case "rm":
		return runRm(args[2:])
	case "pull":
		return runPull(args[2:])
	case "version", "--version", "-v":
		fmt.Printf("visionserve %s\n", Version)
		if engine.TRTAvailable() {
			fmt.Printf("TensorRT: available (%s)\n", engine.TRTLibPath())
		} else {
			fmt.Println("TensorRT: not found — install for 10-50× faster GPU inference")
			fmt.Println("          check LD_LIBRARY_PATH or visit https://developer.nvidia.com/tensorrt")
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
