package cli

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"os"

	// register image decoders
	_ "image/jpeg"
	_ "image/png"

	"github.com/disintegration/imaging"

	"visionserve/internal/imageproc"
	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/pkg/api"
)

// runRun: visionserve run <model> <image> — load + predict + print JSON to stdout.
// Runs in-process (does NOT require a running server) — this is the end-to-end MVP flow.
func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	outFlag := fs.String("out", "", "save the image with drawn bboxes/masks to a file (.png/.jpg extension selects the format)")
	promptFlag := fs.String("prompt", "", "text prompt for open-vocab models, e.g. \"cat. remote.\" (GroundingDINO / Grounded-SAM)")
	boxFlag := fs.String("box", "", "box prompt(s) for SAM, \"x,y,w,h\" (multiple separated by ';')")
	pointFlag := fs.String("point", "", "point prompt(s) for SAM, \"x,y[,label]\" (label 1=fg 0=bg; multiple separated by ';')")
	minSizeFlag := fs.Float64("min-size", 0, "minimum bbox area in pixels² (0 = no limit)")
	maxSizeFlag := fs.Float64("max-size", 0, "maximum bbox area in pixels² (0 = no limit)")

	// Allow flags interleaved with positionals (e.g. `run rf-detr img.jpg --out r.png`). The
	// standard flag package stops at the first positional, so we loop: parse flags -> take 1
	// positional -> parse again.
	var positionals []string
	rem := args
	for len(rem) > 0 {
		if err := fs.Parse(rem); err != nil {
			return err
		}
		rem = fs.Args()
		if len(rem) > 0 {
			positionals = append(positionals, rem[0])
			rem = rem[1:]
		}
	}
	if len(positionals) < 2 {
		return fmt.Errorf("usage: visionserve run [--out file.png] <model> <image>")
	}
	modelName, imagePath := positionals[0], positionals[1]

	reg := registry.New(modelsDir(*modelsFlag))
	warns, err := reg.Scan()
	if err != nil {
		return err
	}
	for _, wn := range warns {
		fmt.Fprintf(os.Stderr, "registry warning: %v\n", wn)
	}
	if _, ok := reg.Get(modelName); !ok {
		return fmt.Errorf("model %q not found in registry (%s)", modelName, reg.Root())
	}

	img, err := loadImage(imagePath)
	if err != nil {
		return err
	}

	prompt, err := models.ParsePrompt(*promptFlag, *boxFlag, *pointFlag)
	if err != nil {
		return err
	}

	mgr := lifecycle.NewManager(reg)
	defer mgr.Close()

	res, err := mgr.PredictPrompt(modelName, img, prompt)
	if err != nil {
		return err
	}

	if *minSizeFlag > 0 || *maxSizeFlag > 0 {
		res = api.FilterBySize(res, *minSizeFlag, *maxSizeFlag)
	}

	// Optional: draw boxes + mask overlays onto the image and save it (demo/visualization).
	// Pure Go, no cgo.
	if *outFlag != "" {
		annotated := imageproc.DrawResult(img, res)
		if err := imaging.Save(annotated, *outFlag); err != nil {
			return fmt.Errorf("failed to save result image %s: %w", *outFlag, err)
		}
		fmt.Fprintf(os.Stderr, "saved image: %s (%d detections, %d masks)\n", *outFlag, len(res.Detections), len(res.Masks))
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(res)
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open image %s: %w", path, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image %s: %w", path, err)
	}
	return img, nil
}
