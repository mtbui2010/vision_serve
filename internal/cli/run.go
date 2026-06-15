package cli

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"
	"time"

	// register image decoders
	_ "image/jpeg"
	_ "image/png"

	"github.com/disintegration/imaging"

	"visionserve/internal/imageproc"
	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/internal/morph"
	"visionserve/internal/registry"
	roipkg "visionserve/internal/roi"
	"visionserve/pkg/api"
)

// clientType labels images this (Go) CLI saves, so an auto-named output never
// collides with one written by the Python or JS client for the same image+model.
const clientType = "go"

// autoName builds a self-describing output filename: <stem>.go.<model>.<task>.<ext>.
func autoName(imagePath, model, task, ext string) string {
	base := filepath.Base(imagePath)
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	if stem == "" {
		stem = "image"
	}
	if task == "" {
		task = "result"
	}
	return fmt.Sprintf("%s.%s.%s.%s.%s", stem, clientType, model, task, ext)
}

// runRun: visionserve run <model> <image> — load + predict + print JSON to stdout.
// Runs in-process (does NOT require a running server) — this is the end-to-end MVP flow.
func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	saveFlag := fs.Bool("save", false, "save an annotated image with an auto name <stem>.go.<model>.<task>.png")
	saveAsFlag := fs.String("save-as", "", "save the annotated image to this exact path (.png/.jpg extension selects the format)")
	outFlag := fs.String("out", "", "alias of --save-as (kept for back-compat)")
	promptFlag := fs.String("prompt", "", "text prompt for open-vocab models, e.g. \"cat. remote.\" (GroundingDINO / Grounded-SAM)")
	boxFlag := fs.String("box", "", "box prompt(s) for SAM, \"x,y,w,h\" (multiple separated by ';')")
	pointFlag := fs.String("point", "", "point prompt(s) for SAM, \"x,y[,label]\" (label 1=fg 0=bg; multiple separated by ';')")
	minSizeFlag := fs.Float64("min-size", 0, "minimum bbox area as %% of image area (0 = no limit, e.g. 0.1 = 0.1%%)")
	maxSizeFlag := fs.Float64("max-size", 0, "maximum bbox area as %% of image area (0 = no limit, e.g. 90 = 90%%)")
	roiFlag := fs.String("roi", "", "region of interest \"x,y,w,h\" in ORIGINAL pixels: process only this crop, map results back")
	methodFlag := fs.String("method", "", "background model: auto|depth|sam|cv|automask (default auto)")
	bgMaxAreaFlag := fs.Float64("bg-max-area", 0, "background model (sam/automask): a mask >= this %% of image is background")
	fgMinAreaFlag := fs.Float64("fg-min-area", 0, "background model (sam/automask): drop masks below this %% of image")
	gridSizeFlag := fs.Int("grid-size", 0, "background/MobileSAM automask grid N (N*N decoder calls)")
	dilateFlag := fs.Int("dilate", 0, "morph every output mask by |N| px: >0 enlarge (dilate), <0 shrink (erode)")

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
	prompt.Method = *methodFlag
	prompt.BgMaxArea, prompt.FgMinArea = *bgMaxAreaFlag, *fgMinAreaFlag
	prompt.GridSize = *gridSizeFlag

	// Region of interest (crop semantics): infer on the crop, but keep the ORIGINAL image
	// (img) for visualization since results are mapped back to original coordinates.
	fullW, fullH := img.Bounds().Dx(), img.Bounds().Dy()
	inferImg := img
	rect, hasROI := roipkg.Clamp(roipkg.Parse(*roiFlag), fullW, fullH)
	if hasROI {
		inferImg = roipkg.Crop(img, rect)
		roipkg.ShiftPrompt(&prompt, rect)
	}

	mgr := lifecycle.NewManager(reg)
	defer mgr.Close()

	// Time ONLY the inference call (client wall-clock). The server's own
	// inference-only measurement is reported separately as res.DurationMs. Both
	// are captured BEFORE any image is drawn/saved, so visualization never
	// inflates the reported latency.
	clientStart := time.Now()
	res, err := mgr.PredictPrompt(modelName, inferImg, prompt)
	if err != nil {
		return err
	}
	clientMs := float64(time.Since(clientStart).Microseconds()) / 1000.0

	if hasROI {
		res = roipkg.MapResult(res, rect, fullW, fullH)
	}
	morph.ApplyToMasks(res.Masks, fullW, fullH, *dilateFlag)
	if *minSizeFlag > 0 || *maxSizeFlag > 0 {
		res = api.FilterBySizePct(res, *minSizeFlag, *maxSizeFlag, fullW, fullH)
	}

	// Resolve the output path: --save-as / --out (explicit) take precedence; a
	// bare --save auto-names <stem>.go.<model>.<task>.png.
	outPath := *saveAsFlag
	if outPath == "" {
		outPath = *outFlag
	}
	if outPath == "" && *saveFlag {
		outPath = autoName(imagePath, modelName, string(res.Task), "png")
	}

	// Optional: draw boxes + mask overlays onto the image and save it (demo/visualization).
	// Pure Go, no cgo. NOT counted in the durations reported above.
	if outPath != "" {
		annotated := imageproc.DrawResult(img, res)
		if err := imaging.Save(annotated, outPath); err != nil {
			return fmt.Errorf("failed to save result image %s: %w", outPath, err)
		}
	}

	device := res.Device
	if device == "" {
		device = "?"
	}
	fmt.Fprintf(os.Stderr, "predict: model=%s task=%s device=%s  client=%.1fms server=%.1fms  (%d detections, %d masks, %d grasps)\n",
		res.Model, res.Task, device, clientMs, res.DurationMs, len(res.Detections), len(res.Masks), len(res.Grasps))
	if outPath != "" {
		fmt.Fprintf(os.Stderr, "saved: %s\n", outPath)
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
