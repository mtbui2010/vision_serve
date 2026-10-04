package cli

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"io"
	"math"
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
	"visionserve/internal/registry"
	"visionserve/internal/server"
	"visionserve/internal/templates"
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

// runTemplateSet is the name `run --template` registers its images under in the in-process
// template store (the server's POST /api/templates store does not exist without a server).
const runTemplateSet = "run"

// maxDepthFileBytes bounds the --depth file read into memory: the server caps an uploaded depth
// part at the same 128 MB (server.maxTensorBytes), and no valid map is larger.
const maxDepthFileBytes = 128 << 20

// listFlag is a repeatable string flag (--template a.png --template b.png).
type listFlag []string

func (l *listFlag) String() string     { return strings.Join(*l, ",") }
func (l *listFlag) Set(v string) error { *l = append(*l, v); return nil }

// runOptions are the `visionserve run` flags that become request options: the fields of
// api.PredictJSONRequest (same names as the server's form fields, dashed), plus the two that
// stand in for an upload: the depth map as a file, and template images (there is no server-side
// template store to name).
type runOptions struct {
	api.PredictJSONRequest
	depthFile string
	templates listFlag
}

// addRequestFlags declares the request-option flags of `run` on fs.
func addRequestFlags(fs *flag.FlagSet, o *runOptions) {
	q := &o.PredictJSONRequest
	fs.StringVar(&q.Prompt, "prompt", "", "text prompt for open-vocab models, e.g. \"cat. remote.\" (GroundingDINO / Grounded-SAM)")
	fs.StringVar(&q.Box, "box", "", "box prompt(s) for SAM, \"x,y,w,h\" (multiple separated by ';')")
	fs.StringVar(&q.Point, "point", "", "point prompt(s) for SAM, \"x,y[,label]\" (label 1=fg 0=bg; multiple separated by ';')")
	fs.Float64Var(&q.BoxThreshold, "box-threshold", 0, "GroundingDINO family: minimum box score (0 = manifest default)")
	fs.Float64Var(&q.TextThreshold, "text-threshold", 0, "GroundingDINO family: second score floor (0 = manifest default, 0.25)")
	fs.Float64Var(&q.MinSize, "min-size", 0, "minimum bbox area as %% of image area (0 = no limit, e.g. 0.1 = 0.1%%)")
	fs.Float64Var(&q.MaxSize, "max-size", 0, "maximum bbox area as %% of image area (0 = no limit, e.g. 90 = 90%%)")
	fs.StringVar(&q.ROI, "roi", "", "region of interest \"x,y,w,h\" in ORIGINAL pixels (or 0..1 fractions): process only this crop, map results back")
	fs.StringVar(&q.Method, "method", "", "algorithm: background auto|depth|sam|cv|automask (default auto); rfdetr-textalign exact|dual")
	fs.Float64Var(&q.BgMaxArea, "bg-max-area", 0, "background model (sam/automask): a mask >= this %% of image is background")
	fs.Float64Var(&q.FgMinArea, "fg-min-area", 0, "background model (sam/automask): drop masks below this %% of image")
	fs.IntVar(&q.GridSize, "grid-size", 0, "background/MobileSAM automask grid N (N*N decoder calls)")
	fs.IntVar(&q.Dilate, "dilate", 0, "morph every output mask by |N| px: >0 enlarge (dilate), <0 shrink (erode)")
	fs.Float64Var(&q.GripperMin, "gripper-min", 0, "grasp models: smallest jaw opening in ORIGINAL pixels (0 = manifest default)")
	fs.Float64Var(&q.GripperMax, "gripper-max", 0, "grasp models: largest jaw opening in ORIGINAL pixels (0 = manifest default)")
	fs.Float64Var(&q.ClaimThreshold, "claim-threshold", 0, "rfdetr-textalign* --method dual: probability the trained head needs to name a box (0 = default, >= 1 = never)")
	fs.Float64Var(&q.CropTemp, "crop-temp", 0, "softmax temperature of the SigLIP crop namer (0 = model default; lower = more decisive)")
	fs.StringVar(&o.depthFile, "depth", "", "aligned depth map for the background model: raw little-endian file of --depth-dtype")
	fs.StringVar(&q.DepthDtype, "depth-dtype", "", "element type of the --depth file: uint16 (default) or float32")
	fs.IntVar(&q.DepthWidth, "depth-width", 0, "width of the --depth map (default: the image's)")
	fs.IntVar(&q.DepthHeight, "depth-height", 0, "height of the --depth map (default: the image's)")
	fs.Var(&o.templates, "template", "instance_detection models: a template image file (repeat for several); replaces the server's template_name")
}

// request turns the parsed options into the request POST /api/predict would decode: the depth
// file becomes depth_base64, template images are decoded and registered in an in-process store
// the caller wires into the lifecycle manager (nil without --template).
func (o *runOptions) request(model string) (server.Request, *templates.Store, error) {
	q := o.PredictJSONRequest
	q.Model = model
	for name, v := range map[string]float64{
		"box-threshold": q.BoxThreshold, "text-threshold": q.TextThreshold, "min-size": q.MinSize,
		"max-size": q.MaxSize, "bg-max-area": q.BgMaxArea, "fg-min-area": q.FgMinArea,
		"gripper-min": q.GripperMin, "gripper-max": q.GripperMax, "claim-threshold": q.ClaimThreshold,
		"crop-temp": q.CropTemp,
	} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return server.Request{}, nil, fmt.Errorf("--%s must be a finite number, got %v", name, v)
		}
	}
	if err := o.checkDepthFlags(); err != nil {
		return server.Request{}, nil, err
	}
	if o.depthFile != "" {
		raw, err := readDepthFile(o.depthFile)
		if err != nil {
			return server.Request{}, nil, err
		}
		q.DepthBase64 = base64.StdEncoding.EncodeToString(raw)
	}
	var store *templates.Store
	if len(o.templates) > 0 {
		imgs := make([]image.Image, 0, len(o.templates))
		for _, p := range o.templates {
			img, err := loadImage(p)
			if err != nil {
				return server.Request{}, nil, fmt.Errorf("--template: %w", err)
			}
			imgs = append(imgs, img)
		}
		store = templates.New()
		if err := store.Register(runTemplateSet, imgs); err != nil {
			return server.Request{}, nil, fmt.Errorf("--template: %w", err)
		}
		q.TemplateName = runTemplateSet
	}
	return server.Request{PredictJSONRequest: q}, store, nil
}

// checkDepthFlags refuses depth flags the server would silently misread: an unknown dtype (the
// server reads anything but float32 as uint16), half a size, or size flags without a file.
func (o *runOptions) checkDepthFlags() error {
	q := &o.PredictJSONRequest
	switch q.DepthDtype {
	case "", "uint16", "float32":
	default:
		return fmt.Errorf("--depth-dtype must be uint16 or float32, got %q", q.DepthDtype)
	}
	if o.depthFile == "" {
		if q.DepthDtype != "" || q.DepthWidth != 0 || q.DepthHeight != 0 {
			return fmt.Errorf("--depth-dtype / --depth-width / --depth-height need --depth")
		}
		return nil
	}
	if (q.DepthWidth == 0) != (q.DepthHeight == 0) || q.DepthWidth < 0 || q.DepthHeight < 0 {
		return fmt.Errorf("--depth-width and --depth-height go together (both > 0), or neither (= the image's size); got %dx%d",
			q.DepthWidth, q.DepthHeight)
	}
	return nil
}

func readDepthFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("--depth: %w", err)
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, maxDepthFileBytes+1))
	if err != nil {
		return nil, fmt.Errorf("--depth: reading %s: %w", path, err)
	}
	if len(raw) > maxDepthFileBytes {
		return nil, fmt.Errorf("--depth: %s is larger than %d MB", path, maxDepthFileBytes>>20)
	}
	return raw, nil
}

// runRun: visionserve run <model> <image> — load + predict + print JSON to stdout.
// Runs in-process (does NOT require a running server) — this is the end-to-end MVP flow.
func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	modelsFlag := fs.String("models", "", "model registry directory")
	saveFlag := fs.Bool("save", false, "save an annotated image with an auto name <stem>.go.<model>.<task>.png")
	saveAsFlag := fs.String("save-as", "", "save the annotated image to this exact path (.png/.jpg extension selects the format)")
	outFlag := fs.String("out", "", "alias of --save-as (kept for back-compat)")
	var opts runOptions
	addRequestFlags(fs, &opts)
	trtFlag := addTensorRTFlag(fs)

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
	applyTensorRTFlag(trtFlag) // run loads the model in this process
	if len(positionals) < 2 {
		return fmt.Errorf("usage: visionserve run [--out file.png] <model> <image>")
	}
	modelName, imagePath := positionals[0], positionals[1]

	// Request options first: a flag mistake (bad depth file, missing template) is reported
	// before the registry scan and the model load.
	req, tmpl, err := opts.request(modelName)
	if err != nil {
		return err
	}

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

	// The same request model and prompt mapping as POST /api/predict (one place per option).
	prompt, err := req.ToPrompt(img.Bounds().Dx(), img.Bounds().Dy())
	if err != nil {
		return err
	}

	mgr := lifecycle.NewManager(reg)
	defer mgr.Close()
	// Always a store, as under `serve`: without one the manager would ignore a template name, and
	// an instance_detection model run with no --template gets the server's "no templates" error.
	if tmpl == nil {
		tmpl = templates.New()
	}
	mgr.SetTemplateStore(tmpl)

	// Time ONLY the prediction (client wall-clock): ROI crop + inference + mapping back. The
	// server's own inference-only measurement is reported separately as res.DurationMs. Both
	// are captured BEFORE any image is drawn/saved, so visualization never inflates the
	// reported latency. The image passed in stays the ORIGINAL (for drawing): results come
	// back in original coordinates.
	clientStart := time.Now()
	res, err := server.Predict(context.Background(), mgr, modelName, img, prompt)
	if err != nil {
		return err
	}
	clientMs := float64(time.Since(clientStart).Microseconds()) / 1000.0

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
	// Same EXIF handling as the server (server/limits.go decodeImage), so `run` and the API agree.
	img, err := imaging.Decode(f, imaging.AutoOrientation(true))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image %s: %w", path, err)
	}
	return img, nil
}
