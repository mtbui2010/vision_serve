package preprocess_test

// The MIGRATED model packages, driven through their public surface (models.New + Preprocess, or
// Infer with a runner that captures the first session input), must feed exactly what the frozen
// pre-migration code fed, on every test size and image type. golden pins a few images per model;
// this sweeps many.

import (
	"errors"
	"image"
	"path/filepath"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/preprocess"

	_ "visionserve/internal/models/classification"
	_ "visionserve/internal/models/clip"
	_ "visionserve/internal/models/depth"
	_ "visionserve/internal/models/detr"
	_ "visionserve/internal/models/efficientsam"
	_ "visionserve/internal/models/mobilesam"
	_ "visionserve/internal/models/nanosam"
	_ "visionserve/internal/models/paddleocr"
	_ "visionserve/internal/models/sam2"
	_ "visionserve/internal/models/scrfd"
)

func toCfg(name string, c legacyCfg) models.Config {
	return models.Config{Name: name, Width: c.Width, Height: c.Height, Mean: c.Mean, Std: c.Std,
		Letterbox: c.Letterbox, Crop: c.Crop, KeepAspect: c.KeepAspect, MultipleOf: c.MultipleOf}
}

func TestPlainModelsMatchFrozenCode(t *testing.T) {
	scrfdNorm := legacyCfg{Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}}
	cases := []struct {
		arch string
		cfg  legacyCfg
		old  func(image.Image, legacyCfg) (engine.Tensor, preprocess.Meta)
	}{
		{"rf-detr", legacyCfg{Width: 560, Height: 560, Mean: imagenetMean, Std: imagenetStd}, oldDETR},
		{"rf-detr", legacyCfg{Width: 384, Height: 384, Letterbox: true, Mean: imagenetMean, Std: imagenetStd}, oldDETR},
		// legacy flags rf-detr never read are still ignored
		{"rf-detr", legacyCfg{Width: 512, Height: 512, Crop: "center", Mean: imagenetMean, Std: imagenetStd}, oldDETR},
		{"rt-detr", legacyCfg{Width: 640, Height: 640, Letterbox: true, Mean: imagenetMean, Std: imagenetStd}, oldDETR},
		{"rt-detr", legacyCfg{Width: 640, Height: 480, Mean: imagenetMean, Std: imagenetStd}, oldDETR},
		{"efficientnet", legacyCfg{Width: 224, Height: 224, Mean: imagenetMean, Std: imagenetStd}, oldClassification},
		{"mobilenet-v3", legacyCfg{Width: 160, Height: 128, Letterbox: true}, oldClassification},
		{"midas", legacyCfg{Width: 256, Height: 256, Mean: imagenetMean, Std: imagenetStd}, oldDepth},
		{"midas", legacyCfg{Width: 256, Height: 256, Letterbox: true, Mean: imagenetMean, Std: imagenetStd}, oldDepth},
		{"depth-anything-v2", legacyCfg{Width: 518, Height: 518, KeepAspect: true, MultipleOf: 14, Mean: imagenetMean, Std: imagenetStd}, oldDepth},
		{"clip", legacyCfg{Width: 224, Height: 224, Crop: "center", Mean: oldCLIPMean, Std: oldCLIPStd}, oldCLIP},
		{"clip", legacyCfg{Width: 224, Height: 224}, oldCLIP},
		{"clip", legacyCfg{Width: 224, Height: 224, Letterbox: true, Std: []float32{0.3, 0.3, 0.3}}, oldCLIP},
		{"scrfd", legacyCfg{Width: 640, Height: 640, Letterbox: true, Mean: scrfdNorm.Mean, Std: scrfdNorm.Std}, oldSCRFD},
		{"scrfd", legacyCfg{Width: 640, Height: 480, Mean: scrfdNorm.Mean, Std: scrfdNorm.Std}, oldSCRFD},
		{"scrfd", legacyCfg{Width: 320, Height: 320}, oldSCRFD},
	}
	ins, _ := sweep()
	for _, c := range cases {
		c := c
		t.Run(c.arch, func(t *testing.T) {
			t.Parallel()
			b, err := models.New(c.arch, toCfg(c.arch, c.cfg))
			if err != nil {
				t.Fatal(err)
			}
			m := b.(models.Model)
			for _, in := range ins {
				got, gotMeta, err := m.Preprocess(in.img)
				if err != nil {
					t.Fatalf("%s: %v", in.name, err)
				}
				want, wantMeta := c.old(in.img, c.cfg)
				sameTensor(t, in.name, got, want)
				sameMeta(t, in.name, gotMeta, wantMeta)
			}
		})
	}
}

var errCaptured = errors.New("captured")

// captureRunner records the inputs of the first session call and aborts the pipeline.
type captureRunner struct {
	role   string
	inputs map[string]engine.Tensor
}

func (r *captureRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if r.inputs == nil {
		r.role, r.inputs = role, in
	}
	return nil, errCaptured
}
func (r *captureRunner) InputNames(string) []string  { return nil }
func (r *captureRunner) OutputNames(string) []string { return nil }

// firstInput runs the pipeline model on img and returns its first session input.
func firstInput(t *testing.T, pm models.PipelineModel, img image.Image, p models.Prompt) engine.Tensor {
	t.Helper()
	r := &captureRunner{}
	if _, err := pm.Infer(img, p, r); !errors.Is(err, errCaptured) {
		t.Fatalf("Infer: %v (want the capture abort)", err)
	}
	if len(r.inputs) != 1 {
		t.Fatalf("first call (%s) has %d inputs, want 1", r.role, len(r.inputs))
	}
	for _, v := range r.inputs {
		return v
	}
	return engine.Tensor{}
}

// paddleDir holds the committed PaddleOCR charset the model loads at construction.
var paddleDir = filepath.Join("..", "..", "..", "models", "paddle-ocr")

func TestPipelineEncodersMatchFrozenCode(t *testing.T) {
	files := map[string]string{"encoder": "enc.onnx", "decoder": "dec.onnx", "det": "det.onnx", "rec": "rec.onnx"}
	box := func(img image.Image) models.Prompt {
		b := img.Bounds()
		return models.Prompt{Boxes: [][4]float64{{0, 0, float64(b.Dx()), float64(b.Dy())}}}
	}
	cases := []struct {
		arch string
		cfg  models.Config
		big  bool
		old  func(image.Image) engine.Tensor
	}{
		{"mobile-sam", models.Config{Width: 1024, Height: 1024, Layout: "NHWC"}, true,
			func(img image.Image) engine.Tensor { t, _ := oldMobileSAM(img); return t }},
		// MobileSAM built from a composite's manifest (grasp: 1024 NCHW) ignores that input block.
		{"mobile-sam", models.Config{Width: 800, Height: 800, Layout: "NCHW", Letterbox: true, Mean: imagenetMean, Std: imagenetStd}, true,
			func(img image.Image) engine.Tensor { t, _ := oldMobileSAM(img); return t }},
		{"nano-sam", models.Config{Width: 1024, Height: 1024}, true,
			func(img image.Image) engine.Tensor { t, _ := oldNanoSAM(img); return t }},
		{"sam2", models.Config{Width: 1024, Height: 1024}, true,
			func(img image.Image) engine.Tensor { t, _, _ := oldSAM2(img); return t }},
		{"efficient-sam", models.Config{Width: 1024, Height: 1024}, false, oldEfficientSAM},
		{"paddle-ocr", models.Config{Width: 960, Height: 960, Dir: paddleDir}, false,
			func(img image.Image) engine.Tensor { t, _, _, _ := oldPaddleDet(img, 960); return t }},
		{"paddle-ocr", models.Config{Width: 320, Height: 320, Dir: paddleDir}, false,
			func(img image.Image) engine.Tensor { t, _, _, _ := oldPaddleDet(img, 320); return t }},
	}
	small, big := sweep()
	for _, c := range cases {
		c := c
		t.Run(c.arch, func(t *testing.T) {
			t.Parallel()
			c.cfg.Name, c.cfg.Files = c.arch, files
			b, err := models.New(c.arch, c.cfg)
			if err != nil {
				t.Fatal(err)
			}
			pm := b.(models.PipelineModel)
			ins := small
			if c.big {
				ins = big
			}
			for _, in := range ins {
				sameTensor(t, in.name, firstInput(t, pm, in.img, box(in.img)), c.old(in.img))
			}
		})
	}
}
