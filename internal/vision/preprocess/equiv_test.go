package preprocess_test

import (
	"image"
	"testing"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
	"visionserve/internal/vision/preprocess"
)

var (
	imagenetMean = []float32{0.485, 0.456, 0.406}
	imagenetStd  = []float32{0.229, 0.224, 0.225}
)

// specCase pairs a Spec with the frozen code path it replaces.
type specCase struct {
	name string
	spec preprocess.Spec
	old  func(img image.Image) (engine.Tensor, preprocess.Meta)
	big  bool // 1024×1024 output: run on bigOutSizes only
}

func specCases() []specCase {
	sq := func(w, h int, mean, std []float32) preprocess.Spec {
		return preprocess.Spec{Resize: preprocess.Squash, Width: w, Height: h, Mean: mean, Std: std}
	}
	return []specCase{
		{"detr-squash-560", sq(560, 560, imagenetMean, imagenetStd), func(img image.Image) (engine.Tensor, preprocess.Meta) {
			return oldDETR(img, legacyCfg{Width: 560, Height: 560, Mean: imagenetMean, Std: imagenetStd})
		}, false},
		{"detr-squash-512x384-nonorm", sq(512, 384, nil, nil), func(img image.Image) (engine.Tensor, preprocess.Meta) {
			return oldDETR(img, legacyCfg{Width: 512, Height: 384})
		}, false},
		{"detr-letterbox-384", preprocess.Spec{Resize: preprocess.Letterbox, Width: 384, Height: 384, Mean: imagenetMean, Std: imagenetStd},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldDETR(img, legacyCfg{Width: 384, Height: 384, Letterbox: true, Mean: imagenetMean, Std: imagenetStd})
			}, false},
		{"detr-letterbox-640x480", preprocess.Spec{Resize: preprocess.Letterbox, Width: 640, Height: 480, Mean: imagenetMean, Std: imagenetStd},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldDETR(img, legacyCfg{Width: 640, Height: 480, Letterbox: true, Mean: imagenetMean, Std: imagenetStd})
			}, false},
		{"classification-224", sq(224, 224, imagenetMean, imagenetStd), func(img image.Image) (engine.Tensor, preprocess.Meta) {
			return oldClassification(img, legacyCfg{Width: 224, Height: 224, Mean: imagenetMean, Std: imagenetStd})
		}, false},
		{"midas-256", sq(256, 256, imagenetMean, imagenetStd), func(img image.Image) (engine.Tensor, preprocess.Meta) {
			return oldDepth(img, legacyCfg{Width: 256, Height: 256, Mean: imagenetMean, Std: imagenetStd})
		}, false},
		{"da2-keep-aspect-518-14", preprocess.Spec{Resize: preprocess.KeepAspect, Width: 518, Height: 518, MultipleOf: 14, Mean: imagenetMean, Std: imagenetStd},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldDepth(img, legacyCfg{Width: 518, Height: 518, KeepAspect: true, MultipleOf: 14, Mean: imagenetMean, Std: imagenetStd})
			}, false},
		{"clip-center-crop-224", preprocess.Spec{Resize: preprocess.CenterCrop, Width: 224, Height: 224, Mean: oldCLIPMean, Std: oldCLIPStd},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldCLIP(img, legacyCfg{Width: 224, Height: 224, Crop: "center"})
			}, false},
		{"clip-center-crop-336x224", preprocess.Spec{Resize: preprocess.CenterCrop, Width: 336, Height: 224, Mean: oldCLIPMean, Std: oldCLIPStd},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldCLIP(img, legacyCfg{Width: 336, Height: 224, Crop: "center", Mean: oldCLIPMean, Std: oldCLIPStd})
			}, false},
		{"clip-squash-defaults", sq(224, 224, oldCLIPMean, oldCLIPStd), func(img image.Image) (engine.Tensor, preprocess.Meta) {
			return oldCLIP(img, legacyCfg{Width: 224, Height: 224})
		}, false},
		{"scrfd-640", preprocess.Spec{Resize: preprocess.TopLeftPad, Width: 640, Height: 640, NoRescale: true,
			Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldSCRFD(img, legacyCfg{Width: 640, Height: 640, Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}})
			}, false},
		{"scrfd-640x480", preprocess.Spec{Resize: preprocess.TopLeftPad, Width: 640, Height: 480, NoRescale: true,
			Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldSCRFD(img, legacyCfg{Width: 640, Height: 480, Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}})
			}, false},
		{"scrfd-no-normalize", preprocess.Spec{Resize: preprocess.TopLeftPad, Width: 320, Height: 320},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldSCRFD(img, legacyCfg{Width: 320, Height: 320})
			}, false},
		{"scrfd-mean-only", preprocess.Spec{Resize: preprocess.TopLeftPad, Width: 320, Height: 320, NoRescale: true, Mean: []float32{127.5}, Legacy: true},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				return oldSCRFD(img, legacyCfg{Width: 320, Height: 320, Mean: []float32{127.5}})
			}, false},
		{"mobilesam-long-side-1024", preprocess.Spec{Resize: preprocess.LongSide, Width: 1024, Height: 1024, NoRescale: true, Layout: preprocess.HWC},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				t, scale := oldMobileSAM(img)
				b := img.Bounds()
				return t, preprocess.Meta{OrigWidth: b.Dx(), OrigHeight: b.Dy(), ScaleX: scale, ScaleY: scale}
			}, true},
		{"nanosam-long-side-pad-1024", preprocess.Spec{Resize: preprocess.LongSidePad, Width: 1024, Height: 1024, Mean: imagenetMean, Std: imagenetStd},
			func(img image.Image) (engine.Tensor, preprocess.Meta) {
				t, _ := oldNanoSAM(img)
				b := img.Bounds()
				_, _, scale := preprocess.LongSideSize(b.Dx(), b.Dy(), 1024, 1024, false)
				return t, preprocess.Meta{OrigWidth: b.Dx(), OrigHeight: b.Dy(), ScaleX: scale, ScaleY: scale}
			}, true},
		{"sam2-squash-1024", sq(1024, 1024, imagenetMean, imagenetStd), func(img image.Image) (engine.Tensor, preprocess.Meta) {
			t, _, _ := oldSAM2(img)
			b := img.Bounds()
			return t, preprocess.Meta{OrigWidth: b.Dx(), OrigHeight: b.Dy(), ScaleX: 1024 / float64(b.Dx()), ScaleY: 1024 / float64(b.Dy())}
		}, true},
		{"efficientsam-none", preprocess.Spec{Resize: preprocess.None}, func(img image.Image) (engine.Tensor, preprocess.Meta) {
			b := img.Bounds()
			return oldEfficientSAM(img), preprocess.Meta{OrigWidth: b.Dx(), OrigHeight: b.Dy(), ScaleX: 1, ScaleY: 1}
		}, false},
		{"paddle-det-960", preprocess.Spec{Resize: preprocess.LongSidePad, Width: 960, Height: 960, MultipleOf: 32, NoUpscale: true,
			Mean: imagenetMean, Std: imagenetStd}, func(img image.Image) (engine.Tensor, preprocess.Meta) {
			t, m, _, _ := oldPaddleDet(img, 960)
			return t, m
		}, false},
		{"paddle-det-320", preprocess.Spec{Resize: preprocess.LongSidePad, Width: 320, Height: 320, MultipleOf: 32, NoUpscale: true,
			Mean: imagenetMean, Std: imagenetStd}, func(img image.Image) (engine.Tensor, preprocess.Meta) {
			t, m, _, _ := oldPaddleDet(img, 320)
			return t, m
		}, false},
	}
}

// TestSpecMatchesFrozenCode: every Spec reproduces the code path it replaced, bit for bit, on
// every test size and image type.
func TestSpecMatchesFrozenCode(t *testing.T) {
	small, big := inputs(sizes), inputs(bigOutSizes)
	if testing.Short() {
		small, big = inputs(typeSizes), inputs([][2]int{{3, 5}, {640, 480}})
	}
	for _, c := range specCases() {
		c := c
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			ins := small
			if c.big {
				ins = big
			}
			for _, in := range ins {
				got, gotMeta, err := c.spec.Apply(in.img)
				if err != nil {
					t.Fatalf("%s: %v", in.name, err)
				}
				want, wantMeta := c.old(in.img)
				sameTensor(t, in.name, got, want)
				sameMeta(t, in.name, gotMeta, wantMeta)
			}
		})
	}
}

// TestTensorMatchesImageToCHWFloat: Spec.Tensor (the shared normalise helper) is the old
// imageproc.ImageToCHWFloat for every image type, with and without mean/std.
func TestTensorMatchesImageToCHWFloat(t *testing.T) {
	norms := []struct {
		name      string
		mean, std []float32
	}{
		{"none", nil, nil}, {"imagenet", imagenetMean, imagenetStd}, {"half", []float32{0.5, 0.5, 0.5}, []float32{0.5, 0.5, 0.5}},
		{"partial", []float32{0.3}, []float32{0.2, 0, 0.7, 9}},
	}
	for _, in := range inputs(typeSizes) {
		for _, n := range norms {
			got := preprocess.Spec{Mean: n.mean, Std: n.std}.Tensor(in.img)
			sameTensor(t, in.name+"/"+n.name, got, oldImageToCHWFloat(in.img, n.mean, n.std))
		}
	}
}

// TestPaddleRecNormalizeMatches: the rec crop's normalisation loop is Spec.Tensor with 0.5/0.5.
func TestPaddleRecNormalizeMatches(t *testing.T) {
	for _, in := range inputs([][2]int{{17, 31}, {400, 37}, {1, 1}}) {
		resized := imaging.Resize(in.img, 123, 48, imaging.Linear)
		got := preprocess.Spec{Mean: []float32{0.5, 0.5, 0.5}, Std: []float32{0.5, 0.5, 0.5}}.Tensor(resized)
		sameTensor(t, in.name, got, oldPaddleRecNormalize(resized, 123, 48))
	}
}

// The prompt scales the SAM models keep computing themselves agree with the Meta Apply returns
// (mobilesam: the identical float64; nanosam: the float32 of it), and the float32 scale NanoSAM
// used gives the same rounded size as the float64 one (checked exhaustively up to 12000 px when
// LongSideSize replaced it; spot-checked here).
func TestLongSideScaleAgreesWithSAMScales(t *testing.T) {
	for _, s := range sizes {
		w, h := s[0], s[1]
		_, _, scale := preprocess.LongSideSize(w, h, 1024, 1024, false)
		if want := float64(1024) / float64(max(w, h)); scale != want {
			t.Fatalf("%dx%d: LongSideSize scale %v, mobilesam %v", w, h, scale, want)
		}
		if want := float32(1024) / float32(max(w, h)); float32(scale) != want {
			t.Fatalf("%dx%d: float32(scale) %v, nanosam %v", w, h, float32(scale), want)
		}
	}
	for long := 1; long <= 3000; long++ {
		s32 := float32(1024) / float32(long)
		for short := 1; short <= long; short++ {
			nw, _, _ := preprocess.LongSideSize(short, long, 1024, 1024, false)
			if old := int(roundHalfAway(float64(short) * float64(s32))); max(old, 1) != nw {
				t.Fatalf("%dx%d: %d, nanosam float32 arithmetic gives %d", short, long, nw, old)
			}
		}
	}
}

func roundHalfAway(x float64) float64 {
	if x < 0 {
		return -roundHalfAway(-x)
	}
	f := float64(int64(x))
	if x-f >= 0.5 {
		return f + 1
	}
	return f
}
