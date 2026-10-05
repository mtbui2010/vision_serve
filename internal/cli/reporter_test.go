package cli

import (
	"image"
	"reflect"
	"testing"

	"visionserve/internal/models"
	"visionserve/internal/vision/preprocess"

	_ "visionserve/internal/models/clip"
	_ "visionserve/internal/models/scrfd"
)

// ResolvedPreprocess (what inspect shows, and what --image inverts) is exactly the spec the
// model's own Preprocess applies, for every architecture that reports one: legacy fields read the
// architecture's way (SCRFD's letterbox = top-left pad in 0..255 units, CLIP's default mean/std)
// and `preprocess:` blocks alike. Checked tensor-for-tensor on an odd-sized photo.
func TestResolvedPreprocessIsWhatPreprocessApplies(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 37, 23))
	for i := range img.Pix {
		img.Pix[i] = uint8((i * 53) % 251)
	}
	imagenet := models.Config{Width: 32, Height: 24, Mean: imagenetMean, Std: imagenetStd}
	block := func(s preprocess.Spec) models.Config {
		return models.Config{Width: s.Width, Height: s.Height, Preprocess: &s}
	}
	cases := map[string]struct {
		arch string
		cfg  models.Config
	}{
		"efficientnet legacy":      {"efficientnet", imagenet},
		"rf-detr legacy letterbox": {"rf-detr", models.Config{Width: 32, Height: 32, Letterbox: true, Mean: imagenetMean, Std: imagenetStd}},
		"rf-detr legacy crop":      {"rf-detr", models.Config{Width: 32, Height: 32, Crop: "center"}}, // ignored: squash
		"rt-detr block nhwc":       {"rt-detr", block(preprocess.Spec{Resize: preprocess.Letterbox, Width: 32, Height: 32, Layout: preprocess.NHWC, PadValue: 114})},
		"midas legacy":             {"midas", imagenet},
		"depth keep_aspect":        {"depth-anything-v2", models.Config{Width: 28, Height: 28, KeepAspect: true, MultipleOf: 14}},
		"scrfd legacy letterbox":   {"scrfd", models.Config{Width: 32, Height: 32, Letterbox: true, Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}}},
		"clip defaults":            {"clip", models.Config{Width: 24, Height: 24, Crop: "center"}},
	}
	for name, c := range cases {
		c.cfg.Name = "m"
		base, err := models.New(c.arch, c.cfg)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		mdl := base.(models.Model)
		rep, ok := base.(models.PreprocessReporter)
		if !ok {
			t.Fatalf("%s: %s does not report its preprocessing", name, c.arch)
		}
		spec, err := rep.ResolvedPreprocess()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want, wantMeta, err := mdl.Preprocess(img)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		got, gotMeta, err := spec.Apply(img)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !reflect.DeepEqual(got.Shape, want.Shape) || !reflect.DeepEqual(got.Data, want.Data) || gotMeta != wantMeta {
			t.Errorf("%s: the reported spec %+v does not reproduce Preprocess (shape %v vs %v)", name, spec, got.Shape, want.Shape)
		}
	}
}
