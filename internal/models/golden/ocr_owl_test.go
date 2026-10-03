package golden

import (
	"fmt"
	"image"
	"math/rand"
	"path/filepath"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"

	_ "visionserve/internal/models/owlvit"
	_ "visionserve/internal/models/paddleocr"
)

// ---------------------------------------------------------------------------------------
// PP-OCRv4 — det x [1,3,H,W] → sigmoid_0.tmp_0 [1,1,H,W] (probabilities); rec x
// [1,3,48,W] → softmax_11.tmp_0 [1,W/8,6625] (6623 keys + blank + space).
// ---------------------------------------------------------------------------------------

func ocrGen(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	x := in["x"]
	h, w := int(x.Shape[2]), int(x.Shape[3])
	if role == "det" {
		r := rand.New(rand.NewSource(int64(h*7919 + w)))
		p := make([]float32, h*w)
		for i := range p {
			p[i] = float32(0.25 * r.Float64()) // background below the 0.3 threshold
		}
		fill := func(x0, y0, x1, y1 int, v float32) {
			for y := max(y0, 0); y < min(y1, h); y++ {
				for x := max(x0, 0); x < min(x1, w); x++ {
					p[y*w+x] = v
				}
			}
		}
		for k := 0; k < 7; k++ { // text lines
			x0, y0 := r.Intn(w), r.Intn(h)
			fill(x0, y0, x0+10+r.Intn(w/2), y0+4+r.Intn(20), float32(0.35+0.6*r.Float64()))
		}
		for k := 0; k < 12; k++ { // specks below minComponentSize
			x0, y0 := r.Intn(w), r.Intn(h)
			fill(x0, y0, x0+2, y0+3, 0.9)
		}
		fill(0, 0, 25, 9, 0.8)          // touches the top-left border
		fill(w-30, h-6, w+5, h+5, 0.31) // bottom-right border, barely above threshold
		return []engine.Tensor{engine.F32(p, 1, 1, int64(h), int64(w))}, nil
	}
	const c = 6625
	t := w / 8
	r := rand.New(rand.NewSource(seedOf(x.Data[:min(len(x.Data), 4096)])))
	probs := make([]float32, t*c)
	prev := 0
	for s := 0; s < t; s++ {
		row := probs[s*c : (s+1)*c]
		for i := range row {
			row[i] = 1e-5
		}
		var k int
		switch u := r.Float64(); {
		case u < 0.3:
			k = 0 // blank
		case u < 0.45:
			k = prev // repeat
		case u < 0.5:
			k = c - 1 // the space class (index len(keys)+1)
		default:
			k = 1 + r.Intn(c-2)
		}
		row[k] = float32(0.4 + 0.59*r.Float64())
		prev = k
	}
	return []engine.Tensor{engine.F32(probs, 1, int64(t), c)}, nil
}

func TestGoldenPaddleOCR(t *testing.T) {
	g := newGolden(t, "paddleocr")
	dir := filepath.Join(repoRoot, "models", "paddle-ocr")
	for _, c := range []struct {
		name string
		cfg  models.Config
	}{
		{"manifest", models.Config{Name: "paddle-ocr", Width: 960, Height: 960, ConfThresh: 0.3, Dir: dir,
			Files: map[string]string{"det": "det.onnx", "rec": "rec.onnx"}}},
		{"defaults-640", models.Config{Name: "paddle-ocr", Width: 640, Dir: dir,
			Files: map[string]string{"det": "det.onnx", "rec": "rec.onnx"}}},
	} {
		pm := newPipeline(t, "paddle-ocr", c.cfg)
		for _, im := range []struct {
			name string
			img  image.Image
		}{
			{"synth640x480", synthImage(640, 480, 21)},
			{"synth1200x500", synthImage(1200, 500, 22)},
			{"sample810x1080", loadSample(t)},
		} {
			r := newFakeRunner(
				map[string][]string{"det": {"x"}, "rec": {"x"}},
				map[string][]string{"det": {"sigmoid_0.tmp_0"}, "rec": {"softmax_11.tmp_0"}},
				ocrGen)
			g.add(c.name+"/"+im.name, runPipeline(t, pm, im.img, models.Prompt{}, r))
		}
	}
	g.check()
}

// ---------------------------------------------------------------------------------------
// OWLv2 (owlv2_base_patch16, 960) — query_pixel_values + query_image_features [1,3,960,960]
// → logits [1,3600,1] + pred_boxes [1,3600,4] (cxcywh over the padded square).
// ---------------------------------------------------------------------------------------

func owlGen(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	q := in["query_pixel_values"].Data
	tm := in["query_image_features"].Data
	const p = 3600
	rb := rand.New(rand.NewSource(seedOf(q[:4096], q[len(q)-4096:])))
	boxes := make([]float32, p*4)
	type cl struct{ x, y, w, h float64 }
	clusters := make([]cl, 12)
	for i := range clusters {
		clusters[i] = cl{rb.Float64(), rb.Float64(), 0.03 + 0.25*rb.Float64(), 0.03 + 0.25*rb.Float64()}
	}
	for i := 0; i < p; i++ {
		c := clusters[rb.Intn(len(clusters))]
		j := func() float64 { return 1 + 0.15*rb.NormFloat64() }
		boxes[i*4] = float32(c.x + 0.02*rb.NormFloat64())
		boxes[i*4+1] = float32(c.y + 0.02*rb.NormFloat64())
		boxes[i*4+2] = float32(c.w * j())
		boxes[i*4+3] = float32(c.h * j())
	}
	rl := rand.New(rand.NewSource(seedOf(tm[:4096], q[:64])))
	logits := randNorm(rl, p, -4, 2)
	return []engine.Tensor{engine.F32(logits, 1, p, 1), engine.F32(boxes, 1, p, 4)}, nil
}

func TestGoldenOWLv2(t *testing.T) {
	g := newGolden(t, "owlvit")
	cm := []float32{0.48145466, 0.4578275, 0.40821073}
	cs := []float32{0.26862954, 0.26130258, 0.27577711}
	tmpls := []image.Image{synthImage(64, 48, 31), synthImage(90, 120, 32), synthImage(40, 40, 33)}
	for _, c := range []struct {
		name string
		cfg  models.Config
	}{
		{"manifest", models.Config{Name: "owlv2_base_patch16", Width: 960, Height: 960, Mean: cm, Std: cs,
			ConfThresh: 0.1, MaxDet: 10, InstanceSimThreshold: 0.1, InstanceMaxTemplates: 2, InstancePatchSize: 16,
			Files: map[string]string{"model": "m.onnx"}}},
		{"defaults-max100", models.Config{Name: "owl", Width: 960, Height: 960,
			Files: map[string]string{"model": "m.onnx"}}},
	} {
		pm := newPipeline(t, "owlvit", c.cfg)
		for _, im := range []struct {
			name string
			img  image.Image
		}{
			{"synth640x480", synthImage(640, 480, 34)},
			{"synth1500x1000", synthImage(1500, 1000, 35)},
		} {
			for _, nt := range []int{0, 1, 3} {
				r := newFakeRunner(
					map[string][]string{"model": {"query_pixel_values", "query_image_features"}},
					map[string][]string{"model": {"logits", "pred_boxes"}},
					owlGen)
				g.add(fmt.Sprintf("%s/%s/t%d", c.name, im.name, nt),
					runPipeline(t, pm, im.img, models.Prompt{TemplateImages: tmpls[:nt]}, r))
			}
		}
	}
	g.check()
}
