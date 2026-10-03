package golden

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"

	_ "visionserve/internal/models/efficientsam"
	_ "visionserve/internal/models/nanosam"
	_ "visionserve/internal/models/sam2"
)

func samImages(t *testing.T) []struct {
	name string
	img  image.Image
} {
	return []struct {
		name string
		img  image.Image
	}{
		{"synth640x480", synthImage(640, 480, 11)},
		{"synth300x500", synthImage(300, 500, 12)},
		{"sample810x1080", loadSample(t)},
	}
}

// samPrompts are the prompt variants every SAM model is pinned on (ORIGINAL coords).
func samPrompts() []struct {
	name string
	p    models.Prompt
} {
	return []struct {
		name string
		p    models.Prompt
	}{
		{"boxes", models.Prompt{Boxes: [][4]float64{{40, 30, 180, 150}, {150.5, 200.25, 400, 600}}}},
		{"points", models.Prompt{Points: []models.Point{{X: 120, Y: 90, Label: 1}, {X: 260.5, Y: 180, Label: 0}, {X: 30, Y: 300, Label: 1}}}},
		{"boxes+points", models.Prompt{Boxes: [][4]float64{{10, 20, 100, 80}}, Points: []models.Point{{X: 50, Y: 60, Label: 1}}}},
		{"text", models.Prompt{Text: "cat. remote."}},
		{"empty", models.Prompt{}},
	}
}

// embedding returns a deterministic constant-ish tensor of the given shape (its content is
// irrelevant to the decoders' postprocess; its digest pins that the model forwards it).
func embedding(seed float32, shape ...int64) engine.Tensor {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	d := make([]float32, n)
	for i := range d {
		d[i] = seed + float32(i%97)*1e-3
	}
	return engine.F32(d, shape...)
}

// ---------------------------------------------------------------------------------------
// MobileSAM — encoder input_image [H,W,3] → image_embeddings [1,256,64,64]; decoder
// (single export) masks [1,1,H,W] at orig_im_size + iou_predictions [1,1] + low_res_masks
// [1,1,256,256]; the multi export has 4 channels and iou [1,4].
// ---------------------------------------------------------------------------------------

func mobileSAMGen(channels int, quant float32) genFunc {
	return func(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
		if role == "encoder" {
			x := in["input_image"]
			return []engine.Tensor{embedding(float32(len(x.Data)%7), 1, 256, 64, 64)}, nil
		}
		size := in["orig_im_size"].Data
		h, w := int(size[0]), int(size[1])
		pc := append([]float32(nil), in["point_coords"].Data...)
		if quant > 0 { // AMG: neighbouring grid points decode to the SAME mask (duplicates for NMS)
			for i := range pc {
				pc[i] = float32(math.Floor(float64(pc[i] / quant)))
			}
		}
		r := rand.New(rand.NewSource(seedOf(pc, in["point_labels"].Data)))
		masks := make([]float32, channels*h*w)
		iou := make([]float32, channels)
		for k := 0; k < channels; k++ {
			randBlob(r).render(masks[k*h*w:(k+1)*h*w], w, h)
			iou[k] = float32(0.80 + 0.01*float64(r.Intn(21)))
		}
		low := make([]float32, channels*256*256)
		return []engine.Tensor{
			engine.F32(masks, 1, int64(channels), int64(h), int64(w)),
			engine.F32(iou, 1, int64(channels)),
			engine.F32(low, 1, int64(channels), 256, 256),
		}, nil
	}
}

func mobileSAMRunner(channels int, quant float32) *fakeRunner {
	return newFakeRunner(
		map[string][]string{"encoder": {"input_image"}, "decoder": {"image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"}},
		map[string][]string{"encoder": {"image_embeddings"}, "decoder": {"masks", "iou_predictions", "low_res_masks"}},
		mobileSAMGen(channels, quant))
}

type maskInferer interface {
	InferMasks(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Mask, []mobilesam.MaskBitmap, error)
}

func TestGoldenMobileSAM(t *testing.T) {
	g := newGolden(t, "mobilesam")
	cfg := models.Config{Name: "mobile-sam", Width: 1024, Height: 1024,
		Files: map[string]string{"encoder": "enc.onnx", "decoder": "dec.onnx"}}
	for _, ch := range []int{1, 4} {
		pm := newPipeline(t, "mobile-sam", cfg)
		for _, im := range samImages(t) {
			for _, p := range samPrompts() {
				if p.name == "empty" {
					continue // AMG cases below
				}
				g.add(fmt.Sprintf("ch%d/%s/%s", ch, im.name, p.name), runPipeline(t, pm, im.img, p.p, mobileSAMRunner(ch, 0)))
			}
		}
	}
	// Automatic Mask Generator: work-frame filter pass + full-res final pass (640×480,
	// 810×1080), a frame already ≤256 px (no final pass), and a per-request grid override.
	pm := newPipeline(t, "mobile-sam", cfg)
	for _, c := range []struct {
		name string
		img  image.Image
		grid int
		ch   int
	}{
		{"amg/synth640x480", synthImage(640, 480, 11), 0, 1},
		{"amg/synth200x150", synthImage(200, 150, 13), 0, 1},
		{"amg/sample-grid6-ch4", loadSample(t), 6, 4},
	} {
		r := mobileSAMRunner(c.ch, 192)
		rec := runPipeline(t, pm, c.img, models.Prompt{GridSize: c.grid}, r)
		g.add(c.name, rec)
	}
	// InferMasks exposes the raw bitmaps (consumed by grasp/background).
	mi, ok := pm.(maskInferer)
	if !ok {
		t.Fatal("mobile-sam does not implement InferMasks")
	}
	for _, c := range []struct {
		name string
		img  image.Image
		p    models.Prompt
		q    float32
	}{
		{"infermasks/boxes", synthImage(640, 480, 11), samPrompts()[0].p, 0},
		{"infermasks/amg", synthImage(320, 240, 14), models.Prompt{GridSize: 8}, 192},
	} {
		r := mobileSAMRunner(1, c.q)
		masks, bms, err := mi.InferMasks(c.img, c.p, r)
		rec := pipelineRec{}
		rec.Calls, rec.CallInputs = r.callsRecord()
		if err != nil {
			rec.Err = err.Error()
		} else {
			rec.Result = &models.Result{Masks: masks}
			for _, b := range bms {
				rec.Bitmaps = append(rec.Bitmaps, recBitmap(b.Data, b.W, b.H, b.BBox, b.Conf))
			}
		}
		g.add(c.name, rec)
	}
	// Segment is the reusable box→mask core (grounded-sam, hybrid).
	for _, im := range samImages(t) {
		r := mobileSAMRunner(1, 0)
		dec := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run("decoder", in) }
		enc := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run("encoder", in) }
		masks, err := mobilesam.Segment(im.img, [][4]float64{{40, 30, 180, 150}, {0, 0, 50, 40}}, enc, dec,
			"input_image", []string{"masks", "iou_predictions", "low_res_masks"})
		rec := pipelineRec{}
		rec.Calls, rec.CallInputs = r.callsRecord()
		if err != nil {
			rec.Err = err.Error()
		} else {
			rec.Result = &models.Result{Masks: masks}
		}
		g.add("segment/"+im.name, rec)
	}
	g.check()
}

// ---------------------------------------------------------------------------------------
// EfficientSAM — encoder batched_images [1,3,H,W] → image_embeddings [1,256,64,64];
// decoder output_masks [1,1,3,H,W] at orig_im_size (int64) + iou_predictions [1,1,3] +
// a low-res [1,3,256,256] tensor ("onnx::Shape_1830").
// ---------------------------------------------------------------------------------------

func efficientSAMGen(div int) genFunc {
	return func(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
		if role == "encoder" {
			return []engine.Tensor{embedding(0.25, 1, 256, 64, 64)}, nil
		}
		size := in["orig_im_size"].DataI64
		h, w := int(size[0])/div, int(size[1])/div
		r := rand.New(rand.NewSource(seedOf(in["batched_point_coords"].Data, in["batched_point_labels"].Data)))
		masks := make([]float32, 3*h*w)
		iou := make([]float32, 3)
		for k := 0; k < 3; k++ {
			randBlob(r).render(masks[k*h*w:(k+1)*h*w], w, h)
			iou[k] = float32(r.Float64())
		}
		return []engine.Tensor{
			engine.F32(masks, 1, 1, 3, int64(h), int64(w)),
			engine.F32(iou, 1, 1, 3),
			engine.F32(make([]float32, 3*256*256), 1, 3, 256, 256),
		}, nil
	}
}

func TestGoldenEfficientSAM(t *testing.T) {
	g := newGolden(t, "efficientsam")
	cfg := models.Config{Name: "efficient-sam", Width: 1024, Height: 1024,
		Files: map[string]string{"encoder": "enc.onnx", "decoder": "dec.onnx"}}
	pm := newPipeline(t, "efficient-sam", cfg)
	for _, div := range []int{1, 2} {
		for _, im := range samImages(t) {
			for _, p := range samPrompts() {
				r := newFakeRunner(
					map[string][]string{"encoder": {"batched_images"}, "decoder": {"image_embeddings", "batched_point_coords", "batched_point_labels", "orig_im_size"}},
					map[string][]string{"encoder": {"image_embeddings"}, "decoder": {"output_masks", "iou_predictions", "onnx::Shape_1830"}},
					efficientSAMGen(div))
				g.add(fmt.Sprintf("div%d/%s/%s", div, im.name, p.name), runPipeline(t, pm, im.img, p.p, r))
			}
		}
	}
	g.check()
}

// ---------------------------------------------------------------------------------------
// NanoSAM — encoder image [1,3,1024,1024] → image_embeddings; decoder (no orig_im_size)
// iou_predictions [1,4] + low_res_masks [1,4,256,256].
// ---------------------------------------------------------------------------------------

func nanoSAMGen(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role == "encoder" {
		return []engine.Tensor{embedding(0.5, 1, 256, 64, 64)}, nil
	}
	r := rand.New(rand.NewSource(seedOf(in["point_coords"].Data, in["point_labels"].Data)))
	const m = 4
	masks := make([]float32, m*256*256)
	iou := make([]float32, m)
	for k := 0; k < m; k++ {
		randBlob(r).render(masks[k*256*256:(k+1)*256*256], 256, 256)
		iou[k] = float32(r.Float64())
	}
	return []engine.Tensor{engine.F32(iou, 1, m), engine.F32(masks, 1, m, 256, 256)}, nil
}

func TestGoldenNanoSAM(t *testing.T) {
	g := newGolden(t, "nanosam")
	cfg := models.Config{Name: "nano-sam", Width: 1024, Height: 1024,
		Files: map[string]string{"encoder": "enc.onnx", "decoder": "dec.onnx"}}
	pm := newPipeline(t, "nano-sam", cfg)
	for _, im := range samImages(t) {
		for _, p := range samPrompts() {
			r := newFakeRunner(
				map[string][]string{"encoder": {"image"}, "decoder": {"image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input"}},
				map[string][]string{"encoder": {"image_embeddings"}, "decoder": {"iou_predictions", "low_res_masks"}},
				nanoSAMGen)
			g.add(im.name+"/"+p.name, runPipeline(t, pm, im.img, p.p, r))
		}
	}
	g.check()
}

// ---------------------------------------------------------------------------------------
// SAM2 (SharpAI/sam2-hiera-tiny-onnx) — encoder image [1,3,1024,1024] → high_res_feats_0
// [1,32,256,256], high_res_feats_1 [1,64,128,128], image_embed [1,256,64,64]; decoder masks
// [1,3,256,256] (low-res, clipped ±32) + iou_predictions [1,3].
// ---------------------------------------------------------------------------------------

func sam2Gen(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role == "encoder" {
		return []engine.Tensor{
			embedding(1, 1, 32, 256, 256),
			embedding(2, 1, 64, 128, 128),
			embedding(3, 1, 256, 64, 64),
		}, nil
	}
	r := rand.New(rand.NewSource(seedOf(in["point_coords"].Data, in["point_labels"].Data)))
	masks := make([]float32, 3*256*256)
	iou := make([]float32, 3)
	for k := 0; k < 3; k++ {
		randBlob(r).render(masks[k*256*256:(k+1)*256*256], 256, 256)
		iou[k] = float32(r.Float64())
	}
	return []engine.Tensor{engine.F32(masks, 1, 3, 256, 256), engine.F32(iou, 1, 3)}, nil
}

func TestGoldenSAM2(t *testing.T) {
	g := newGolden(t, "sam2")
	cfg := models.Config{Name: "sam2", Width: 1024, Height: 1024,
		Files: map[string]string{"encoder": "enc.onnx", "decoder": "dec.onnx"}}
	pm := newPipeline(t, "sam2", cfg)
	for _, named := range []bool{true, false} {
		for _, im := range samImages(t) {
			for _, p := range samPrompts() {
				encOut := []string{"high_res_feats_0", "high_res_feats_1", "image_embed"}
				if !named {
					encOut = nil // positional fallback
				}
				r := newFakeRunner(
					map[string][]string{"encoder": {"image"}, "decoder": {"image_embed", "high_res_feats_0", "high_res_feats_1", "point_coords", "point_labels", "mask_input", "has_mask_input"}},
					map[string][]string{"encoder": encOut, "decoder": {"masks", "iou_predictions"}},
					sam2Gen)
				g.add(fmt.Sprintf("named=%v/%s/%s", named, im.name, p.name), runPipeline(t, pm, im.img, p.p, r))
			}
		}
	}
	g.check()
}
