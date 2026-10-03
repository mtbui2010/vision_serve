package golden

import (
	"image"
	"math/rand"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"

	_ "visionserve/internal/models/classification"
	_ "visionserve/internal/models/clip"
	_ "visionserve/internal/models/depth"
	_ "visionserve/internal/models/detr" // rf-detr + rt-detr
	_ "visionserve/internal/models/scrfd"
)

// testImages are the inputs every plain-model preprocess is pinned on.
func testImages(t *testing.T) []struct {
	name string
	img  image.Image
} {
	return []struct {
		name string
		img  image.Image
	}{
		{"synth640x480", synthImage(640, 480, 1)},
		{"synth97x173", synthImage(97, 173, 2)},
		{"sample810x1080", loadSample(t)},
	}
}

// ---------------------------------------------------------------------------------------
// RF-DETR / RT-DETR (NMS-free DETR decoders)
// ---------------------------------------------------------------------------------------

// detrOuts builds Q-query DETR outputs: logits [1,Q,C] (pre-sigmoid) and boxes [1,Q,4]
// (cxcywh or xyxy in [0,1]); some boxes cross the image border to exercise clamping.
func detrOuts(seed int64, q, c int, xyxy bool) (logits, boxes engine.Tensor) {
	r := rand.New(rand.NewSource(seed))
	l := randNorm(r, q*c, -8, 2.5)
	b := make([]float32, q*4)
	for i := 0; i < q; i++ {
		cx, cy := r.Float64()*1.1-0.05, r.Float64()*1.1-0.05
		w, h := 0.02+0.7*r.Float64(), 0.02+0.7*r.Float64()
		if xyxy {
			b[i*4], b[i*4+1], b[i*4+2], b[i*4+3] = float32(cx-w/2), float32(cy-h/2), float32(cx+w/2), float32(cy+h/2)
		} else {
			b[i*4], b[i*4+1], b[i*4+2], b[i*4+3] = float32(cx), float32(cy), float32(w), float32(h)
		}
	}
	return engine.F32(l, 1, int64(q), int64(c)), engine.F32(b, 1, int64(q), 4)
}

func TestGoldenRFDETR(t *testing.T) {
	t.Parallel()
	g := newGolden(t, "rfdetr")
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	coco91 := readLabels(t, "models/rf-detr/coco91.txt")
	cfgs := []struct {
		name string
		cfg  models.Config
	}{
		// models/rf-detr/manifest.yaml (squash 560) and models/rf-detr-nano (letterbox 384).
		{"base560", models.Config{Name: "rf-detr", Width: 560, Height: 560, Mean: mean, Std: std,
			BoxFormat: "cxcywh", ConfThresh: 0.5, MaxDet: 300, Labels: coco91}},
		{"nano384lb", models.Config{Name: "rf-detr-nano", Width: 384, Height: 384, Letterbox: true, Mean: mean, Std: std,
			BoxFormat: "cxcywh", ConfThresh: 0.5, MaxDet: 300, Labels: coco91}},
		{"xyxy512max7", models.Config{Name: "x", Width: 512, Height: 512, Mean: mean, Std: std,
			BoxFormat: "xyxy", ConfThresh: 0.3, MaxDet: 7, Labels: []string{"a", "b", "c"}}},
		{"defaultfmt", models.Config{Name: "y", Width: 512, Height: 384, ConfThresh: 0.6}},
	}
	// Real attention output of models/rfdetr-small (cross_attn_weights [3,1,16,300,1024]).
	attn := engine.F32(make([]float32, 3*16*300*1024), 3, 1, 16, 300, 1024)
	for _, c := range cfgs {
		m := newModel(t, "rf-detr", c.cfg)
		for ii, im := range testImages(t) {
			pre, meta := doPre(m, im.img)
			g.add(c.name+"/pre/"+im.name, pre)
			lg, bx := detrOuts(int64(100+ii), 300, 91, c.cfg.BoxFormat == "xyxy")
			// rf-detr-base-real: [logits, boxes]; rf-detr-nano: [pred_boxes, pred_logits];
			// rfdetr-small: [dets, labels, cross_attn_weights].
			g.add(c.name+"/post-lb/"+im.name, doPost(m, []engine.Tensor{lg, bx}, meta))
			g.add(c.name+"/post-bl/"+im.name, doPost(m, []engine.Tensor{bx, lg}, meta))
			g.add(c.name+"/post-bla/"+im.name, doPost(m, []engine.Tensor{bx, lg, attn}, meta))
		}
		meta := models.PreprocessMeta{OrigWidth: 200, OrigHeight: 100, ScaleX: 2.8, ScaleY: 5.6}
		lg, bx := detrOuts(7, 300, 91, false)
		g.add(c.name+"/err-one", doPost(m, []engine.Tensor{lg}, meta))
		g.add(c.name+"/err-nobox", doPost(m, []engine.Tensor{lg, lg}, meta))
		lg2, _ := detrOuts(8, 200, 91, false)
		g.add(c.name+"/err-q", doPost(m, []engine.Tensor{lg2, bx}, meta))
	}
	g.check()
}

func TestGoldenRTDETR(t *testing.T) {
	t.Parallel()
	g := newGolden(t, "rtdetr")
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	coco80 := readLabels(t, "models/rt-detr/coco80.txt")
	cfgs := []struct {
		name string
		cfg  models.Config
	}{
		// models/rt-detr/manifest.yaml (letterbox 640, COCO-80) + a squash variant.
		{"lb640", models.Config{Name: "rt-detr", Width: 640, Height: 640, Letterbox: true, Mean: mean, Std: std,
			BoxFormat: "cxcywh", ConfThresh: 0.5, MaxDet: 300, Labels: coco80}},
		{"sq640max5", models.Config{Name: "rt", Width: 640, Height: 480, Mean: mean, Std: std,
			BoxFormat: "xyxy", ConfThresh: 0.4, MaxDet: 5, Labels: coco80[:3]}},
	}
	attn := engine.F32(make([]float32, 2*300*8), 2, 300, 8)
	for _, c := range cfgs {
		m := newModel(t, "rt-detr", c.cfg)
		for ii, im := range testImages(t) {
			pre, meta := doPre(m, im.img)
			g.add(c.name+"/pre/"+im.name, pre)
			// onnx-community/RT-DETR-l-hf: pred_logits [1,300,80], pred_boxes [1,300,4].
			lg, bx := detrOuts(int64(200+ii), 300, 80, c.cfg.BoxFormat == "xyxy")
			g.add(c.name+"/post-lb/"+im.name, doPost(m, []engine.Tensor{lg, bx}, meta))
			g.add(c.name+"/post-bl/"+im.name, doPost(m, []engine.Tensor{bx, lg}, meta))
			// RT-DETR is stricter than RF-DETR: exactly two outputs.
			g.add(c.name+"/err-three/"+im.name, doPost(m, []engine.Tensor{bx, lg, attn}, meta))
		}
		meta := models.PreprocessMeta{OrigWidth: 200, OrigHeight: 100, ScaleX: 3.2, ScaleY: 3.2, PadY: 160}
		lg, bx := detrOuts(9, 300, 80, false)
		g.add(c.name+"/err-one", doPost(m, []engine.Tensor{lg}, meta))
		g.add(c.name+"/err-nobox", doPost(m, []engine.Tensor{lg, lg}, meta))
		lg2, _ := detrOuts(10, 100, 80, false)
		g.add(c.name+"/err-q", doPost(m, []engine.Tensor{lg2, bx}, meta))
	}
	g.check()
}

// ---------------------------------------------------------------------------------------
// SCRFD (anchor-based, NMS)
// ---------------------------------------------------------------------------------------

// scrfdOuts builds the 9 det_10g.onnx outputs for a W×H input, in the export's order
// (score_8/16/32 [N,1], bbox_8/16/32 [N,4], kps_8/16/32 [N,10]; 2-D, no batch dim). Scores
// are probabilities (in-graph Sigmoid): mostly background, plus clusters of strong anchors
// around a few "faces" so NMS has thousands of overlapping candidates.
func scrfdOuts(seed int64, w, h int, batched bool) []engine.Tensor {
	r := rand.New(rand.NewSource(seed))
	type face struct{ x, y, s float64 }
	faces := make([]face, 25)
	for i := range faces {
		faces[i] = face{r.Float64() * float64(w), r.Float64() * float64(h), 8 + 70*r.Float64()}
	}
	var scores, boxes, kps []engine.Tensor
	for _, s := range []int{8, 16, 32} {
		gw, gh := w/s, h/s
		n := gw * gh * 2
		sc := make([]float32, n)
		bb := make([]float32, n*4)
		kp := randUniform(r, n*10, -2, 2)
		for k := 0; k < n; k++ {
			row := k / (gw * 2)
			col := (k % (gw * 2)) / 2
			cx, cy := float64(col*s), float64(row*s)
			p := r.Float64() * r.Float64() * 0.52
			l, tp, rt, b := 0.5+3*r.Float64(), 0.5+3*r.Float64(), 0.5+3*r.Float64(), 0.5+3*r.Float64()
			for _, f := range faces {
				dx, dy := cx-f.x, cy-f.y
				if dx*dx+dy*dy < f.s*f.s && f.s/float64(s) > 0.6 && f.s/float64(s) < 6 {
					p = 0.5 + 0.49*r.Float64()
					ss := f.s / float64(s)
					l = (cx-(f.x-f.s))/float64(s) + 0.3*r.NormFloat64()
					tp = (cy-(f.y-f.s))/float64(s) + 0.3*r.NormFloat64()
					rt = ((f.x+f.s)-cx)/float64(s) + 0.3*r.NormFloat64()
					b = ((f.y+f.s)-cy)/float64(s) + 0.3*r.NormFloat64()
					_ = ss
				}
			}
			sc[k] = float32(p)
			bb[k*4], bb[k*4+1], bb[k*4+2], bb[k*4+3] = float32(l), float32(tp), float32(rt), float32(b)
		}
		shape := func(c int) []int64 {
			if batched {
				return []int64{1, int64(n), int64(c)}
			}
			return []int64{int64(n), int64(c)}
		}
		scores = append(scores, engine.F32(sc, shape(1)...))
		boxes = append(boxes, engine.F32(bb, shape(4)...))
		kps = append(kps, engine.F32(kp, shape(10)...))
	}
	return append(append(scores, boxes...), kps...)
}

func TestGoldenSCRFD(t *testing.T) {
	t.Parallel()
	g := newGolden(t, "scrfd")
	base := models.Config{Name: "scrfd", Width: 640, Height: 640, Letterbox: true,
		Mean: []float32{127.5, 127.5, 127.5}, Std: []float32{128, 128, 128}, ConfThresh: 0.5, MaxDet: 1000}
	small := base
	small.MaxDet = 5
	small.ConfThresh = 0.8
	zero := base
	zero.ConfThresh, zero.MaxDet = 0, 0
	rect := base
	rect.Width, rect.Height = 640, 480
	for _, c := range []struct {
		name string
		cfg  models.Config
	}{{"det10g", base}, {"max5", small}, {"defaults", zero}, {"rect640x480", rect}} {
		m := newModel(t, "scrfd", c.cfg)
		for ii, im := range testImages(t) {
			pre, meta := doPre(m, im.img)
			g.add(c.name+"/pre/"+im.name, pre)
			outs := scrfdOuts(int64(300+ii), c.cfg.Width, c.cfg.Height, false)
			g.add(c.name+"/post/"+im.name, doPost(m, outs, meta))
			bo := scrfdOuts(int64(400+ii), c.cfg.Width, c.cfg.Height, true)
			// shuffled order + batch dim: identified by (N, C), not position.
			bo[0], bo[8] = bo[8], bo[0]
			bo[3], bo[5] = bo[5], bo[3]
			g.add(c.name+"/post-batched/"+im.name, doPost(m, bo, meta))
		}
	}
	m := newModel(t, "scrfd", base)
	meta := models.PreprocessMeta{OrigWidth: 640, OrigHeight: 640, ScaleX: 1, ScaleY: 1}
	bad := scrfdOuts(5, 640, 640, false)
	bad[0].Data[17] = 3.5 // a logit, not a probability
	g.add("err-notprob", doPost(m, bad, meta))
	g.add("partial-match-320", doPost(m, scrfdOuts(6, 320, 320, false), meta))
	g.add("err-nomatch", doPost(m, scrfdOuts(6, 100, 100, false), meta))
	b2 := scrfdOuts(7, 640, 640, true)
	b2[0].Shape = []int64{2, b2[0].Shape[1] / 2, 1}
	g.add("err-batch2", doPost(m, b2, meta))
	none := scrfdOuts(8, 640, 640, false)
	for i := 0; i < 3; i++ {
		for k := range none[i].Data {
			none[i].Data[k] = 0.1
		}
	}
	g.add("no-faces", doPost(m, none, meta))
	g.check()
}

// ---------------------------------------------------------------------------------------
// Classification, depth, CLIP image
// ---------------------------------------------------------------------------------------

func TestGoldenClassification(t *testing.T) {
	t.Parallel()
	g := newGolden(t, "classification")
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	in1k := readLabels(t, "models/mobilenet-v3/imagenet1k.txt")
	for _, c := range []struct {
		name, arch string
		cfg        models.Config
	}{
		{"mobilenet-v3", "mobilenet-v3", models.Config{Name: "mobilenet-v3", Width: 224, Height: 224, Mean: mean, Std: std, MaxDet: 5, Labels: in1k}},
		{"efficientnet-nolabels", "efficientnet", models.Config{Name: "efficientnet-b0", Width: 224, Height: 224, Mean: mean, Std: std}},
		{"top50", "mobilenet-v3", models.Config{Name: "m", Width: 160, Height: 128, MaxDet: 50, Labels: in1k[:10]}},
	} {
		m := newModel(t, c.arch, c.cfg)
		for ii, im := range testImages(t) {
			pre, meta := doPre(m, im.img)
			g.add(c.name+"/pre/"+im.name, pre)
			r := rand.New(rand.NewSource(int64(500 + ii)))
			l := randNorm(r, 1000, 0, 3)
			g.add(c.name+"/post/"+im.name, doPost(m, []engine.Tensor{engine.F32(l, 1, 1000)}, meta))
			g.add(c.name+"/post1d/"+im.name, doPost(m, []engine.Tensor{engine.F32(l, 1000)}, meta))
		}
		g.add(c.name+"/err-batch", doPost(m, []engine.Tensor{engine.F32(make([]float32, 2000), 2, 1000)}, models.PreprocessMeta{}))
		g.add(c.name+"/err-none", doPost(m, nil, models.PreprocessMeta{}))
	}
	g.check()
}

func TestGoldenDepth(t *testing.T) {
	t.Parallel()
	g := newGolden(t, "depth")
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	for _, c := range []struct {
		name, arch string
		cfg        models.Config
	}{
		{"midas", "midas", models.Config{Name: "midas", Width: 256, Height: 256, Mean: mean, Std: std}},
		{"da2", "depth-anything-v2", models.Config{Name: "depth-anything-v2", Width: 518, Height: 518, KeepAspect: true, MultipleOf: 14, Mean: mean, Std: std}},
	} {
		m := newModel(t, c.arch, c.cfg)
		for ii, im := range testImages(t) {
			pre, meta := doPre(m, im.img)
			g.add(c.name+"/pre/"+im.name, pre)
			h, w := int(pre.Tensor.Shape[2]), int(pre.Tensor.Shape[3])
			r := rand.New(rand.NewSource(int64(600 + ii)))
			d := randUniform(r, h*w, -3, 40)
			g.add(c.name+"/post/"+im.name, doPost(m, []engine.Tensor{engine.F32(d, 1, int64(h), int64(w))}, meta))
			g.add(c.name+"/post2d/"+im.name, doPost(m, []engine.Tensor{engine.F32(d, int64(h), int64(w))}, meta))
		}
		flat := make([]float32, 64*48)
		for i := range flat {
			flat[i] = 2.5
		}
		g.add(c.name+"/flat", doPost(m, []engine.Tensor{engine.F32(flat, 1, 48, 64)}, models.PreprocessMeta{}))
		tiny := []float32{1, 1 + 1e-7, 1, 1}
		g.add(c.name+"/tinyrange", doPost(m, []engine.Tensor{engine.F32(tiny, 1, 2, 2)}, models.PreprocessMeta{}))
		g.add(c.name+"/err-len", doPost(m, []engine.Tensor{engine.F32(flat[:10], 1, 48, 64)}, models.PreprocessMeta{}))
		g.add(c.name+"/err-shape", doPost(m, []engine.Tensor{engine.F32(flat, 2, 24, 64)}, models.PreprocessMeta{}))
	}
	g.check()
}

func TestGoldenCLIPImage(t *testing.T) {
	t.Parallel()
	g := newGolden(t, "clip")
	cm := []float32{0.48145466, 0.4578275, 0.40821073}
	cs := []float32{0.26862954, 0.26130258, 0.27577711}
	for _, c := range []struct {
		name string
		cfg  models.Config
	}{
		{"crop", models.Config{Name: "clip", Width: 224, Height: 224, Crop: "center", Mean: cm, Std: cs}},
		{"squash-defaults", models.Config{Name: "clip", Width: 224, Height: 224}},
	} {
		m := newModel(t, "clip", c.cfg)
		for ii, im := range testImages(t) {
			pre, meta := doPre(m, im.img)
			g.add(c.name+"/pre/"+im.name, pre)
			r := rand.New(rand.NewSource(int64(700 + ii)))
			e := randNorm(r, 512, 0, 0.3)
			g.add(c.name+"/post/"+im.name, doPost(m, []engine.Tensor{engine.F32(e, 1, 512)}, meta))
		}
		g.add(c.name+"/zero", doPost(m, []engine.Tensor{engine.F32(make([]float32, 512), 512)}, models.PreprocessMeta{}))
		g.add(c.name+"/err", doPost(m, []engine.Tensor{engine.F32(make([]float32, 1024), 2, 512)}, models.PreprocessMeta{}))
	}
	g.check()
}
