package mobilesam

import (
	"fmt"
	"image"
	"math"
	"reflect"
	"sync"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// fakeDecoder mimics the REAL MobileSAM decoder I/O (verified on
// models/mobile-sam/mobile_sam_decoder_{single,multi}.onnx):
//
//	inputs : image_embeddings [1,256,64,64], point_coords [1,n,2], point_labels [1,n],
//	         mask_input [1,1,256,256], has_mask_input [1], orig_im_size [2] = (H,W)
//	outputs: masks [1,M,H,W] (upsampled to orig_im_size IN the graph),
//	         iou_predictions [1,M], low_res_masks [1,M,256,256]   (M=1 single, 4 multi)
//
// Scene (original-image pixels): rectangles; a point inside one yields that rectangle's
// mask, a point outside all of them yields the "everything else" mask. Masks are rendered
// at whatever frame orig_im_size asks for, like the real graph.
type fakeDecoder struct {
	origW, origH int
	scale        float64 // original → 1024 space (what the model multiplies point coords by)
	rects        [][4]int
	m            int // mask channels (1 = single decoder, 4 = multi)

	mu    sync.Mutex
	calls map[[2]int]int // orig_im_size (W,H) → call count
	// intoBufs counts the calls that wrote their masks into each distinct caller buffer
	// (keyed by its first element's address); intoCalls is their total.
	intoBufs  map[*float32]int
	intoCalls int
}

// runInto is run honouring caller buffers like ONNX Runtime does: an output named in into is
// written into that buffer (its shape must be exactly the produced shape) and returned
// aliasing it.
func (f *fakeDecoder) runInto(in, into map[string]engine.Tensor) ([]engine.Tensor, error) {
	outs, err := f.run(in)
	if err != nil || into == nil {
		return outs, err
	}
	for name, buf := range into {
		i := -1
		for k, n := range realDecOutNames {
			if n == name {
				i = k
			}
		}
		if i < 0 {
			return nil, fmt.Errorf("into names unknown output %q", name)
		}
		if !reflect.DeepEqual(buf.Shape, outs[i].Shape) || len(buf.Data) != len(outs[i].Data) {
			return nil, fmt.Errorf("into %q shape %v (len %d), output is %v", name, buf.Shape, len(buf.Data), outs[i].Shape)
		}
		for k := range buf.Data { // ORT overwrites the whole buffer: poison it first
			buf.Data[k] = float32(math.NaN())
		}
		copy(buf.Data, outs[i].Data)
		outs[i] = engine.Tensor{Data: buf.Data, Shape: outs[i].Shape}
		f.mu.Lock()
		f.intoBufs[&buf.Data[0]]++
		f.intoCalls++
		f.mu.Unlock()
	}
	return outs, nil
}

func (f *fakeDecoder) run(in map[string]engine.Tensor) ([]engine.Tensor, error) {
	mi := in["mask_input"]
	if len(mi.Shape) != 4 || mi.Shape[2] != 256 || mi.Shape[3] != 256 {
		return nil, fmt.Errorf("mask_input shape %v", mi.Shape)
	}
	for _, v := range mi.Data {
		if v != 0 {
			return nil, fmt.Errorf("shared zero mask_input was modified")
		}
	}
	if in["has_mask_input"].Data[0] != 0 {
		return nil, fmt.Errorf("has_mask_input must be 0")
	}
	sz := in["orig_im_size"].Data
	H, W := int(sz[0]), int(sz[1])
	f.mu.Lock()
	f.calls[[2]int{W, H}]++
	f.mu.Unlock()

	pc := in["point_coords"].Data
	ox, oy := float64(pc[0])/f.scale, float64(pc[1])/f.scale // back to original pixels
	hit := -1
	for k, r := range f.rects {
		if ox >= float64(r[0]) && ox < float64(r[0]+r[2]) && oy >= float64(r[1]) && oy < float64(r[1]+r[3]) {
			hit = k
		}
	}
	inside := func(x, y int) bool { // (x,y) in the H×W frame → in original rect?
		fx := (float64(x) + 0.5) * float64(f.origW) / float64(W)
		fy := (float64(y) + 0.5) * float64(f.origH) / float64(H)
		for k, r := range f.rects {
			if (hit < 0 || k == hit) && fx >= float64(r[0]) && fx < float64(r[0]+r[2]) &&
				fy >= float64(r[1]) && fy < float64(r[1]+r[3]) {
				return true
			}
		}
		return false
	}
	masks := make([]float32, f.m*H*W)
	ious := make([]float32, f.m)
	best := f.m - 1 // put the good channel last to exercise bestChannel
	for c := 0; c < f.m; c++ {
		ious[c] = 0.5
	}
	ious[best] = 0.97
	if hit < 0 {
		ious[best] = 0.9
	}
	off := best * H * W
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			in := inside(x, y)
			if hit < 0 {
				in = !in
			}
			if in {
				masks[off+y*W+x] = 5
			} else {
				masks[off+y*W+x] = -5
			}
		}
	}
	return []engine.Tensor{
		engine.F32(masks, 1, int64(f.m), int64(H), int64(W)),
		engine.F32(ious, 1, int64(f.m)),
		engine.F32(make([]float32, f.m*256*256), 1, int64(f.m), 256, 256),
	}, nil
}

var realDecOutNames = []string{"masks", "iou_predictions", "low_res_masks"}

func newFake(w, h, m int, rects [][4]int) *fakeDecoder {
	return &fakeDecoder{
		origW: w, origH: h, scale: 1024 / float64(max(w, h)), rects: rects, m: m,
		calls: map[[2]int]int{}, intoBufs: map[*float32]int{},
	}
}

// plainRun is f as a decodeFunc that cannot write into caller buffers (ignores into).
func (f *fakeDecoder) plainRun(in, _ map[string]engine.Tensor) ([]engine.Tensor, error) {
	return f.run(in)
}

// Large image: candidates are filtered/NMS'd in the 256-px work frame and ONLY the kept
// points are decoded again at original resolution. Output masks are at ORIGINAL
// resolution with original-pixel bboxes.
func TestAutoSegmentLowResFilterFullResKept(t *testing.T) {
	for _, m := range []int{1, 4} { // single and multi decoder exports
		t.Run(fmt.Sprintf("M=%d", m), func(t *testing.T) {
			const W, H = 1200, 800
			rects := [][4]int{{100, 100, 300, 200}, {700, 400, 200, 300}}
			f := newFake(W, H, m, rects)
			img := image.NewNRGBA(image.Rect(0, 0, W, H))
			emb := engine.F32(make([]float32, 256*64*64), 1, 256, 64, 64)

			out, err := autoSegment(img, emb, f.scale, f.plainRun, realDecOutNames, 8)
			if err != nil {
				t.Fatal(err)
			}
			// 2 rectangles + the "everything else" mask (area < 95%, conf 0.9 ≥ 0.85).
			if len(out) != 3 {
				t.Fatalf("got %d masks, want 3", len(out))
			}
			ww, wh := workFrame(W, H)
			if ww != 256 || wh != 171 {
				t.Fatalf("work frame %dx%d, want 256x171", ww, wh)
			}
			if got := f.calls[[2]int{ww, wh}]; got != 64 {
				t.Errorf("filter-pass calls = %d, want 64 (8x8 grid)", got)
			}
			if got := f.calls[[2]int{W, H}]; got != 3 {
				t.Errorf("full-res calls = %d, want 3 (kept masks only)", got)
			}
			// Highest conf first: the two rectangles (0.97) in grid order, then background.
			want := [][4]float64{{100, 100, 300, 200}, {700, 400, 200, 300}}
			for k, bm := range out[:2] {
				if bm.W != W || bm.H != H || len(bm.Data) != W*H {
					t.Fatalf("mask %d is %dx%d (len %d), want original %dx%d", k, bm.W, bm.H, len(bm.Data), W, H)
				}
				if bm.BBox != want[k] {
					t.Errorf("mask %d bbox %v, want %v (original pixels)", k, bm.BBox, want[k])
				}
				if bm.Conf < 0.96 {
					t.Errorf("mask %d conf %v, want 0.97 (best channel)", k, bm.Conf)
				}
			}
		})
	}
}

// A small image (long side ≤ 256) is its own work frame: no second pass.
func TestAutoSegmentSmallImageSinglePass(t *testing.T) {
	const W, H = 200, 150
	f := newFake(W, H, 1, [][4]int{{20, 20, 60, 40}})
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	emb := engine.F32(make([]float32, 256*64*64), 1, 256, 64, 64)
	out, err := autoSegment(img, emb, f.scale, f.plainRun, realDecOutNames, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.calls) != 1 || f.calls[[2]int{W, H}] != 16 {
		t.Fatalf("calls %v, want only 16 at %dx%d", f.calls, W, H)
	}
	if len(out) == 0 || out[0].W != W || out[0].H != H || out[0].BBox != [4]float64{20, 20, 60, 40} {
		t.Fatalf("unexpected output %+v", out)
	}
}

// fakeRunner serves the fake decoder plus a constant encoder, as a models.Runner.
type fakeRunner struct{ dec *fakeDecoder }

func (r fakeRunner) Run(role string, in map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role == roleEncoder {
		return []engine.Tensor{engine.F32(make([]float32, 256*64*64), 1, 256, 64, 64)}, nil
	}
	return r.dec.run(in)
}

// intoRunner is fakeRunner that also implements models.IntoRunner, like lifecycle's runner.
type intoRunner struct{ fakeRunner }

func (r intoRunner) RunInto(role string, in, into map[string]engine.Tensor) ([]engine.Tensor, error) {
	if role == roleEncoder {
		return r.Run(role, in)
	}
	return r.dec.runInto(in, into)
}

func (fakeRunner) InputNames(role string) []string {
	if role == roleEncoder {
		return []string{"input_image"}
	}
	return []string{"image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"}
}
func (fakeRunner) OutputNames(role string) []string {
	if role == roleEncoder {
		return []string{"image_embeddings"}
	}
	return realDecOutNames
}

// Infer encodes each mask as soon as it is final instead of holding every bitmap; its masks
// must be exactly InferMasks' (the bitmaps encoded at the end), for automask with and without
// the full-resolution pass and for prompts.
func TestInferMatchesInferMasks(t *testing.T) {
	m := &mobileSAM{gridSize: 8}
	cases := []struct {
		name   string
		w, h   int
		prompt string
	}{
		{"automask-two-pass", 1200, 800, ""},
		{"automask-small", 200, 150, ""},
		{"box", 1200, 800, "box"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rects := [][4]int{{c.w / 12, c.h / 8, c.w / 4, c.h / 4}, {c.w * 7 / 12, c.h / 2, c.w / 6, c.h / 3}}
			img := image.NewNRGBA(image.Rect(0, 0, c.w, c.h))
			var p models.Prompt
			if c.prompt == "box" {
				p.Boxes = [][4]float64{{float64(rects[0][0]), float64(rects[0][1]), float64(rects[0][2]), float64(rects[0][3])}}
			}
			res, err := m.Infer(img, p, fakeRunner{newFake(c.w, c.h, 1, rects)})
			if err != nil {
				t.Fatal(err)
			}
			want, _, err := m.InferMasks(img, p, fakeRunner{newFake(c.w, c.h, 1, rects)})
			if err != nil {
				t.Fatal(err)
			}
			if len(want) == 0 || !reflect.DeepEqual(res.Masks, want) {
				t.Fatalf("Infer masks %+v\nwant %+v", res.Masks, want)
			}
		})
	}
}

// NMS: higher confidence wins, ties break by grid index, disjoint masks both survive.
func TestNMSCandidates(t *testing.T) {
	const w, h = 10, 10
	rect := func(x0, y0, x1, y1 int, conf float64, idx int) aCandidate {
		ten := engine.F32(make([]float32, w*h), 1, 1, h, w)
		for y := y0; y <= y1; y++ {
			for x := x0; x <= x1; x++ {
				ten.Data[y*w+x] = 1
			}
		}
		c := thresholdBest(&ten, nil)
		c.conf, c.idx = conf, idx
		return c
	}
	cands := []aCandidate{
		rect(0, 0, 4, 4, 0.90, 3),
		rect(0, 0, 4, 4, 0.95, 7), // same mask, higher conf → kept
		rect(6, 6, 9, 9, 0.90, 1), // disjoint → kept
		rect(6, 6, 9, 9, 0.90, 0), // identical + same conf, lower idx → wins the tie
	}
	kept := nmsCandidates(cands, autoDedupeIoU)
	if len(kept) != 2 || kept[0].idx != 7 || kept[1].idx != 0 {
		got := []int{}
		for _, k := range kept {
			got = append(got, k.idx)
		}
		t.Fatalf("kept idx %v, want [7 0]", got)
	}
}

// With a Runner that writes outputs into caller buffers, the final pass decodes every kept
// point into one buffer per worker (at most autoFinalWorkers of them, reused across points),
// the filter pass passes none, and the masks are exactly the plain Runner's.
func TestAutoSegmentFinalPassReusesBuffers(t *testing.T) {
	for _, m := range []int{1, 4} {
		t.Run(fmt.Sprintf("M=%d", m), func(t *testing.T) {
			const W, H = 1200, 800
			var rects [][4]int // a 4×3 grid of separate objects → 12 kept masks + background
			for i := 0; i < 4; i++ {
				for j := 0; j < 3; j++ {
					rects = append(rects, [4]int{40 + 300*i, 40 + 260*j, 150, 120})
				}
			}
			img := image.NewNRGBA(image.Rect(0, 0, W, H))
			sam := &mobileSAM{gridSize: 16}
			fi := newFake(W, H, m, rects)
			got, err := sam.Infer(img, models.Prompt{}, intoRunner{fakeRunner{fi}})
			if err != nil {
				t.Fatal(err)
			}
			want, err := sam.Infer(img, models.Prompt{}, fakeRunner{newFake(W, H, m, rects)})
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Masks) != len(rects)+1 || !reflect.DeepEqual(got.Masks, want.Masks) {
				t.Fatalf("into masks (%d) differ from plain masks (%d)", len(got.Masks), len(want.Masks))
			}
			if full := fi.calls[[2]int{W, H}]; fi.intoCalls != full || full != len(rects)+1 {
				t.Errorf("%d of %d full-res calls wrote into a caller buffer, want all %d", fi.intoCalls, full, len(rects)+1)
			}
			if n := len(fi.intoBufs); n == 0 || n > autoFinalWorkers {
				t.Errorf("%d distinct buffers, want 1..%d (one per final-pass worker)", n, autoFinalWorkers)
			}
		})
	}
}

func TestMaskOutputName(t *testing.T) {
	for _, c := range []struct {
		names []string
		want  string
	}{
		{realDecOutNames, "masks"},
		{[]string{"iou_predictions", "low_res_masks", "masks"}, "masks"},
		{[]string{"out0", "out1"}, ""}, // picked by shape: no buffer
		{[]string{"low_res_masks", "iou_predictions"}, ""},
	} {
		if got := maskOutputName(c.names); got != c.want {
			t.Errorf("maskOutputName(%v) = %q, want %q", c.names, got, c.want)
		}
	}
}

// InferMasksEach hands fn exactly InferMasks' bitmaps, in order, with either Runner. The final
// pass thresholds into one buffer per worker (fn must copy), so at most autoFinalWorkers
// distinct backing arrays reach fn for the full-resolution masks.
func TestInferMasksEachMatchesInferMasks(t *testing.T) {
	const W, H = 1200, 800
	var rects [][4]int
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			rects = append(rects, [4]int{40 + 300*i, 40 + 260*j, 150, 120})
		}
	}
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	sam := &mobileSAM{gridSize: 16}
	_, want, err := sam.InferMasks(img, models.Prompt{}, fakeRunner{newFake(W, H, 1, rects)})
	if err != nil {
		t.Fatal(err)
	}
	for name, r := range map[string]models.Runner{
		"plain": fakeRunner{newFake(W, H, 1, rects)},
		"into":  intoRunner{fakeRunner{newFake(W, H, 1, rects)}},
	} {
		var mu sync.Mutex
		backing := map[*bool]bool{}
		got, err := sam.InferMasksEach(img, models.Prompt{}, r, func(b MaskBitmap) any {
			mu.Lock()
			backing[&b.Data[0]] = true
			mu.Unlock()
			b.Data = append([]bool(nil), b.Data...) // fn must not keep the borrowed Data
			return b
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want) || len(want) != len(rects)+1 {
			t.Fatalf("%s: %d masks, want %d (= %d)", name, len(got), len(want), len(rects)+1)
		}
		for i := range want {
			if !reflect.DeepEqual(got[i].(MaskBitmap), want[i]) {
				t.Fatalf("%s: mask %d differs from InferMasks'", name, i)
			}
		}
		if len(backing) > autoFinalWorkers {
			t.Errorf("%s: %d distinct bitmap buffers reached fn, want ≤ %d (one per final-pass worker)", name, len(backing), autoFinalWorkers)
		}
	}
}
