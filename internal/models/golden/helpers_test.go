package golden

import (
	"image"
	"math/rand"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/explain"
	"visionserve/internal/models"
	"visionserve/internal/morph"
	"visionserve/internal/registry"
	"visionserve/pkg/api"
)

// randMask draws a few filled rectangles and specks into a w×h row-major bitmap.
func randMask(r *rand.Rand, w, h int) []bool {
	b := make([]bool, w*h)
	for k := 0; k < 4; k++ {
		x0, y0 := r.Intn(w), r.Intn(h)
		x1, y1 := x0+1+r.Intn(w/2+1), y0+1+r.Intn(h/2+1)
		for y := y0; y < y1 && y < h; y++ {
			for x := x0; x < x1 && x < w; x++ {
				b[y*w+x] = true
			}
		}
	}
	for k := 0; k < w*h/50; k++ {
		b[r.Intn(w*h)] = true
	}
	return b
}

// TestGoldenRLEAndMorph pins the public RLE codec (pkg/api) and the mask morphology that
// re-encodes RLE + recomputes the tight bbox (internal/morph).
func TestGoldenRLEAndMorph(t *testing.T) {
	g := newGolden(t, "rle_morph")
	r := rand.New(rand.NewSource(41))
	for _, sz := range [][2]int{{1, 1}, {7, 5}, {64, 48}, {333, 217}} {
		w, h := sz[0], sz[1]
		bin := randMask(r, w, h)
		rle := api.EncodeMaskRLE(bin, w, h)
		dec := api.DecodeMaskRLE(rle, w, h)
		g.add("rle/"+itoa(w)+"x"+itoa(h), map[string]any{"rle": rle, "decoded": shaBool(dec), "in": shaBool(bin)})
		full := make([]bool, w*h)
		for i := range full {
			full[i] = true
		}
		g.add("rle-full/"+itoa(w)+"x"+itoa(h), api.EncodeMaskRLE(full, w, h))
		g.add("rle-empty/"+itoa(w)+"x"+itoa(h), api.EncodeMaskRLE(make([]bool, w*h), w, h))
		for _, rad := range []int{-3, -1, 0, 1, 4} {
			masks := []api.Mask{{RLE: rle, BBox: [4]float64{1, 2, 3, 4}, Conf: 0.7}, {RLE: "", Conf: 0.1}}
			morph.ApplyToMasks(masks, w, h, rad)
			g.add("morph/"+itoa(w)+"x"+itoa(h)+"/r"+itoa(rad), masks)
		}
	}
	g.add("rle/degenerate", []string{api.EncodeMaskRLE(nil, 3, 3), api.EncodeMaskRLE([]bool{true}, 0, 1)})
	g.add("decode/bad", []string{shaBool(api.DecodeMaskRLE("3 x 2", 3, 2)), shaBool(api.DecodeMaskRLE("", 2, 2))})
	g.check()
}

func itoa(i int) string {
	if i < 0 {
		return "-" + itoa(-i)
	}
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}

// TestGoldenExplain pins the explain heatmaps (min-max normalise + nearest upsample).
func TestGoldenExplain(t *testing.T) {
	g := newGolden(t, "explain")
	r := rand.New(rand.NewSource(51))

	// rfdetr-small cross_attn_weights layout [L=3,B=1,H=16,Q=300,S]; S=1024 is a 32×32 grid,
	// S=300 is non-square (1×N strip).
	for _, s := range []int{1024, 300} {
		attn := engine.F32(randUniform(r, 3*16*300*s, 0, 1), 3, 1, 16, 300, int64(s))
		ex, err := explain.New(&registry.ExplainConfig{Type: "attention", Outputs: map[string]string{"attention": "cross_attn_weights"}})
		if err != nil {
			t.Fatal(err)
		}
		for _, q := range []int{0, 17, 299} {
			hm, w, h, err := ex.Heatmap([]engine.Tensor{{}, attn}, []string{"dets", "cross_attn_weights"}, models.PreprocessMeta{}, q, 333, 217)
			rec := map[string]any{"w": w, "h": h}
			if err != nil {
				rec["err"] = err.Error()
			} else {
				rec["heatmap"] = recTensor(engine.F32(hm, int64(h), int64(w)))
			}
			g.add("attention/s"+itoa(s)+"/q"+itoa(q), rec)
		}
	}

	feat := engine.F32(randNorm(r, 64*20*15, 0, 1), 1, 64, 20, 15)
	ex, err := explain.New(&registry.ExplainConfig{Type: "score_cam", Outputs: map[string]string{"features": "feat"}, TopChannels: 8})
	if err != nil {
		t.Fatal(err)
	}
	hm, w, h, err := ex.Heatmap([]engine.Tensor{feat}, []string{"feat"}, models.PreprocessMeta{}, 0, 160, 120)
	if err != nil {
		t.Fatal(err)
	}
	g.add("scorecam/structural", map[string]any{"w": w, "h": h, "heatmap": recTensor(engine.F32(hm, int64(h), int64(w)))})

	orig := synthImage(160, 120, 52)
	calls := 0
	runner := func(masked image.Image) (float32, error) {
		calls++
		b := masked.Bounds()
		var s float64
		for y := b.Min.Y; y < b.Max.Y; y += 7 {
			for x := b.Min.X; x < b.Max.X; x += 5 {
				rr, gg, bb, _ := masked.At(x, y).RGBA()
				s += float64(rr+2*gg+bb) / 65535
			}
		}
		return float32(s/1000 - 0.5), nil
	}
	hm, w, h, err = explain.ScoreCAMHeatmap(feat, orig, runner, 6, 160, 120)
	if err != nil {
		t.Fatal(err)
	}
	g.add("scorecam/full", map[string]any{"w": w, "h": h, "calls": calls, "heatmap": recTensor(engine.F32(hm, int64(h), int64(w)))})

	g.add("normalize", [][]float32{
		explain.Normalize([]float32{3, -1, 2, 7.5}),
		explain.Normalize([]float32{2, 2, 2}),
		explain.Normalize([]float32{1, 1 + 1e-9}),
		explain.Normalize(nil),
	})
	g.check()
}
