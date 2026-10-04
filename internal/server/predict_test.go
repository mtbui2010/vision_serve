package server

import (
	"context"
	"image"
	"testing"

	"visionserve/internal/models"
	"visionserve/pkg/api"
)

// fixedPredictor answers every prediction with a copy of one result.
type fixedPredictor struct{ res api.Result }

func (f fixedPredictor) PredictPrompt(context.Context, string, image.Image, models.Prompt) (api.Result, error) {
	res := f.res
	res.Detections = append([]api.Detection(nil), f.res.Detections...)
	res.Masks = append([]api.Mask(nil), f.res.Masks...)
	return res, nil
}

// rectMask is a w×h mask filled inside box (x, y, bw, bh).
func rectMask(w, h int, box [4]float64) string {
	bits := make([]bool, w*h)
	for y := int(box[1]); y < int(box[1]+box[3]); y++ {
		for x := int(box[0]); x < int(box[0]+box[2]); x++ {
			bits[y*w+x] = true
		}
	}
	return api.EncodeMaskRLE(bits, w, h)
}

func countSet(rle string, w, h int) int {
	n := 0
	for _, b := range api.DecodeMaskRLE(rle, w, h) {
		if b {
			n++
		}
	}
	return n
}

// pairedResult is a Grounded-SAM-shaped result on a 100×100 image: detection i and mask i are one
// object, and the mask carries the detection's box and conf.
func pairedResult() api.Result {
	const w, h = 100, 100
	boxes := [][4]float64{
		{10, 10, 20, 20}, // 4 % of the image; eroding by 3 px shrinks its tight box to 1.96 %
		{50, 50, 10, 10}, // 1 %
		{60, 0, 40, 40},  // 16 %
	}
	classes := []string{"dog", "cup", "bench"}
	var res api.Result
	for i, b := range boxes {
		conf := 0.9 - 0.1*float64(i)
		res.Detections = append(res.Detections, api.Detection{BBox: b, Class: classes[i], Conf: conf})
		res.Masks = append(res.Masks, api.Mask{RLE: rectMask(w, h, b), BBox: b, Conf: conf})
	}
	return res
}

// One decision per object: with dilate and min_size/max_size together, a detection and its mask
// are kept or dropped together, and every kept mask still carries its detection's box and conf
// (so clients that pair by box, e.g. group_by_class, keep the class). The bug: masks were judged
// by their re-tightened box, detections by theirs, and the two lists came back different lengths.
func TestPredictKeepsDetectionsAndMasksPaired(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 100, 100))
	cases := []struct {
		name      string
		prompt    models.Prompt
		wantClass []string
	}{
		{"erode only", models.Prompt{Dilate: -3}, []string{"dog", "cup", "bench"}},
		{"dilate only", models.Prompt{Dilate: 4}, []string{"dog", "cup", "bench"}},
		{"erode + min_size", models.Prompt{Dilate: -3, MinSize: 3}, []string{"dog", "bench"}},
		{"dilate + max_size", models.Prompt{Dilate: 5, MaxSize: 5}, []string{"dog", "cup"}},
		{"min_size only", models.Prompt{MinSize: 3}, []string{"dog", "bench"}},
	}
	for _, c := range cases {
		in := pairedResult()
		res, err := Predict(context.Background(), fixedPredictor{in}, "m", img, c.prompt)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if len(res.Detections) != len(c.wantClass) || len(res.Masks) != len(res.Detections) {
			t.Fatalf("%s: %d detections, %d masks; want %d of each", c.name, len(res.Detections), len(res.Masks), len(c.wantClass))
		}
		if !api.MasksPairDetections(res) {
			t.Errorf("%s: result no longer paired", c.name)
		}
		for i, d := range res.Detections {
			m := res.Masks[i]
			if d.Class != c.wantClass[i] || m.BBox != d.BBox || m.Conf != d.Conf {
				t.Errorf("%s: object %d: detection %+v, mask box %v conf %v", c.name, i, d, m.BBox, m.Conf)
			}
			// The morphology still happened: the pixels changed, only the box was kept.
			before := int(d.BBox[2] * d.BBox[3])
			got := countSet(m.RLE, 100, 100)
			switch {
			case c.prompt.Dilate < 0 && got >= before, c.prompt.Dilate > 0 && got <= before,
				c.prompt.Dilate == 0 && got != before:
				t.Errorf("%s: object %d has %d pixels (was %d)", c.name, i, got, before)
			}
		}
	}
}

// Masks that belong to no detection (SAM prompts, automask, background) still get the tight box
// of the reshaped mask, as documented.
func TestPredictRetightensUnpairedMasks(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 100, 100))
	in := pairedResult()
	in.Detections = nil
	res, err := Predict(context.Background(), fixedPredictor{in}, "m", img, models.Prompt{Dilate: -3})
	if err != nil {
		t.Fatal(err)
	}
	if want := [4]float64{13, 13, 14, 14}; res.Masks[0].BBox != want {
		t.Errorf("unpaired mask box after erode 3 = %v, want %v", res.Masks[0].BBox, want)
	}
}
