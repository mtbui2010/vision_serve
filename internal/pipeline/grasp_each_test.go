package pipeline

import (
	"image"
	"reflect"
	"sync"
	"testing"

	"visionserve/internal/models"
	"visionserve/internal/vision/mask"
)

// listSeg is a non-streaming BitmapSegmenter over fixed bitmaps (automask only).
type listSeg struct{ bms []mask.Bitmap }

func (s listSeg) masks() []models.Mask {
	out := make([]models.Mask, len(s.bms))
	for i, b := range s.bms {
		out[i] = models.Mask{RLE: mask.EncodeRLE(b), BBox: b.BBox(), Conf: 0.9}
	}
	return out
}

func (s listSeg) SegmentBitmaps(Call, [][4]float64) ([]models.Mask, []mask.Bitmap, error) {
	return s.masks(), s.bms, nil
}

// eachSeg streams the same bitmaps like MobileSAM's final pass: fn runs concurrently, each call
// on a scratch copy that is clobbered as soon as fn returns (fn must not keep it).
type eachSeg struct {
	listSeg
	stream bool // false: reports it cannot stream
	calls  int
}

func (s *eachSeg) SegmentEach(_ Call, _ [][4]float64, fn func(models.Mask, mask.Bitmap) any) ([]any, bool, error) {
	if !s.stream {
		return nil, false, nil
	}
	s.calls++
	ms := s.masks()
	out := make([]any, len(s.bms))
	var wg sync.WaitGroup
	for i := range s.bms {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			b := s.bms[i]
			scratch := mask.Bitmap{Data: append([]bool(nil), b.Data...), W: b.W, H: b.H}
			out[i] = fn(ms[i], scratch)
			for k := range scratch.Data {
				scratch.Data[k] = !scratch.Data[k]
			}
		}(i)
	}
	wg.Wait()
	return out, true, nil
}

func rectBitmap(w, h, x0, y0, x1, y1 int) mask.Bitmap {
	b := mask.New(h, w)
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			b.Data[y*w+x] = true
		}
	}
	return b
}

// The streaming automask branch gives exactly the non-streaming result: same masks, same size
// filter, same grasps in the same order, with or without min_size/max_size.
func TestGraspAutomaskEachMatchesBitmaps(t *testing.T) {
	const w, h = 320, 240
	bms := []mask.Bitmap{
		rectBitmap(w, h, 20, 30, 80, 70),    // ~3 % of the image
		rectBitmap(w, h, 150, 40, 170, 200), // thin bar
		rectBitmap(w, h, 0, 0, 300, 230),    // ~90 %
		rectBitmap(w, h, 200, 150, 260, 190),
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	planner := AnalyticGrasp{GripMin: 5, GripMax: 120, MaxPerMask: 20}
	for _, p := range []models.Prompt{{}, {MinSize: 5}, {MaxSize: 50}, {MinSize: 2, MaxSize: 10}} {
		want, err := Grasp{Segmenter: listSeg{bms}, Planner: planner}.Infer(Call{Img: img, Prompt: p})
		if err != nil {
			t.Fatal(err)
		}
		es := &eachSeg{listSeg: listSeg{bms}, stream: true}
		got, err := Grasp{Segmenter: es, Planner: planner}.Infer(Call{Img: img, Prompt: p})
		if err != nil {
			t.Fatal(err)
		}
		if es.calls != 1 {
			t.Fatalf("prompt %+v: streaming segmenter not used", p)
		}
		if (p.MinSize == 0 && p.MaxSize == 0 && len(want.Grasps) == 0) || !reflect.DeepEqual(got, want) {
			t.Fatalf("prompt %+v: streaming result differs\n got  %d masks %d grasps\n want %d masks %d grasps",
				p, len(got.Masks), len(got.Grasps), len(want.Masks), len(want.Grasps))
		}
		// A segmenter that cannot stream falls back to SegmentBitmaps.
		fb, err := Grasp{Segmenter: &eachSeg{listSeg: listSeg{bms}}, Planner: planner}.Infer(Call{Img: img, Prompt: p})
		if err != nil || !reflect.DeepEqual(fb, want) {
			t.Fatalf("prompt %+v: fallback differs (err %v)", p, err)
		}
	}
}
