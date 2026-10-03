package background

import (
	"image"
	"reflect"
	"sync"
	"testing"

	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"
)

// eachSeg is fakeSeg that also streams: fn runs concurrently, each call on a scratch copy that
// is clobbered once fn returns (fn must not keep it), as in MobileSAM's final pass.
type eachSeg struct {
	fakeSeg
	each int
}

func (f *eachSeg) InferMasksEach(_ image.Image, _ models.Prompt, _ models.Runner, fn func(mobilesam.MaskBitmap) any) ([]any, error) {
	f.each++
	out := make([]any, len(f.bms))
	var wg sync.WaitGroup
	for i := range f.bms {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			b := f.bms[i]
			b.Data = append([]bool(nil), b.Data...)
			out[i] = fn(b)
			for k := range b.Data {
				b.Data[k] = !b.Data[k]
			}
		}(i)
	}
	wg.Wait()
	return out, nil
}

// The streaming automask union is exactly the one built from the returned bitmaps: the same
// masks qualify and OR is order-independent. No qualifying mask is still nil.
func TestBackgroundAutomaskEachMatchesBitmaps(t *testing.T) {
	const w, h = 100, 80
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	sets := map[string][]mobilesam.MaskBitmap{
		"mixed": {
			rectMask(w, h, 0, 60, 100, 80),  // border strip: qualifies
			rectMask(w, h, 30, 20, 50, 40),  // small interior object: does not
			rectMask(w, h, 0, 0, 90, 75),    // large: qualifies
			rectMask(w, h, 40, 10, 100, 30), // touches the right edge
		},
		"none":  {rectMask(w, h, 30, 20, 50, 40)},
		"empty": nil,
	}
	for name, bms := range sets {
		for _, p := range []models.Prompt{{}, {FgMinArea: 10}, {BgMaxArea: 20}} {
			m := &backgroundModel{seg: &fakeSeg{bms: bms}, hasSAM: true}
			want, err := m.backgroundAutomask(img, p, nil)
			if err != nil {
				t.Fatal(err)
			}
			es := &eachSeg{fakeSeg: fakeSeg{bms: bms}}
			got, err := (&backgroundModel{seg: es, hasSAM: true}).backgroundAutomask(img, p, nil)
			if err != nil {
				t.Fatal(err)
			}
			if es.each != 1 || es.calls != 0 {
				t.Fatalf("%s: streaming path not taken (each %d, InferMasks %d)", name, es.each, es.calls)
			}
			if (got == nil) != (want == nil) || !reflect.DeepEqual(got, want) {
				t.Fatalf("%s %+v: streaming union differs (nil %v vs %v)", name, p, got == nil, want == nil)
			}
			if name == "mixed" && p.FgMinArea == 0 && p.BgMaxArea == 0 && want == nil {
				t.Fatal("mixed: no surface found; the test scene is wrong")
			}
		}
	}
}
