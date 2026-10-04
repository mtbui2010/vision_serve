package pipeline

import (
	"image"
	"sync"
	"testing"

	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"
)

// promptRecorder is a MaskInferer (and, when each is set, a MaskEacher) that records the
// prompt the segmenter stage hands the model and returns no masks.
type promptRecorder struct {
	mu   sync.Mutex
	seen []models.Prompt
}

func (r *promptRecorder) record(p models.Prompt) {
	r.mu.Lock()
	r.seen = append(r.seen, p)
	r.mu.Unlock()
}

func (r *promptRecorder) InferMasks(_ image.Image, p models.Prompt, _ models.Runner) ([]models.Mask, []mobilesam.MaskBitmap, error) {
	r.record(p)
	return nil, nil, nil
}

type eachRecorder struct{ promptRecorder }

func (r *eachRecorder) InferMasksEach(_ image.Image, p models.Prompt, _ models.Runner, _ func(mobilesam.MaskBitmap) any) ([]any, error) {
	r.record(p)
	return nil, nil
}

// The class-agnostic grasp model's automatic masks honour the request's grid_size, on both the
// streaming and the non-streaming segmenter path. They used to get only the boxes, so grid_size
// was silently ignored. Box prompts carry the boxes; nothing else of the request leaks through.
func TestSAMBitmapsPassesGridSize(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 32, 32))
	call := Call{Img: img, Prompt: models.Prompt{GridSize: 4, Text: "cup.", MinSize: 1}}
	planner := AnalyticGrasp{GripMin: 1, GripMax: 10, MaxPerMask: 1}

	plain := &promptRecorder{}
	if _, err := (Grasp{Segmenter: SAMBitmaps{Model: plain}, Planner: planner}).Infer(call); err != nil {
		t.Fatal(err)
	}
	each := &eachRecorder{}
	if _, err := (Grasp{Segmenter: SAMBitmaps{Model: each}, Planner: planner}).Infer(call); err != nil {
		t.Fatal(err)
	}
	for name, seen := range map[string][]models.Prompt{"InferMasks": plain.seen, "InferMasksEach": each.seen} {
		if len(seen) != 1 {
			t.Fatalf("%s: %d calls, want 1", name, len(seen))
		}
		if seen[0].GridSize != 4 || seen[0].Text != "" || seen[0].MinSize != 0 {
			t.Errorf("%s got prompt %+v, want only GridSize=4", name, seen[0])
		}
	}

	// Box prompts: the boxes reach the model.
	boxes := [][4]float64{{1, 2, 3, 4}}
	if _, _, err := (SAMBitmaps{Model: plain}).SegmentBitmaps(call, boxes); err != nil {
		t.Fatal(err)
	}
	if got := plain.seen[len(plain.seen)-1]; len(got.Boxes) != 1 || got.Boxes[0] != boxes[0] {
		t.Errorf("box prompt lost its boxes: %+v", got)
	}
}
