package siglip

import (
	"fmt"
	"image"

	"github.com/disintegration/imaging"
	"visionserve/internal/engine"

	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

func init() {
	models.Register("siglip-image", NewImage)
}

// The SigLIP vision tower's input contract, taken from the checkpoint's own
// preprocessor_config.json rather than assumed:
//
//   - resize to exactly 224x224 with a BICUBIC filter — a squash, NOT shortest-side resize and
//     NOT centre crop, so a wide box is compressed rather than cropped;
//   - rescale 1/255, then (v - 0.5) / 0.5, i.e. the range [-1, 1].
//
// All three differ from CLIP, which shortest-side-resizes, centre-crops, and normalises with its
// own per-channel constants. Feeding SigLIP CLIP's preprocessing produces embeddings that look
// perfectly reasonable and are wrong, which is the failure this project has already paid for once
// on the text side (docs/FINDINGS.md: a hand-built tokenizer scored 13.4% against CLIP's 70.9%,
// and it was broken encoding, not a weak model).
const (
	ImageSize = 224
	// pixelValuesName is the graph input of a standard SigLIP vision export.
	pixelValuesName = "pixel_values"
)

var (
	siglipMean = []float32{0.5, 0.5, 0.5}
	siglipStd  = []float32{0.5, 0.5, 0.5}
)

// imageModel is the SigLIP vision tower (Apache-2.0). It is a plain embedder over whole images;
// the interesting entry point for detection is CropTensor + DecodeImageEmbeddings, which a
// naming head calls directly with a batch of boxes.
type imageModel struct {
	cfg models.Config
}

// NewImage builds the vision tower. It owns no session — lifecycle.Manager keeps role "model"
// alive and Infer reaches it through the Runner.
func NewImage(cfg models.Config) (models.Base, error) { return &imageModel{cfg: cfg}, nil }

func (m *imageModel) Name() string      { return m.cfg.Name }
func (m *imageModel) Task() models.Task { return models.TaskEmbed }
func (m *imageModel) Roles() []string   { return []string{roleModel} }

// Infer embeds the whole image as a single crop, so `siglip-image` is usable on its own for
// image-to-text retrieval. Detection callers should use CropTensor with real boxes instead.
func (m *imageModel) Infer(img image.Image, _ models.Prompt, r models.Runner) (models.Result, error) {
	b := img.Bounds()
	full := [][4]float64{{0, 0, float64(b.Dx()), float64(b.Dy())}}
	embs, err := EmbedCrops(img, full, func(in map[string]engine.Tensor) ([]engine.Tensor, error) {
		return r.Run(roleModel, in)
	}, r.InputNames(roleModel))
	if err != nil {
		return models.Result{}, err
	}
	return models.Result{Task: models.TaskEmbed, Embeddings: embs}, nil
}

// CropTensor builds ONE [N,3,224,224] tensor from N boxes on img, in box order.
//
// It is deliberately batched rather than a per-box call. Measured on an RTX A6000, the vision
// tower costs 3.7-4.4 ms for a single crop but 1.3-1.7 ms per crop at batch 8-16: unbatched
// naming overtakes the ~20 ms detector at 8-9 boxes, batched it stays comparable past 30. A loop
// over single crops is not a slower version of this function, it is a different cost class.
//
// Boxes are [x, y, w, h] in ORIGINAL image coordinates (the convention every Detection uses) and
// are clamped to the image. A box that does not intersect the image, or is degenerate after
// clamping, is an error rather than a silently black crop: it means the caller's coordinates are
// wrong, and a black crop would embed to something plausible and be scored like any other.
func CropTensor(img image.Image, boxes [][4]float64) (engine.Tensor, error) {
	if len(boxes) == 0 {
		return engine.Tensor{}, fmt.Errorf("siglip-image: no boxes to embed")
	}
	plane := ImageSize * ImageSize
	data := make([]float32, len(boxes)*3*plane)
	bounds := img.Bounds()

	for i, b := range boxes {
		rect, err := clampBox(b, bounds)
		if err != nil {
			return engine.Tensor{}, fmt.Errorf("siglip-image: box %d: %w", i, err)
		}
		crop := imaging.Crop(img, rect)
		resized := imageproc.ResizeBicubic(crop, ImageSize, ImageSize)
		t := imageproc.ImageToCHWFloat(resized, siglipMean, siglipStd)
		if len(t.Data) != 3*plane {
			return engine.Tensor{}, fmt.Errorf("siglip-image: box %d produced %d values, want %d",
				i, len(t.Data), 3*plane)
		}
		copy(data[i*3*plane:], t.Data)
	}
	return engine.F32(data, int64(len(boxes)), 3, ImageSize, ImageSize), nil
}

// clampBox turns an [x,y,w,h] float box into an integer rectangle inside bounds.
func clampBox(b [4]float64, bounds image.Rectangle) (image.Rectangle, error) {
	x0 := bounds.Min.X + int(b[0]+0.5)
	y0 := bounds.Min.Y + int(b[1]+0.5)
	x1 := x0 + int(b[2]+0.5)
	y1 := y0 + int(b[3]+0.5)

	if x0 < bounds.Min.X {
		x0 = bounds.Min.X
	}
	if y0 < bounds.Min.Y {
		y0 = bounds.Min.Y
	}
	if x1 > bounds.Max.X {
		x1 = bounds.Max.X
	}
	if y1 > bounds.Max.Y {
		y1 = bounds.Max.Y
	}
	if x1-x0 < 1 || y1-y0 < 1 {
		return image.Rectangle{}, fmt.Errorf(
			"box [%.1f %.1f %.1f %.1f] is empty after clamping to the %dx%d image — the "+
				"coordinates are probably not in original-image space",
			b[0], b[1], b[2], b[3], bounds.Dx(), bounds.Dy())
	}
	return image.Rect(x0, y0, x1, y1), nil
}

// EmbedCrops crops, preprocesses and embeds every box in ONE session call, returning one
// L2-normalised row per box, in box order.
//
// The tower does not normalise internally — neither does the text tower — so both sides are
// normalised here and a score is a plain dot product. Getting this wrong yields cosines that are
// merely proportional to the right ones, which preserves an argmax and quietly breaks any
// threshold.
func EmbedCrops(img image.Image, boxes [][4]float64,
	run func(map[string]engine.Tensor) ([]engine.Tensor, error), inputNames []string) ([][]float32, error) {
	in, err := CropTensor(img, boxes)
	if err != nil {
		return nil, err
	}
	name := pixelValuesName
	if len(inputNames) > 0 {
		found := false
		for _, n := range inputNames {
			if n == pixelValuesName {
				found = true
				break
			}
		}
		if !found {
			name = inputNames[0]
		}
	}
	outs, err := run(map[string]engine.Tensor{name: in})
	if err != nil {
		return nil, fmt.Errorf("siglip-image: inference: %w", err)
	}
	return DecodeImageEmbeddings(outs, len(boxes))
}

// DecodeImageEmbeddings reshapes an [N, D] output into N L2-normalised rows. D is not asserted to
// any width: 768 for base, 1152 for so400m.
func DecodeImageEmbeddings(outs []engine.Tensor, n int) ([][]float32, error) {
	if len(outs) == 0 {
		return nil, fmt.Errorf("siglip-image: expected 1 output tensor, got 0")
	}
	t := outs[0]
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("siglip-image: unexpected output shape %v (expected [N,D])", t.Shape)
	}
	if int(t.Shape[0]) != n {
		return nil, fmt.Errorf("siglip-image: output batch %d != %d crops", t.Shape[0], n)
	}
	dim := int(t.Shape[1])
	if dim <= 0 || len(t.Data) < n*dim {
		return nil, fmt.Errorf("siglip-image: output data length %d < %d*%d", len(t.Data), n, dim)
	}
	embs := make([][]float32, n)
	for i := 0; i < n; i++ {
		embs[i] = l2Normalize(t.Data[i*dim : (i+1)*dim])
	}
	return embs, nil
}

// ScoreCrops returns a [len(crops), len(text)] similarity matrix, row-major. Both sides are
// already L2-normalised, so a cosine is a dot product.
func ScoreCrops(crops, text [][]float32) ([]float32, error) {
	if len(crops) == 0 || len(text) == 0 {
		return nil, fmt.Errorf("siglip-image: nothing to score (%d crops, %d names)", len(crops), len(text))
	}
	dim := len(crops[0])
	out := make([]float32, len(crops)*len(text))
	for i, c := range crops {
		if len(c) != dim {
			return nil, fmt.Errorf("siglip-image: crop %d has width %d, want %d", i, len(c), dim)
		}
		for j, t := range text {
			if len(t) != dim {
				return nil, fmt.Errorf("siglip-image: name %d has width %d but crops are %d — "+
					"the two towers are different checkpoints", j, len(t), dim)
			}
			var s float32
			for k := range c {
				s += c[k] * t[k]
			}
			out[i*len(text)+j] = s
		}
	}
	return out, nil
}
