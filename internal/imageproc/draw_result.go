package imageproc

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"strconv"
	"strings"

	"visionserve/pkg/api"
)

// DrawResult renders a full Result onto a copy of img: semi-transparent mask overlays
// (segmentation) UNDER the boxes, then detection/mask bounding boxes with labels.
// Pure Go (no cgo), like the rest of imageproc.
func DrawResult(img image.Image, res api.Result) *image.RGBA {
	b := img.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, img, b.Min, draw.Src)

	thick := b.Dx() / 400
	if thick < 2 {
		thick = 2
	}

	// 1) Mask overlays first (so boxes/labels stay on top).
	for i, mk := range res.Masks {
		col := palette[i%len(palette)]
		overlayMask(out, mk.RLE, b, col)
	}

	// 2) Detection boxes + "class conf%".
	for i, d := range res.Detections {
		col := palette[i%len(palette)]
		x, y, w, h := boxToInts(d.BBox)
		drawRectOutline(out, b.Min.X+x, b.Min.Y+y, w, h, thick, col)
		drawLabel(out, b.Min.X+x, b.Min.Y+y, fmt.Sprintf("%s %.0f%%", d.Class, d.Conf*100), col)
	}

	// 3) Mask bounding boxes + confidence (when there are no detections to label them).
	for i, mk := range res.Masks {
		if mk.BBox == [4]float64{} {
			continue
		}
		col := palette[i%len(palette)]
		x, y, w, h := boxToInts(mk.BBox)
		drawRectOutline(out, b.Min.X+x, b.Min.Y+y, w, h, thick, col)
		drawLabel(out, b.Min.X+x, b.Min.Y+y, fmt.Sprintf("mask %.0f%%", mk.Conf*100), col)
	}

	return out
}

func boxToInts(bb [4]float64) (x, y, w, h int) {
	return int(bb[0] + 0.5), int(bb[1] + 0.5), int(bb[2] + 0.5), int(bb[3] + 0.5)
}

// overlayMask blends col at ~45% over the foreground pixels of a column-major RLE mask
// whose counts cover the whole image (H*W). h/w are taken from the image bounds.
func overlayMask(out *image.RGBA, rle string, b image.Rectangle, col color.RGBA) {
	if rle == "" {
		return
	}
	w, h := b.Dx(), b.Dy()
	const alpha = 0.45
	val := false
	idx := 0
	total := w * h
	for _, f := range strings.Fields(rle) {
		c, err := strconv.Atoi(f)
		if err != nil {
			return
		}
		for k := 0; k < c && idx < total; k++ {
			if val {
				x := idx / h // column-major: outer = column (x), inner = row (y)
				y := idx % h
				px := out.RGBAAt(b.Min.X+x, b.Min.Y+y)
				out.SetRGBA(b.Min.X+x, b.Min.Y+y, blend(px, col, alpha))
			}
			idx++
		}
		val = !val
	}
}

func blend(bg, fg color.RGBA, a float64) color.RGBA {
	return color.RGBA{
		R: uint8(float64(bg.R)*(1-a) + float64(fg.R)*a),
		G: uint8(float64(bg.G)*(1-a) + float64(fg.G)*a),
		B: uint8(float64(bg.B)*(1-a) + float64(fg.B)*a),
		A: 0xFF,
	}
}
