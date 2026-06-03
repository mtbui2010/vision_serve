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

// DrawResult renders a full Result onto a copy of img.
// Branches by task:
//   - detection / segmentation / open_vocab: mask overlays + bbox labels (existing behaviour).
//   - classification / embed: top-K label text in top-left corner.
//   - depth: depth_map rendered as a turbo colormap image.
//
// Pure Go (no cgo), like the rest of imageproc.
func DrawResult(img image.Image, res api.Result) *image.RGBA {
	switch res.Task {
	case api.TaskClassification, api.TaskEmbed:
		return drawClassifications(img, res)
	case api.TaskDepth:
		return drawDepth(img, res)
	default:
		return drawDetectionsAndMasks(img, res)
	}
}

func drawDetectionsAndMasks(img image.Image, res api.Result) *image.RGBA {
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

// drawClassifications draws top-K classification labels as stacked text lines.
func drawClassifications(img image.Image, res api.Result) *image.RGBA {
	b := img.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, img, b.Min, draw.Src)

	const marginX, marginY, lineH = 10, 10, 22
	for i, c := range res.Classifications {
		col := palette[i%len(palette)]
		drawLabel(out, b.Min.X+marginX, b.Min.Y+marginY+i*lineH,
			fmt.Sprintf("%s %.0f%%", c.Class, c.Conf*100), col)
	}
	return out
}

// drawDepth renders res.DepthMap as a turbo-style colormap image at original bounds.
// Piecewise linear: t=0 blue, t=0.25 cyan, t=0.5 green, t=0.75 yellow, t=1 red.
func drawDepth(img image.Image, res api.Result) *image.RGBA {
	b := img.Bounds()
	out := image.NewRGBA(b)

	dm := res.DepthMap
	dw, dh := res.DepthWidth, res.DepthHeight
	if len(dm) == 0 || dw <= 0 || dh <= 0 {
		// No depth data — return blank copy of original.
		draw.Draw(out, b, img, b.Min, draw.Src)
		return out
	}

	// Normalise to [0, 1].
	minV, maxV := dm[0], dm[0]
	for _, v := range dm[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	rng := maxV - minV
	if rng == 0 {
		rng = 1
	}

	outW, outH := b.Dx(), b.Dy()
	for oy := 0; oy < outH; oy++ {
		dy := oy * dh / outH
		if dy >= dh {
			dy = dh - 1
		}
		for ox := 0; ox < outW; ox++ {
			dx := ox * dw / outW
			if dx >= dw {
				dx = dw - 1
			}
			t := float64(dm[dy*dw+dx]-minV) / float64(rng)
			out.SetRGBA(b.Min.X+ox, b.Min.Y+oy, turboColor(t))
		}
	}
	return out
}

// turboColor maps t ∈ [0,1] to a turbo-inspired RGB color.
func turboColor(t float64) color.RGBA {
	var r, g, bv float64
	switch {
	case t < 0.25:
		s := t * 4
		r, g, bv = 0, s, 1
	case t < 0.5:
		s := (t - 0.25) * 4
		r, g, bv = 0, 1, 1-s
	case t < 0.75:
		s := (t - 0.5) * 4
		r, g, bv = s, 1, 0
	default:
		s := (t - 0.75) * 4
		r, g, bv = 1, 1-s, 0
	}
	clamp := func(v float64) uint8 {
		if v < 0 {
			return 0
		}
		if v > 1 {
			return 255
		}
		return uint8(v*255 + 0.5)
	}
	return color.RGBA{R: clamp(r), G: clamp(g), B: clamp(bv), A: 0xFF}
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
