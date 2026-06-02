package imageproc

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"visionserve/pkg/api"
)

// Package imageproc uses pure Go (no cgo) — overlay drawing too: image/draw +
// basicfont (an embedded bitmap font, no external font file needed).

// palette is a set of distinct colors, assigned by detection index (cycling).
var palette = []color.RGBA{
	{0xE6, 0x19, 0x4B, 0xFF}, // red
	{0x3C, 0xB4, 0x4B, 0xFF}, // green
	{0x43, 0x63, 0xD8, 0xFF}, // blue
	{0xF5, 0x82, 0x31, 0xFF}, // orange
	{0x91, 0x1E, 0xB4, 0xFF}, // purple
	{0x46, 0xF0, 0xF0, 0xFF}, // cyan
	{0xF0, 0x32, 0xE6, 0xFF}, // magenta
	{0xBF, 0xEF, 0x45, 0xFF}, // chartreuse
	{0xFA, 0xBE, 0xD4, 0xFF}, // pink
	{0x00, 0x80, 0x80, 0xFF}, // teal
}

// DrawDetections draws bboxes + "class conf%" labels onto an RGBA COPY of img.
// BBox is in ORIGINAL image coordinates [x,y,w,h] (matching the api.Detection schema) — drawn directly.
func DrawDetections(img image.Image, dets []api.Detection) *image.RGBA {
	b := img.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, img, b.Min, draw.Src)

	// border thickness scales slightly with image size for visibility on large images.
	thick := b.Dx() / 400
	if thick < 2 {
		thick = 2
	}

	for i, d := range dets {
		col := palette[i%len(palette)]
		x := int(d.BBox[0] + 0.5)
		y := int(d.BBox[1] + 0.5)
		w := int(d.BBox[2] + 0.5)
		h := int(d.BBox[3] + 0.5)
		drawRectOutline(out, b.Min.X+x, b.Min.Y+y, w, h, thick, col)
		label := fmt.Sprintf("%s %.0f%%", d.Class, d.Conf*100)
		drawLabel(out, b.Min.X+x, b.Min.Y+y, label, col)
	}
	return out
}

// fillRect fills a solid rectangle [x0,y0)-[x1,y1), clamped to the image bounds.
func fillRect(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	b := img.Bounds()
	if x0 < b.Min.X {
		x0 = b.Min.X
	}
	if y0 < b.Min.Y {
		y0 = b.Min.Y
	}
	if x1 > b.Max.X {
		x1 = b.Max.X
	}
	if y1 > b.Max.Y {
		y1 = b.Max.Y
	}
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			img.Set(x, y, c)
		}
	}
}

// drawRectOutline draws a rectangle outline t px thick (4 edge strips).
func drawRectOutline(img *image.RGBA, x, y, w, h, t int, c color.Color) {
	if w <= 0 || h <= 0 {
		return
	}
	fillRect(img, x, y, x+w, y+t, c)     // top
	fillRect(img, x, y+h-t, x+w, y+h, c) // bottom
	fillRect(img, x, y, x+t, y+h, c)     // left
	fillRect(img, x+w-t, y, x+w, y+h, c) // right
}

// drawLabel draws a box-colored background + contrasting text, placed above the box edge (or inside
// it if it would overflow the image).
func drawLabel(img *image.RGBA, x, y int, label string, boxColor color.RGBA) {
	face := basicfont.Face7x13
	const pad = 2
	tw := font.MeasureString(face, label).Ceil()
	asc := face.Metrics().Ascent.Ceil()
	bgW := tw + 2*pad
	bgH := face.Metrics().Height.Ceil() + 2*pad

	ly := y - bgH // by default place above the box edge
	if ly < img.Bounds().Min.Y {
		ly = y // not enough room above -> place just inside the top edge
	}
	fillRect(img, x, ly, x+bgW, ly+bgH, boxColor)

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(contrast(boxColor)),
		Face: face,
		Dot:  fixed.P(x+pad, ly+pad+asc),
	}
	d.DrawString(label)
}

// contrast picks black/white text based on background brightness (Rec.601 luminance).
func contrast(c color.RGBA) color.RGBA {
	lum := 0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B)
	if lum > 140 {
		return color.RGBA{0, 0, 0, 0xFF}
	}
	return color.RGBA{0xFF, 0xFF, 0xFF, 0xFF}
}
