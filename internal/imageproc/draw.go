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

// Package imageproc dùng Go thuần (không cgo) — vẽ overlay cũng vậy: image/draw +
// basicfont (bitmap font nhúng sẵn, không cần file font ngoài).

// palette là bộ màu phân biệt, gán theo index detection (xoay vòng).
var palette = []color.RGBA{
	{0xE6, 0x19, 0x4B, 0xFF}, // đỏ
	{0x3C, 0xB4, 0x4B, 0xFF}, // xanh lá
	{0x43, 0x63, 0xD8, 0xFF}, // xanh dương
	{0xF5, 0x82, 0x31, 0xFF}, // cam
	{0x91, 0x1E, 0xB4, 0xFF}, // tím
	{0x46, 0xF0, 0xF0, 0xFF}, // cyan
	{0xF0, 0x32, 0xE6, 0xFF}, // magenta
	{0xBF, 0xEF, 0x45, 0xFF}, // chartreuse
	{0xFA, 0xBE, 0xD4, 0xFF}, // hồng
	{0x00, 0x80, 0x80, 0xFF}, // teal
}

// DrawDetections vẽ bbox + nhãn "class conf%" lên một BẢN SAO RGBA của img.
// BBox theo toạ độ ảnh GỐC [x,y,w,h] (đúng schema api.Detection) — vẽ trực tiếp.
func DrawDetections(img image.Image, dets []api.Detection) *image.RGBA {
	b := img.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, img, b.Min, draw.Src)

	// độ dày viền co giãn nhẹ theo kích thước ảnh để nhìn rõ trên ảnh lớn.
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

// fillRect tô đặc hình chữ nhật [x0,y0)-[x1,y1), clamp vào biên ảnh.
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

// drawRectOutline vẽ viền chữ nhật dày t px (4 dải cạnh).
func drawRectOutline(img *image.RGBA, x, y, w, h, t int, c color.Color) {
	if w <= 0 || h <= 0 {
		return
	}
	fillRect(img, x, y, x+w, y+t, c)     // trên
	fillRect(img, x, y+h-t, x+w, y+h, c) // dưới
	fillRect(img, x, y, x+t, y+h, c)     // trái
	fillRect(img, x+w-t, y, x+w, y+h, c) // phải
}

// drawLabel vẽ nền màu box + chữ tương phản, đặt phía trên mép box (hoặc bên trong
// nếu tràn ra ngoài ảnh).
func drawLabel(img *image.RGBA, x, y int, label string, boxColor color.RGBA) {
	face := basicfont.Face7x13
	const pad = 2
	tw := font.MeasureString(face, label).Ceil()
	asc := face.Metrics().Ascent.Ceil()
	bgW := tw + 2*pad
	bgH := face.Metrics().Height.Ceil() + 2*pad

	ly := y - bgH // mặc định đặt trên mép box
	if ly < img.Bounds().Min.Y {
		ly = y // không đủ chỗ phía trên -> đặt ngay trong mép trên
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

// contrast chọn đen/trắng cho chữ theo độ sáng nền (luminance Rec.601).
func contrast(c color.RGBA) color.RGBA {
	lum := 0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B)
	if lum > 140 {
		return color.RGBA{0, 0, 0, 0xFF}
	}
	return color.RGBA{0xFF, 0xFF, 0xFF, 0xFF}
}
