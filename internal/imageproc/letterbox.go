// Package imageproc chứa tiện ích xử lý ảnh dùng chung, viết bằng Go thuần
// (github.com/disintegration/imaging + package image) để giữ binary gọn.
// KHÔNG dùng OpenCV/cgo ở đây (CLAUDE.md).
package imageproc

import (
	"image"
	"image/color"

	"github.com/disintegration/imaging"
)

// LetterboxResult chứa ảnh đã letterbox + thông tin để map ngược toạ độ.
// Quan hệ: input_coord = orig_coord * Scale + Pad.
type LetterboxResult struct {
	Img   *image.NRGBA
	Scale float64
	PadX  int
	PadY  int
}

// Letterbox resize ảnh giữ nguyên tỉ lệ vào khung WxH rồi pad cho đủ kích thước.
// padColor là màu nền vùng pad.
//
// TODO(verify): màu pad đúng cho RF-DETR cần xác nhận theo cách model được train/
// export (nhiều DETR pad bằng 0, một số pipeline dùng xám 114). Mặc định để (0,0,0).
func Letterbox(src image.Image, w, h int, padColor color.NRGBA) LetterboxResult {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()

	scale := float64(w) / float64(sw)
	if s := float64(h) / float64(sh); s < scale {
		scale = s
	}
	newW := int(float64(sw)*scale + 0.5)
	newH := int(float64(sh)*scale + 0.5)
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}

	resized := imaging.Resize(src, newW, newH, imaging.Linear)

	canvas := imaging.New(w, h, padColor)
	padX := (w - newW) / 2
	padY := (h - newH) / 2
	canvas = imaging.Paste(canvas, resized, image.Pt(padX, padY))

	return LetterboxResult{Img: canvas, Scale: scale, PadX: padX, PadY: padY}
}

// MapBoxToOriginal map một bbox [x,y,w,h] từ toạ độ ảnh đã letterbox về ảnh GỐC.
// Đây là bước hay sai nhất (CLAUDE.md) — tách riêng để test kỹ.
func (lb LetterboxResult) MapBoxToOriginal(x, y, w, h float64) (ox, oy, ow, oh float64) {
	ox = (x - float64(lb.PadX)) / lb.Scale
	oy = (y - float64(lb.PadY)) / lb.Scale
	ow = w / lb.Scale
	oh = h / lb.Scale
	return
}
