package imageproc

import (
	"image"

	"github.com/disintegration/imaging"
)

// Resize resize ảnh về đúng WxH (KHÔNG giữ tỉ lệ). Dùng cho model không letterbox.
// Khi cần giữ tỉ lệ, dùng Letterbox.
func Resize(src image.Image, w, h int) *image.NRGBA {
	return imaging.Resize(src, w, h, imaging.Linear)
}

// ResizeScale trả về tỉ lệ scale theo từng chiều (để map ngược toạ độ khi không letterbox).
func ResizeScale(origW, origH, dstW, dstH int) (scaleX, scaleY float64) {
	return float64(dstW) / float64(origW), float64(dstH) / float64(origH)
}
