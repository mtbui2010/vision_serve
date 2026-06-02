package rfdetr

import (
	"image"
	"image/color"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
)

// preprocess: ảnh gốc -> tensor NCHW [1,3,H,W] đã letterbox + normalize.
// PreprocessMeta lưu scale/pad để postprocess map box về toạ độ ảnh GỐC.
func (m *rfDETR) preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	var lb imageproc.LetterboxResult
	var processed image.Image

	if m.cfg.Letterbox {
		// TODO(verify): màu pad đúng theo cách RF-DETR được train/export. Mặc định đen.
		lb = imageproc.Letterbox(img, m.cfg.Width, m.cfg.Height, color.NRGBA{0, 0, 0, 255})
		processed = lb.Img
	} else {
		processed = imageproc.Resize(img, m.cfg.Width, m.cfg.Height)
		sx, sy := imageproc.ResizeScale(origW, origH, m.cfg.Width, m.cfg.Height)
		lb = imageproc.LetterboxResult{Scale: sx, PadX: 0, PadY: 0}
		// khi không letterbox, scale theo 2 chiều có thể khác nhau
		meta := models.PreprocessMeta{
			OrigWidth: origW, OrigHeight: origH,
			ScaleX: sx, ScaleY: sy, PadX: 0, PadY: 0,
		}
		return imageproc.ImageToCHWFloat(processed, m.cfg.Mean, m.cfg.Std), meta, nil
	}

	meta := models.PreprocessMeta{
		OrigWidth:  origW,
		OrigHeight: origH,
		ScaleX:     lb.Scale,
		ScaleY:     lb.Scale, // letterbox giữ tỉ lệ -> scale 2 chiều bằng nhau
		PadX:       lb.PadX,
		PadY:       lb.PadY,
	}
	return imageproc.ImageToCHWFloat(processed, m.cfg.Mean, m.cfg.Std), meta, nil
}
