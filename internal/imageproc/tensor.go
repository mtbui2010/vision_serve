package imageproc

import (
	"image"

	"visionserve/internal/engine"
)

// ImageToCHWFloat converts an image into a float32 tensor with NCHW layout [1,3,H,W],
// normalized by mean/std (ImageNet-style): v' = (v/255 - mean[c]) / std[c].
// If mean/std are empty -> only divide by 255 (mapping to [0,1]).
//
// Channel order: RGB (channel 0=R, 1=G, 2=B).
func ImageToCHWFloat(img image.Image, mean, std []float32) engine.Tensor {
	nrgba := toNRGBA(img)
	b := nrgba.Bounds()
	w, h := b.Dx(), b.Dy()

	data := make([]float32, 3*h*w)
	plane := h * w
	stride := nrgba.Stride
	pix := nrgba.Pix

	getMean := func(c int) float32 {
		if c < len(mean) {
			return mean[c]
		}
		return 0
	}
	getStd := func(c int) float32 {
		if c < len(std) && std[c] != 0 {
			return std[c]
		}
		return 1
	}
	mr, mg, mb := getMean(0), getMean(1), getMean(2)
	sr, sg, sb := getStd(0), getStd(1), getStd(2)

	for y := 0; y < h; y++ {
		row := y * stride
		for x := 0; x < w; x++ {
			i := row + x*4
			r := float32(pix[i]) / 255.0
			g := float32(pix[i+1]) / 255.0
			bl := float32(pix[i+2]) / 255.0
			idx := y*w + x
			data[idx] = (r - mr) / sr          // R plane
			data[plane+idx] = (g - mg) / sg    // G plane
			data[2*plane+idx] = (bl - mb) / sb // B plane
		}
	}
	return engine.Tensor{Data: data, Shape: []int64{1, 3, int64(h), int64(w)}}
}

func toNRGBA(img image.Image) *image.NRGBA {
	if n, ok := img.(*image.NRGBA); ok {
		return n
	}
	b := img.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			dst.Set(x, y, img.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	return dst
}
