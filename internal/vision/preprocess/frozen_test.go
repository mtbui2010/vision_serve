package preprocess_test

// FROZEN copies of every hand-written preprocessing path this package replaced, verbatim from
// the tree before the migration (refactor/2026-10 @ b246f6c). equiv_test.go runs each one and
// the Spec that replaces it on many images and requires BIT-IDENTICAL tensors and identical
// Meta. Do not "fix" anything here: these are the reference, bugs and all.

import (
	"image"
	"image/color"
	"math"

	"github.com/disintegration/imaging"

	"visionserve/internal/engine"
	"visionserve/internal/vision/preprocess"
)

// ---- internal/imageproc/tensor.go ----------------------------------------------------------

func oldImageToCHWFloat(img image.Image, mean, std []float32) engine.Tensor {
	nrgba := oldToNRGBA(img)
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
			data[idx] = (r - mr) / sr
			data[plane+idx] = (g - mg) / sg
			data[2*plane+idx] = (bl - mb) / sb
		}
	}
	return engine.Tensor{Data: data, Shape: []int64{1, 3, int64(h), int64(w)}}
}

func oldToNRGBA(img image.Image) *image.NRGBA {
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

// ---- internal/imageproc/letterbox.go, resize.go, keepaspect.go -----------------------------

type oldLetterboxResult struct {
	Img   *image.NRGBA
	Scale float64
	PadX  int
	PadY  int
}

func oldLetterbox(src image.Image, w, h int, padColor color.NRGBA) oldLetterboxResult {
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

	return oldLetterboxResult{Img: canvas, Scale: scale, PadX: padX, PadY: padY}
}

func oldResize(src image.Image, w, h int) *image.NRGBA {
	return imaging.Resize(src, w, h, imaging.Linear)
}

func oldResizeBicubic(src image.Image, w, h int) *image.NRGBA {
	return imaging.Resize(src, w, h, imaging.CatmullRom)
}

func oldResizeScale(origW, origH, dstW, dstH int) (scaleX, scaleY float64) {
	return float64(dstW) / float64(origW), float64(dstH) / float64(origH)
}

func oldResizeShortCenterCrop(src image.Image, w, h int) (*image.NRGBA, float64, float64, int, int) {
	b := src.Bounds()
	ow, oh := b.Dx(), b.Dy()
	var rw, rh int
	if ow <= oh {
		rw = w
		rh = int(float64(w) * float64(oh) / float64(ow))
	} else {
		rh = h
		rw = int(float64(h) * float64(ow) / float64(oh))
	}
	if rw < w {
		rw = w
	}
	if rh < h {
		rh = h
	}
	resized := imaging.Resize(src, rw, rh, imaging.CatmullRom)
	offX, offY := (rw-w)/2, (rh-h)/2
	crop := imaging.Crop(resized, image.Rect(offX, offY, offX+w, offY+h))
	return crop, float64(rw) / float64(ow), float64(rh) / float64(oh), offX, offY
}

func oldDPTKeepAspectSize(w, h, tw, th, multiple int) (nw, nh int) {
	if multiple <= 0 {
		multiple = 1
	}
	if w <= 0 || h <= 0 {
		return tw, th
	}
	scaleW := float64(tw) / float64(w)
	scaleH := float64(th) / float64(h)
	if math.Abs(1-scaleW) < math.Abs(1-scaleH) {
		scaleH = scaleW
	} else {
		scaleW = scaleH
	}
	c := func(v float64, m int) int {
		x := int(math.RoundToEven(v/float64(m))) * m
		if x < m {
			x = m
		}
		return x
	}
	return c(scaleW*float64(w), multiple), c(scaleH*float64(h), multiple)
}

// ---- the per-model preprocess functions (internal/models/<arch>/preprocess.go) --------------

// legacyCfg is the subset of models.Config those functions read.
type legacyCfg struct {
	Width, Height int
	Mean, Std     []float32
	Letterbox     bool
	Crop          string
	KeepAspect    bool
	MultipleOf    int
}

// detr (rf-detr / rt-detr).
func oldDETR(img image.Image, cfg legacyCfg) (engine.Tensor, preprocess.Meta) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	if cfg.Letterbox {
		lb := oldLetterbox(img, cfg.Width, cfg.Height, color.NRGBA{0, 0, 0, 255})
		meta := preprocess.Meta{OrigWidth: origW, OrigHeight: origH, ScaleX: lb.Scale, ScaleY: lb.Scale, PadX: lb.PadX, PadY: lb.PadY}
		return oldImageToCHWFloat(lb.Img, cfg.Mean, cfg.Std), meta
	}
	processed := oldResize(img, cfg.Width, cfg.Height)
	sx, sy := oldResizeScale(origW, origH, cfg.Width, cfg.Height)
	meta := preprocess.Meta{OrigWidth: origW, OrigHeight: origH, ScaleX: sx, ScaleY: sy, PadX: 0, PadY: 0}
	return oldImageToCHWFloat(processed, cfg.Mean, cfg.Std), meta
}

// classification (efficientnet / mobilenet-v3).
func oldClassification(img image.Image, cfg legacyCfg) (engine.Tensor, preprocess.Meta) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	resized := oldResize(img, cfg.Width, cfg.Height)
	scaleX, scaleY := oldResizeScale(origW, origH, cfg.Width, cfg.Height)
	meta := preprocess.Meta{OrigWidth: origW, OrigHeight: origH, ScaleX: scaleX, ScaleY: scaleY}
	return oldImageToCHWFloat(resized, cfg.Mean, cfg.Std), meta
}

// depth (midas / depth-anything-v2).
func oldDepth(img image.Image, cfg legacyCfg) (engine.Tensor, preprocess.Meta) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	w, h := cfg.Width, cfg.Height
	var resized *image.NRGBA
	if cfg.KeepAspect {
		w, h = oldDPTKeepAspectSize(origW, origH, cfg.Width, cfg.Height, cfg.MultipleOf)
		resized = oldResizeBicubic(img, w, h)
	} else {
		resized = oldResize(img, w, h)
	}
	scaleX, scaleY := oldResizeScale(origW, origH, w, h)
	meta := preprocess.Meta{OrigWidth: origW, OrigHeight: origH, ScaleX: scaleX, ScaleY: scaleY}
	return oldImageToCHWFloat(resized, cfg.Mean, cfg.Std), meta
}

var (
	oldCLIPMean = []float32{0.48145466, 0.4578275, 0.40821073}
	oldCLIPStd  = []float32{0.26862954, 0.26130258, 0.27577711}
)

// clip (image tower).
func oldCLIP(img image.Image, cfg legacyCfg) (engine.Tensor, preprocess.Meta) {
	pick := func(v, def []float32) []float32 {
		if len(v) == 0 {
			return def
		}
		return v
	}
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	w, h := cfg.Width, cfg.Height
	if w <= 0 {
		w = 224
	}
	if h <= 0 {
		h = 224
	}
	if cfg.Crop == "center" {
		cropped, sx, sy, offX, offY := oldResizeShortCenterCrop(img, w, h)
		meta := preprocess.Meta{OrigWidth: origW, OrigHeight: origH, ScaleX: sx, ScaleY: sy, PadX: -offX, PadY: -offY}
		return oldImageToCHWFloat(cropped, pick(cfg.Mean, oldCLIPMean), pick(cfg.Std, oldCLIPStd)), meta
	}
	resized := oldResize(img, w, h)
	mean, std := cfg.Mean, cfg.Std
	if len(mean) == 0 {
		mean = oldCLIPMean
	}
	if len(std) == 0 {
		std = oldCLIPStd
	}
	scaleX, scaleY := oldResizeScale(origW, origH, w, h)
	meta := preprocess.Meta{OrigWidth: origW, OrigHeight: origH, ScaleX: scaleX, ScaleY: scaleY}
	return oldImageToCHWFloat(resized, mean, std), meta
}

// scrfd.
func oldSCRFD(img image.Image, cfg legacyCfg) (engine.Tensor, preprocess.Meta) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	fit := func(w, h, W, H int) (newW, newH int, scale float64) {
		imRatio := float64(h) / float64(w)
		modelRatio := float64(H) / float64(W)
		if imRatio > modelRatio {
			newH = H
			newW = int(float64(newH) / imRatio)
		} else {
			newW = W
			newH = int(float64(newW) * imRatio)
		}
		if newW < 1 {
			newW = 1
		}
		if newH < 1 {
			newH = 1
		}
		return newW, newH, float64(newH) / float64(h)
	}
	to01 := func(vals []float32) []float32 {
		if len(vals) == 0 {
			return vals
		}
		out := make([]float32, len(vals))
		for i, v := range vals {
			out[i] = v / 255.0
		}
		return out
	}
	newW, newH, scale := fit(origW, origH, cfg.Width, cfg.Height)
	resized := imaging.Resize(img, newW, newH, imaging.Linear)
	canvas := imaging.New(cfg.Width, cfg.Height, color.NRGBA{0, 0, 0, 255})
	canvas = imaging.Paste(canvas, resized, image.Pt(0, 0))
	tensor := oldImageToCHWFloat(canvas, to01(cfg.Mean), to01(cfg.Std))
	meta := preprocess.Meta{OrigWidth: origW, OrigHeight: origH, ScaleX: scale, ScaleY: scale}
	return tensor, meta
}

// mobilesam encoder: long side -> 1024, raw HWC [newH, newW, 3]; returns the prompt scale.
func oldMobileSAM(img image.Image) (engine.Tensor, float64) {
	const encoderSize = 1024
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	scale := float64(encoderSize) / float64(max(origW, origH))
	newW := int(math.Round(float64(origW) * scale))
	newH := int(math.Round(float64(origH) * scale))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}

	resized := imaging.Resize(img, newW, newH, imaging.Linear)

	data := make([]float32, newH*newW*3)
	i := 0
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			data[i] = float32(c.R)
			data[i+1] = float32(c.G)
			data[i+2] = float32(c.B)
			i += 3
		}
	}
	return engine.F32(data, int64(newH), int64(newW), 3), scale
}

var (
	oldImagenetMean = [3]float32{0.485, 0.456, 0.406}
	oldImagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// nanosam encoder: long side -> 1024 (float32 scale!), ImageNet, zero pad bottom/right, NCHW.
func oldNanoSAM(img image.Image) (engine.Tensor, float32) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	const encoderSize = 1024
	scale := float32(encoderSize) / float32(max(origW, origH))
	newW := int(math.Round(float64(origW) * float64(scale)))
	newH := int(math.Round(float64(origH) * float64(scale)))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}

	resized := imaging.Resize(img, newW, newH, imaging.Linear)
	data := make([]float32, 3*encoderSize*encoderSize)
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			r := (float32(c.R)/255.0 - oldImagenetMean[0]) / oldImagenetStd[0]
			g := (float32(c.G)/255.0 - oldImagenetMean[1]) / oldImagenetStd[1]
			bv := (float32(c.B)/255.0 - oldImagenetMean[2]) / oldImagenetStd[2]
			data[0*encoderSize*encoderSize+y*encoderSize+x] = r
			data[1*encoderSize*encoderSize+y*encoderSize+x] = g
			data[2*encoderSize*encoderSize+y*encoderSize+x] = bv
		}
	}
	return engine.F32(data, 1, 3, encoderSize, encoderSize), scale
}

// sam2 encoder: squash 1024, ImageNet, NCHW; per-axis float32 prompt scales.
func oldSAM2(img image.Image) (engine.Tensor, float32, float32) {
	const encoderSize = 1024
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	resized := imaging.Resize(img, encoderSize, encoderSize, imaging.Linear)
	const plane = encoderSize * encoderSize
	data := make([]float32, 3*plane)
	for y := 0; y < encoderSize; y++ {
		row := resized.Pix[y*resized.Stride : y*resized.Stride+encoderSize*4]
		for x := 0; x < encoderSize; x++ {
			p := row[x*4 : x*4+3]
			i := y*encoderSize + x
			data[i] = (float32(p[0])/255.0 - oldImagenetMean[0]) / oldImagenetStd[0]
			data[plane+i] = (float32(p[1])/255.0 - oldImagenetMean[1]) / oldImagenetStd[1]
			data[2*plane+i] = (float32(p[2])/255.0 - oldImagenetMean[2]) / oldImagenetStd[2]
		}
	}
	return engine.F32(data, 1, 3, int64(encoderSize), int64(encoderSize)),
		float32(encoderSize) / float32(origW), float32(encoderSize) / float32(origH)
}

// efficientsam encoder: the ORIGINAL image as NCHW pixel/255.
func oldEfficientSAM(img image.Image) engine.Tensor { return oldImageToCHWFloat(img, nil, nil) }

// paddleocr det: long side <= maxSide (never upscale), ImageNet, zero pad up to multiples of 32.
func oldPaddleDet(img image.Image, maxSide int) (engine.Tensor, preprocess.Meta, int, int) {
	const detGridSize = 32
	b := img.Bounds()
	origW := b.Dx()
	origH := b.Dy()
	scaleX := float64(maxSide) / float64(origW)
	scaleY := float64(maxSide) / float64(origH)
	scale := math.Min(scaleX, scaleY)
	if scale > 1.0 {
		scale = 1.0
	}
	newW := int(math.Round(float64(origW) * scale))
	newH := int(math.Round(float64(origH) * scale))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}
	padW := ((newW + detGridSize - 1) / detGridSize) * detGridSize
	padH := ((newH + detGridSize - 1) / detGridSize) * detGridSize
	resized := imaging.Resize(img, newW, newH, imaging.Linear)
	plane := padW * padH
	data := make([]float32, 3*plane)
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			idx := y*padW + x
			data[idx] = (float32(c.R)/255.0 - oldImagenetMean[0]) / oldImagenetStd[0]
			data[plane+idx] = (float32(c.G)/255.0 - oldImagenetMean[1]) / oldImagenetStd[1]
			data[2*plane+idx] = (float32(c.B)/255.0 - oldImagenetMean[2]) / oldImagenetStd[2]
		}
	}
	meta := preprocess.Meta{OrigWidth: origW, OrigHeight: origH, ScaleX: scale, ScaleY: scale}
	return engine.F32(data, 1, 3, int64(padH), int64(padW)), meta, padW, padH
}

// paddleocr rec: the normalisation loop over the resized text-line crop.
func oldPaddleRecNormalize(resized *image.NRGBA, targetW, recHeight int) engine.Tensor {
	recNormMean := [3]float32{0.5, 0.5, 0.5}
	recNormStd := [3]float32{0.5, 0.5, 0.5}
	plane := recHeight * targetW
	data := make([]float32, 3*plane)
	for y := 0; y < recHeight; y++ {
		for x := 0; x < targetW; x++ {
			c := resized.NRGBAAt(resized.Bounds().Min.X+x, resized.Bounds().Min.Y+y)
			idx := y*targetW + x
			data[idx] = (float32(c.R)/255.0 - recNormMean[0]) / recNormStd[0]
			data[plane+idx] = (float32(c.G)/255.0 - recNormMean[1]) / recNormStd[1]
			data[2*plane+idx] = (float32(c.B)/255.0 - recNormMean[2]) / recNormStd[2]
		}
	}
	return engine.F32(data, 1, 3, int64(recHeight), int64(targetW))
}
