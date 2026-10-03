package scrfd

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)

// arch: SCRFD is fed the way InsightFace scrfd.py detect() feeds it — resize preserving aspect
// ratio so the image fits width × height (new size truncated with int(), as upstream), paste it
// at the TOP-LEFT corner of a black canvas (det_img[:nh, :nw] = resized — NOT a centred
// letterbox), then normalise with SCRFD's non-ImageNet formula:
//
//	pixel_normalized = (pixel - 127.5) / 128.0   (RGB order: blobFromImage swapRB=True)
//
// Placement matters: the anchors are a fixed grid over the input, so a centred pad shifts
// every face relative to the grid and changed the boxes (IoU vs the reference 0.90–0.95
// with a centred pad; see postprocess_test.go). top_left_pad is the only mode; the legacy
// `input.letterbox: true` of SCRFD manifests always meant it.
var arch = prep.Arch{Name: "scrfd", Modes: []prep.Mode{prep.TopLeftPad}}

// spec resolves the manifest's preprocessing. SCRFD's legacy input.normalize is written in
// 0..255 units ([127.5]*3 / [128]*3), so a legacy spec with a mean or std keeps pixels in 0..255
// (NoRescale) — the same arithmetic as before: (p/255 - mean/255) / (std/255). A `preprocess:`
// block says which units it uses itself (rescale: false for 0..255 units).
func spec(cfg models.Config) (prep.Spec, error) {
	s := cfg.PreprocessSpec()
	if s.Legacy && (len(s.Mean) > 0 || len(s.Std) > 0) {
		s.NoRescale = true
	}
	return arch.Resolve(s)
}

// preprocess: original image -> NCHW [1,3,H,W] (see arch). PreprocessMeta carries one
// det_scale for both axes, as upstream, and no pad offset (the image sits at the top-left).
func preprocess(img image.Image, cfg models.Config) (engine.Tensor, models.PreprocessMeta, error) {
	s, err := spec(cfg)
	if err != nil {
		return engine.Tensor{}, models.PreprocessMeta{}, err
	}
	return s.Apply(img)
}
