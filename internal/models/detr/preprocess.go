package detr

import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/vision/preprocess"
)

// arch is what a DETR manifest may declare. squash (the default) is how rfdetr trains
// (square_resize_div_64=True, and its own predict() is F.resize(img, [res, res]) with no
// padding), so RF-DETR manifests squash. Serving a squash-trained checkpoint letterboxed
// measured -7.35 mAP on the fine-tuned 512 detectors and -1.91 on COCO base (BUGS_TO_FIX.md #1).
// letterbox is kept for exports that really were trained aspect-preserving (the RT-DETR
// manifest letterboxes); pad is black. The decoder maps boxes back through the Meta, so a mode
// that crops (center_crop) or changes the input size (keep_aspect) is refused.
func arch(prefix string) preprocess.Arch {
	return preprocess.Arch{Name: prefix, Modes: []preprocess.Mode{preprocess.Squash, preprocess.Letterbox}}
}

// spec resolves the manifest's preprocessing for this model.
func (m *detr) spec() (preprocess.Spec, error) {
	return arch(m.v.prefix).Resolve(m.cfg.PreprocessSpec())
}

// preprocess: original image -> NCHW [1,3,H,W] tensor (cfg.Width×cfg.Height), resized +
// normalized. PreprocessMeta stores scale/pad so postprocess can map boxes back to
// ORIGINAL image coords.
func (m *detr) preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	s, err := m.spec()
	if err != nil {
		return engine.Tensor{}, models.PreprocessMeta{}, err
	}
	return s.Apply(img)
}
