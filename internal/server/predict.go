package server

import (
	"context"
	"image"

	"visionserve/internal/models"
	"visionserve/internal/morph"
	roipkg "visionserve/internal/roi"
	"visionserve/pkg/api"
)

// Predictor runs a prompted prediction; lifecycle.Manager is one.
type Predictor interface {
	PredictPrompt(ctx context.Context, name string, img image.Image, prompt models.Prompt) (api.Result, error)
}

// Predict runs one prediction the way POST /api/predict does: the model sees only the ROI crop
// (prompt shifted into it), results are mapped back to ORIGINAL image coordinates, masks get the
// requested morphology, then the bbox-area filter applies. ctx is checked right before inference
// and the runtime follows it while the request waits (load, session, lock): a request whose client
// has gone does not run (errClientGone). `visionserve run` calls it too, so the CLI and the API
// cannot drift apart.
func Predict(ctx context.Context, p Predictor, model string, img image.Image, prompt models.Prompt) (api.Result, error) {
	fullW, fullH := img.Bounds().Dx(), img.Bounds().Dy()
	img, rect, hasROI := cropROI(img, &prompt)
	if ctx.Err() != nil {
		return api.Result{}, errClientGone
	}
	res, err := p.PredictPrompt(ctx, model, img, prompt)
	if err != nil {
		return api.Result{}, orClientGone(ctx, err)
	}
	if hasROI {
		res = roipkg.MapResult(res, rect, fullW, fullH)
	}
	// Mask morphology (enlarge/shrink) in ORIGINAL-image terms, then size filter. A mask paired
	// with a detection keeps its box, so the filter decides once per object.
	morph.ApplyToResult(&res, fullW, fullH, prompt.Dilate)
	if prompt.MinSize > 0 || prompt.MaxSize > 0 {
		res = api.FilterBySizePct(res, prompt.MinSize, prompt.MaxSize, fullW, fullH)
	}
	return res, nil
}

// cropROI applies the prompt's region of interest (generic, crop semantics): it returns the crop
// the model should see and shifts the prompt (boxes, points, depth) into crop coordinates.
// ok=false (img unchanged) when the request has no usable ROI.
func cropROI(img image.Image, prompt *models.Prompt) (image.Image, image.Rectangle, bool) {
	rect, ok := roipkg.Clamp(prompt.ROI, img.Bounds().Dx(), img.Bounds().Dy())
	if !ok {
		return img, rect, false
	}
	roipkg.ShiftPrompt(prompt, rect)
	return roipkg.Crop(img, rect), rect, true
}
