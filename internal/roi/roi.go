// Package roi implements the generic region-of-interest crop used by both the HTTP
// predict handler and the `run` CLI. Crop semantics: the model sees ONLY the ROI
// rectangle, and results are mapped back to original-image coordinates.
package roi

import (
	"image"
	"image/draw"
	"math"
	"strconv"
	"strings"

	"visionserve/internal/models"
	"visionserve/pkg/api"
)

// Parse parses an "x,y,w,h" ROI string into [4]float64 (zeros when absent/invalid).
func Parse(s string) [4]float64 {
	var roi [4]float64
	s = strings.TrimSpace(s)
	if s == "" {
		return roi
	}
	parts := strings.Split(s, ",")
	if len(parts) != 4 {
		return roi
	}
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return [4]float64{}
		}
		roi[i] = v
	}
	return roi
}

// Clamp converts a [x,y,w,h] ROI to an integer image.Rectangle clamped to the image,
// returning ok=false when the ROI is unset or degenerate (so the full image is used).
//
// The ROI may be given in PIXELS or NORMALIZED 0..1 fractions: when both w and h are ≤ 1
// it is treated as fractions and scaled by the image size. This makes the ROI independent
// of the resolution the client drew it on (avoids the "top-left right, bottom-right wrong"
// scaling mismatch between a display canvas and the actual image).
func Clamp(roi [4]float64, w, h int) (image.Rectangle, bool) {
	if roi[2] <= 0 || roi[3] <= 0 {
		return image.Rectangle{}, false
	}
	// Normalized (fractions) → pixels. Detected by w,h ≤ 1 (a real pixel ROI is larger).
	if roi[2] <= 1.0 && roi[3] <= 1.0 {
		roi = [4]float64{roi[0] * float64(w), roi[1] * float64(h), roi[2] * float64(w), roi[3] * float64(h)}
	}
	x0 := int(math.Round(roi[0]))
	y0 := int(math.Round(roi[1]))
	x1 := int(math.Round(roi[0] + roi[2]))
	y1 := int(math.Round(roi[1] + roi[3]))
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > w {
		x1 = w
	}
	if y1 > h {
		y1 = h
	}
	if x1-x0 < 1 || y1-y0 < 1 {
		return image.Rectangle{}, false
	}
	return image.Rect(x0, y0, x1, y1), true
}

// Crop copies the ROI rectangle of img into a fresh 0-origin RGBA image, so models
// (which often assume Bounds().Min == (0,0)) see a clean crop.
func Crop(img image.Image, rect image.Rectangle) image.Image {
	out := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	draw.Draw(out, out.Bounds(), img, rect.Min, draw.Src)
	return out
}

// ShiftPrompt translates box/point prompt coordinates from original-image space into the
// crop's space (subtract the ROI origin), so prompts line up with the cropped image. An
// external depth map (prompt.Depth, aligned to the full image) is cropped to the ROI too.
func ShiftPrompt(p *models.Prompt, rect image.Rectangle) {
	ox, oy := float64(rect.Min.X), float64(rect.Min.Y)
	for i := range p.Boxes {
		p.Boxes[i][0] -= ox
		p.Boxes[i][1] -= oy
	}
	for i := range p.Points {
		p.Points[i].X -= ox
		p.Points[i].Y -= oy
	}
	if p.Depth != nil && p.DepthW > 0 && p.DepthH > 0 && len(p.Depth) == p.DepthW*p.DepthH {
		cw, ch := rect.Dx(), rect.Dy()
		out := make([]float32, cw*ch)
		for y := 0; y < ch; y++ {
			srcRow := (y + rect.Min.Y) * p.DepthW
			dstRow := y * cw
			for x := 0; x < cw; x++ {
				out[dstRow+x] = p.Depth[srcRow+(x+rect.Min.X)]
			}
		}
		p.Depth, p.DepthW, p.DepthH = out, cw, ch
	}
}

// MapResult shifts crop-space results back into original-image coordinates: detection/mask
// bboxes and grasp centres are offset by the ROI origin, and each mask is re-embedded
// (decode at crop size → paste at the ROI offset → re-encode at full size).
func MapResult(res api.Result, rect image.Rectangle, fullW, fullH int) api.Result {
	ox, oy := float64(rect.Min.X), float64(rect.Min.Y)
	cw, ch := rect.Dx(), rect.Dy()

	for i := range res.Detections {
		res.Detections[i].BBox[0] += ox
		res.Detections[i].BBox[1] += oy
	}
	for i := range res.Grasps {
		res.Grasps[i].X += ox
		res.Grasps[i].Y += oy
	}
	for i := range res.Masks {
		crop := api.DecodeMaskRLE(res.Masks[i].RLE, cw, ch)
		full := make([]bool, fullW*fullH)
		for y := 0; y < ch; y++ {
			for x := 0; x < cw; x++ {
				if crop[y*cw+x] {
					full[(y+rect.Min.Y)*fullW+(x+rect.Min.X)] = true
				}
			}
		}
		res.Masks[i].RLE = api.EncodeMaskRLE(full, fullW, fullH)
		res.Masks[i].BBox[0] += ox
		res.Masks[i].BBox[1] += oy
	}
	return res
}
