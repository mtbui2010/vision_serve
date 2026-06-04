package api

// FilterBySizePct filters detections and masks by bounding-box area expressed as a
// percentage of the total image area (0–100). Zero means no limit for that bound.
// Example: minPct=0.1 keeps only boxes/masks covering at least 0.1% of the image.
func FilterBySizePct(res Result, minPct, maxPct float64, imgW, imgH int) Result {
	area := float64(imgW * imgH)
	var minAbs, maxAbs float64
	if minPct > 0 {
		minAbs = minPct / 100.0 * area
	}
	if maxPct > 0 {
		maxAbs = maxPct / 100.0 * area
	}
	return FilterBySize(res, minAbs, maxAbs)
}

// FilterBySize removes detections and masks whose bounding-box area (w*h, px²) is
// outside [minSize, maxSize]. Zero means no limit for that bound.
func FilterBySize(res Result, minSize, maxSize float64) Result {
	res.Detections = filterDetections(res.Detections, minSize, maxSize)
	res.Masks = filterMasks(res.Masks, minSize, maxSize)
	return res
}

func filterDetections(dets []Detection, minSize, maxSize float64) []Detection {
	if len(dets) == 0 {
		return dets
	}
	out := dets[:0:0] // empty slice, same backing array avoided — allocate fresh
	out = make([]Detection, 0, len(dets))
	for _, d := range dets {
		area := d.BBox[2] * d.BBox[3]
		if minSize > 0 && area < minSize {
			continue
		}
		if maxSize > 0 && area > maxSize {
			continue
		}
		out = append(out, d)
	}
	return out
}

func filterMasks(masks []Mask, minSize, maxSize float64) []Mask {
	if len(masks) == 0 {
		return masks
	}
	out := make([]Mask, 0, len(masks))
	for _, m := range masks {
		area := m.BBox[2] * m.BBox[3]
		if minSize > 0 && area < minSize {
			continue
		}
		if maxSize > 0 && area > maxSize {
			continue
		}
		out = append(out, m)
	}
	return out
}
