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

// MasksPairDetections reports whether res.Masks is index-aligned with res.Detections: as many
// masks as detections, and mask i carrying detection i's box. That is how Grounded-SAM and the
// grasp models return one object (a detection and its mask), and how clients pair them
// (Result.group_by_class). Every step that filters or reshapes a paired result must keep it
// paired: one decision per object, applied to the detection and its mask alike.
func MasksPairDetections(res Result) bool {
	if len(res.Masks) == 0 || len(res.Masks) != len(res.Detections) {
		return false
	}
	for i := range res.Masks {
		if res.Masks[i].BBox != res.Detections[i].BBox {
			return false
		}
	}
	return true
}

// FilterBySize removes detections and masks whose bounding-box area (w*h, px²) is
// outside [minSize, maxSize]. Zero means no limit for that bound. A paired result
// (MasksPairDetections) stays paired: a mask carries its detection's box, so both get the same
// decision.
func FilterBySize(res Result, minSize, maxSize float64) Result {
	res.Detections = filterDetections(res.Detections, minSize, maxSize)
	res.Masks = filterMasks(res.Masks, minSize, maxSize)
	return res
}

func filterDetections(dets []Detection, minSize, maxSize float64) []Detection {
	if len(dets) == 0 {
		return dets
	}
	out := make([]Detection, 0, len(dets)) // fresh slice: never alias the caller's backing array
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
