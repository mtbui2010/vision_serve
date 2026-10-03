package paddleocr

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/vision/mask"
)

const (
	defaultDetThresh = 0.3 // probability map threshold
	// defaultUnclipRatio is PaddleOCR DBPostProcess's unclip_ratio. The DB head predicts a
	// SHRUNK text kernel; "unclip" grows it back by offsetting the polygon outward by
	// d = area * ratio / perimeter (pyclipper offset). For the axis-aligned box we extract,
	// that is an expansion of EVERY side by d — NOT a scale of the box about its centre
	// (which over-grows long lines horizontally and under-grows them vertically).
	defaultUnclipRatio = 1.5
	minComponentSize   = 16 // skip noise components smaller than this (pixels)
)

// point is a 2D pixel coordinate used in the BFS flood-fill.
type point struct{ x, y int }

// unclipDistance is the DB offset distance for an axis-aligned w×h box:
// area*ratio/perimeter = w*h*ratio / (2*(w+h)). Zero for a degenerate box.
func unclipDistance(w, h, ratio float64) float64 {
	per := 2 * (w + h)
	if per <= 0 {
		return 0
	}
	return w * h * ratio / per
}

// extractBBoxes thresholds the [1,1,H,W] probability map at thresh, finds connected
// components via iterative BFS (8-connectivity), returns axis-aligned bounding boxes as
// [x, y, w, h] in the detection model's input space (before mapping to original coords).
// Each box is unclipped per PaddleOCR DBPostProcess: every side is pushed outward by
// d = area*unclipRatio/perimeter, with the box measured on pixel centres (as cv2 contours /
// minAreaRect see it). Returns an error if probMap does not hold exactly h*w values.
func extractBBoxes(probMap []float32, h, w int, thresh, unclipRatio float64) ([][4]float64, error) {
	if h <= 0 || w <= 0 {
		return nil, fmt.Errorf("paddleocr: invalid probability map size %dx%d", w, h)
	}
	if len(probMap) != h*w {
		return nil, fmt.Errorf("paddleocr: probability map has %d values, want %d (%dx%d)", len(probMap), h*w, w, h)
	}

	// Binary text mask: float64(p) > thresh.
	bm, _ := mask.Threshold(probMap, 0, h, w, thresh)
	fg := bm.Data

	// visited is set when a pixel is ENQUEUED, so each pixel enters the queue at most once
	// (marking on dequeue let one pixel be queued by up to 8 neighbours). The queue is
	// indexed rather than re-sliced (queue[1:] keeps the whole backing array alive) and is
	// reused across components.
	visited := make([]bool, h*w)
	var result [][4]float64
	var queue []point

	for startY := 0; startY < h; startY++ {
		for startX := 0; startX < w; startX++ {
			idx := startY*w + startX
			if !fg[idx] || visited[idx] {
				continue
			}

			// BFS flood-fill (iterative to avoid stack overflow on large images).
			visited[idx] = true
			queue = append(queue[:0], point{startX, startY})
			minX, minY, maxX, maxY := startX, startY, startX, startY

			for head := 0; head < len(queue); head++ {
				p := queue[head]
				if p.x < minX {
					minX = p.x
				}
				if p.x > maxX {
					maxX = p.x
				}
				if p.y < minY {
					minY = p.y
				}
				if p.y > maxY {
					maxY = p.y
				}

				// Check 8 neighbors.
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						if dx == 0 && dy == 0 {
							continue
						}
						nx, ny := p.x+dx, p.y+dy
						if nx >= 0 && nx < w && ny >= 0 && ny < h {
							nidx := ny*w + nx
							if !visited[nidx] && fg[nidx] {
								visited[nidx] = true
								queue = append(queue, point{nx, ny})
							}
						}
					}
				}
			}

			// Filter small components (noise). len(queue) == component size.
			if len(queue) < minComponentSize {
				continue
			}

			// Unclip: expand each side by d (pixel-centre extent, as minAreaRect measures it).
			bw := float64(maxX - minX)
			bh := float64(maxY - minY)
			d := unclipDistance(bw, bh, unclipRatio)

			x0 := math.Max(0, float64(minX)-d)
			y0 := math.Max(0, float64(minY)-d)
			x1 := math.Min(float64(w), float64(maxX)+d)
			y1 := math.Min(float64(h), float64(maxY)+d)

			result = append(result, [4]float64{x0, y0, x1 - x0, y1 - y0})
		}
	}

	return result, nil
}

// mapBoxToOriginal maps a box [x, y, w, h] from the detection model input space back to
// original image coordinates using detPreprocessMeta.
func mapBoxToOriginal(box [4]float64, meta detPreprocessMeta) [4]float64 {
	toOrig := meta.Affine()
	if toOrig.ScaleX <= 0 {
		toOrig.ScaleX = 1
	}
	if toOrig.ScaleY <= 0 {
		toOrig.ScaleY = 1
	}
	b := toOrig.BoxToOrig(box)
	x, y, bw, bh := b[0], b[1], b[2], b[3]

	// Clamp to original image bounds.
	origW := float64(meta.OrigWidth)
	origH := float64(meta.OrigHeight)

	x = math.Max(0, x)
	y = math.Max(0, y)
	if x+bw > origW {
		bw = origW - x
	}
	if y+bh > origH {
		bh = origH - y
	}

	return [4]float64{x, y, bw, bh}
}

// ctcDecode performs greedy CTC decoding on logits [T, C] (flat, row-major).
// Returns (text string, avgConf float64).
// Blank class is index 0 (PP-OCRv4 convention).
// charset maps index i to charset[i-1] (charset does not include blank at index 0);
// when c == len(charset)+2 the last class is the implicit space character.
func ctcDecode(logits []float32, t, c int, charset []string) (string, float64) {
	if t == 0 || c == 0 || len(logits) == 0 {
		return "", 0
	}

	var sb strings.Builder
	var confSum float64
	count := 0
	prevIdx := -1

	for step := 0; step < t; step++ {
		base := step * c
		if base+c > len(logits) {
			break
		}

		// Argmax and max value over c classes.
		best := 0
		bestVal := logits[base]
		for i := 1; i < c; i++ {
			if logits[base+i] > bestVal {
				bestVal = logits[base+i]
				best = i
			}
		}

		// Skip blank (index 0) and consecutive duplicates.
		if best == 0 || best == prevIdx {
			prevIdx = best
			continue
		}

		prevIdx = best

		// Map to character: charset[best-1] (charset has no blank entry). PaddleOCR's
		// CTCLabelDecode with use_space_char=True appends " " AFTER the dictionary, so a
		// head with C == len(charset)+2 classes (blank + keys + space) emits the space at
		// index len(charset)+1. Without this every space between words was dropped.
		switch {
		case best-1 < len(charset):
			sb.WriteString(charset[best-1])
		case best == len(charset)+1 && c == len(charset)+2:
			sb.WriteByte(' ')
		}
		confSum += float64(bestVal)
		count++
	}

	avgConf := 0.0
	if count > 0 {
		avgConf = confSum / float64(count)
	}

	return sb.String(), avgConf
}

// loadCharset loads the character dictionary from dir/ppocr_keys_v1.txt.
// Each line in the file is one character; the blank token (index 0) is implicit
// (not stored in the file). Returns an error if the file is missing or empty.
func loadCharset(dir string) ([]string, error) {
	charsetPath := filepath.Join(dir, "ppocr_keys_v1.txt")
	f, err := os.Open(charsetPath)
	if err != nil {
		return nil, fmt.Errorf("paddleocr: failed to open charset %s: %w", charsetPath, err)
	}
	defer f.Close()

	var chars []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		// Each line is one character (or multi-codepoint string); keep as-is.
		chars = append(chars, line)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("paddleocr: failed to read charset %s: %w", charsetPath, err)
	}
	if len(chars) == 0 {
		return nil, fmt.Errorf("paddleocr: charset %s is empty", charsetPath)
	}
	return chars, nil
}

// pickProbMap finds the probability map output from the det session outputs.
// PP-OCRv4 det outputs "sigmoid_0.tmp_0" shape [1, 1, H, W]; falls back to the first
// output with 4 dims and second dim == 1.
func pickProbMap(outNames []string, outs []engine.Tensor) *engine.Tensor {
	for i := range outs {
		name := ""
		if i < len(outNames) {
			name = strings.ToLower(outNames[i])
		}
		if strings.Contains(name, "sigmoid") || strings.Contains(name, "prob") || strings.Contains(name, "map") {
			return &outs[i]
		}
	}
	// Fallback: first output with 4 dims and second dim == 1.
	for i := range outs {
		if len(outs[i].Shape) == 4 && outs[i].Shape[1] == 1 {
			return &outs[i]
		}
	}
	if len(outs) > 0 {
		return &outs[0]
	}
	return nil
}

// pickRecLogits finds the recognition logits from the rec session outputs.
// PP-OCRv4 rec outputs "softmax_11.tmp_0" shape [1, T, num_chars]; falls back to
// the first output with 3 dims.
func pickRecLogits(outNames []string, outs []engine.Tensor) *engine.Tensor {
	for i := range outs {
		name := ""
		if i < len(outNames) {
			name = strings.ToLower(outNames[i])
		}
		if strings.Contains(name, "softmax") || strings.Contains(name, "logit") || strings.Contains(name, "pred") {
			return &outs[i]
		}
	}
	for i := range outs {
		if len(outs[i].Shape) == 3 {
			return &outs[i]
		}
	}
	if len(outs) > 0 {
		return &outs[0]
	}
	return nil
}
