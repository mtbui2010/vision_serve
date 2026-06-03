package paddleocr

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"visionserve/internal/engine"
)

const (
	defaultDetThresh = 0.3 // probability map threshold
	defaultDilate    = 1.5 // bbox expansion factor from center
	minComponentSize = 16  // skip noise components smaller than this (pixels)
)

// point is a 2D pixel coordinate used in the BFS flood-fill.
type point struct{ x, y int }

// extractBBoxes thresholds the [1,1,H,W] probability map at thresh, finds connected
// components via iterative BFS (8-connectivity), returns axis-aligned bounding boxes as
// [x, y, w, h] in the detection model's input space (before mapping to original coords).
// Each box is expanded by the dilate factor from its center before returning.
func extractBBoxes(probMap []float32, h, w int, thresh, dilate float64) [][4]float64 {
	// Build binary mask.
	mask := make([]bool, h*w)
	for i, v := range probMap {
		mask[i] = float64(v) > thresh
	}

	visited := make([]bool, h*w)
	var result [][4]float64

	for startY := 0; startY < h; startY++ {
		for startX := 0; startX < w; startX++ {
			idx := startY*w + startX
			if !mask[idx] || visited[idx] {
				continue
			}

			// BFS flood-fill (iterative to avoid stack overflow on large images).
			queue := []point{{startX, startY}}
			component := []point{}

			for len(queue) > 0 {
				p := queue[0]
				queue = queue[1:]
				pidx := p.y*w + p.x
				if visited[pidx] {
					continue
				}
				visited[pidx] = true
				component = append(component, p)

				// Check 8 neighbors.
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						if dx == 0 && dy == 0 {
							continue
						}
						nx, ny := p.x+dx, p.y+dy
						if nx >= 0 && nx < w && ny >= 0 && ny < h {
							nidx := ny*w + nx
							if !visited[nidx] && mask[nidx] {
								queue = append(queue, point{nx, ny})
							}
						}
					}
				}
			}

			// Filter small components (noise).
			if len(component) < minComponentSize {
				continue
			}

			// Compute axis-aligned bounding box.
			minX, minY := component[0].x, component[0].y
			maxX, maxY := minX, minY
			for _, p := range component {
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
			}

			// Dilate: expand bbox by dilate factor from center.
			cx := float64(minX+maxX) / 2.0
			cy := float64(minY+maxY) / 2.0
			halfW := float64(maxX-minX+1) * dilate / 2.0
			halfH := float64(maxY-minY+1) * dilate / 2.0

			x0 := math.Max(0, cx-halfW)
			y0 := math.Max(0, cy-halfH)
			x1 := math.Min(float64(w-1), cx+halfW)
			y1 := math.Min(float64(h-1), cy+halfH)

			result = append(result, [4]float64{x0, y0, x1 - x0, y1 - y0})
		}
	}

	return result
}

// mapBoxToOriginal maps a box [x, y, w, h] from the detection model input space back to
// original image coordinates using detPreprocessMeta.
func mapBoxToOriginal(box [4]float64, meta detPreprocessMeta) [4]float64 {
	sx := meta.ScaleX
	sy := meta.ScaleY
	if sx <= 0 {
		sx = 1
	}
	if sy <= 0 {
		sy = 1
	}

	x := (box[0] - float64(meta.PadX)) / sx
	y := (box[1] - float64(meta.PadY)) / sy
	bw := box[2] / sx
	bh := box[3] / sy

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
// charset maps index i to charset[i-1] (charset does not include blank at index 0).
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

		// Map to character: charset[best-1] (charset has no blank entry).
		if best-1 < len(charset) {
			sb.WriteString(charset[best-1])
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
