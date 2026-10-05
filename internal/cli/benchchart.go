package cli

// Charts and small helpers for `visionserve bench`'s report (the report itself is clireport's).
// Charts are small PNGs drawn with the standard image package and the embedded bitmap font: no
// plotting dependency.

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// sanitizeJSON replaces non-finite floats (which encoding/json refuses) with nil, recursively,
// by a round trip through a generic value.
func sanitizeJSON(v any) any {
	switch x := v.(type) {
	case float64:
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil
		}
		return x
	case float32:
		return sanitizeJSON(float64(x))
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			out[k] = sanitizeJSON(e)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = sanitizeJSON(e)
		}
		return out
	}
	return v
}

func trimFloat(v float64) string {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return "n/a"
	}
	if math.Abs(v) >= 100 {
		return fmt.Sprintf("%.0f", v)
	}
	if math.Abs(v) >= 10 {
		return fmt.Sprintf("%.1f", v)
	}
	return fmt.Sprintf("%.2f", v)
}

// ---------------------------------------------------------------------------------------------
// Charts: small PNGs drawn with the standard image package and the embedded bitmap font (no
// plotting dependency). White background, so they read the same in light and dark mode.
// ---------------------------------------------------------------------------------------------

var (
	chartInk   = color.RGBA{0x1d, 0x23, 0x27, 0xff}
	chartMuted = color.RGBA{0x8a, 0x94, 0x9c, 0xff}
	chartGrid  = color.RGBA{0xe3, 0xe7, 0xea, 0xff}
	chartBar   = color.RGBA{0x2f, 0x6f, 0xb3, 0xff}
	chartMark  = color.RGBA{0xc2, 0x41, 0x0c, 0xff}
)

const chartScale = 2 // drawn at 2x so the bitmap font stays legible on high-DPI screens

func newCanvas(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	fill(img, img.Bounds(), color.White)
	return img
}

func fill(img *image.RGBA, r image.Rectangle, c color.Color) {
	r = r.Intersect(img.Bounds())
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			img.Set(x, y, c)
		}
	}
}

// drawText draws s with its top-left corner at (x, y) at chartScale.
func drawText(img *image.RGBA, x, y int, s string, c color.Color) {
	face := basicfont.Face7x13
	small := image.NewRGBA(image.Rect(0, 0, font.MeasureString(face, s).Ceil()+2, 16))
	d := &font.Drawer{Dst: small, Src: image.NewUniform(c), Face: face, Dot: fixed.P(1, 12)}
	d.DrawString(s)
	b := small.Bounds()
	for sy := 0; sy < b.Dy(); sy++ {
		for sx := 0; sx < b.Dx(); sx++ {
			if _, _, _, a := small.At(sx, sy).RGBA(); a > 0x7fff {
				fill(img, image.Rect(x+sx*chartScale, y+sy*chartScale, x+(sx+1)*chartScale, y+(sy+1)*chartScale), c)
			}
		}
	}
}

func textWidth(s string) int { return font.MeasureString(basicfont.Face7x13, s).Ceil() * chartScale }

func encodePNG(img image.Image) []byte {
	var buf bytes.Buffer
	_ = png.Encode(&buf, img) // encoding an in-memory RGBA cannot fail
	return buf.Bytes()
}

// histogramChart draws the distribution of samples (ms) in `bins` bins, with p50/p95/p99 marks.
func histogramChart(samples []float64, bins int, marks map[string]float64) []byte {
	W, H := 640*chartScale, 220*chartScale
	img := newCanvas(W, H)
	if len(samples) == 0 {
		drawText(img, 20*chartScale, 20*chartScale, "no samples", chartMuted)
		return encodePNG(img)
	}
	lo, hi := samples[0], samples[0]
	for _, s := range samples {
		lo, hi = math.Min(lo, s), math.Max(hi, s)
	}
	if hi-lo < 1e-9 {
		hi = lo + 1
	}
	counts := make([]int, bins)
	cmax := 0
	for _, s := range samples {
		b := int(float64(bins) * (s - lo) / (hi - lo))
		if b >= bins {
			b = bins - 1
		}
		counts[b]++
		cmax = max(cmax, counts[b])
	}
	left, right, top, bottom := 50*chartScale, 20*chartScale, 44*chartScale, 34*chartScale
	pw, ph := W-left-right, H-top-bottom
	for i := 0; i <= 4; i++ { // horizontal grid
		y := top + ph*i/4
		fill(img, image.Rect(left, y, left+pw, y+1), chartGrid)
	}
	bw := pw / bins
	for i, c := range counts {
		if c == 0 {
			continue
		}
		h := ph * c / cmax
		fill(img, image.Rect(left+i*bw+chartScale, top+ph-h, left+(i+1)*bw-chartScale, top+ph), chartBar)
	}
	fill(img, image.Rect(left, top+ph, left+pw, top+ph+chartScale), chartInk)
	drawText(img, left, top+ph+6*chartScale, trimFloat(lo)+" ms", chartInk)
	hs := trimFloat(hi) + " ms"
	drawText(img, left+pw-textWidth(hs), top+ph+6*chartScale, hs, chartInk)
	drawText(img, 4*chartScale, 2*chartScale, fmt.Sprintf("requests (max %d per bin)", cmax), chartMuted)
	for _, name := range []string{"p50", "p95", "p99"} {
		v, ok := marks[name]
		if !ok {
			continue
		}
		x := left + int(float64(pw)*(v-lo)/(hi-lo))
		for y := top; y < top+ph; y += 6 * chartScale {
			fill(img, image.Rect(x, y, x+chartScale, y+3*chartScale), chartMark)
		}
		drawText(img, x-textWidth(name)/2, top-14*chartScale, name, chartMark)
	}
	return encodePNG(img)
}

// timelineChart draws each request's latency in completion order (spots warm-up drift, thermal
// throttling, a GC pause).
func timelineChart(samples []float64) []byte {
	W, H := 640*chartScale, 200*chartScale
	img := newCanvas(W, H)
	if len(samples) == 0 {
		return encodePNG(img)
	}
	hi := 0.0
	for _, s := range samples {
		hi = math.Max(hi, s)
	}
	if hi <= 0 {
		hi = 1
	}
	left, right, top, bottom := 50*chartScale, 20*chartScale, 20*chartScale, 30*chartScale
	pw, ph := W-left-right, H-top-bottom
	for i := 0; i <= 4; i++ {
		y := top + ph*i/4
		fill(img, image.Rect(left, y, left+pw, y+1), chartGrid)
	}
	n := len(samples)
	for i, s := range samples {
		x := left
		if n > 1 {
			x = left + pw*i/(n-1)
		}
		y := top + ph - int(float64(ph)*s/hi)
		fill(img, image.Rect(x-chartScale, y-chartScale, x+2*chartScale, y+2*chartScale), chartBar)
	}
	fill(img, image.Rect(left, top+ph, left+pw, top+ph+chartScale), chartInk)
	drawText(img, 4*chartScale, top-2*chartScale, trimFloat(hi), chartMuted)
	drawText(img, 4*chartScale, top+ph-12*chartScale, "0 ms", chartMuted)
	drawText(img, left, top+ph+6*chartScale, "request 1", chartInk)
	last := fmt.Sprintf("request %d", n)
	drawText(img, left+pw-textWidth(last), top+ph+6*chartScale, last, chartInk)
	return encodePNG(img)
}
