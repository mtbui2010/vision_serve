package mask

import (
	"strconv"
	"strings"
)

// EncodeRLE encodes b as the public Mask.RLE string: COCO-style UNCOMPRESSED run lengths
// over a COLUMN-major traversal (x outer, y inner) of the W×H grid, runs alternating and
// always STARTING with background (a leading 0 when the first pixel is set), serialized as
// space-separated decimal counts. This is the ONE encoder (pkg/api.EncodeMaskRLE and every
// model delegate here); clients decode it with the same convention.
//
// An empty bitmap (no data or a non-positive dimension) encodes as "".
func EncodeRLE(b Bitmap) string {
	if len(b.Data) == 0 || b.W <= 0 || b.H <= 0 {
		return ""
	}
	w, h, bin := b.W, b.H, b.Data
	buf := make([]byte, 0, 64)
	prev := false // runs start with background
	run := 0
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			if bin[y*w+x] == prev {
				run++
				continue
			}
			buf = strconv.AppendInt(buf, int64(run), 10)
			buf = append(buf, ' ')
			prev = !prev
			run = 1
		}
	}
	buf = strconv.AppendInt(buf, int64(run), 10)
	return string(buf)
}

// DecodeRLE inverts EncodeRLE into an h×w bitmap. An empty string, non-positive
// dimensions or a malformed count stop decoding early; the remaining pixels stay
// background (the bitmap is always h*w long when h*w > 0).
func DecodeRLE(rle string, h, w int) Bitmap {
	n := w * h
	if n < 0 {
		n = 0
	}
	b := Bitmap{Data: make([]bool, n), W: w, H: h}
	if rle == "" || w <= 0 || h <= 0 {
		return b
	}
	val := false
	idx := 0
	for _, f := range strings.Fields(rle) {
		c, err := strconv.Atoi(f)
		if err != nil || c < 0 {
			break
		}
		for k := 0; k < c && idx < n; k++ {
			x := idx / h
			y := idx % h
			b.Data[y*w+x] = val
			idx++
		}
		val = !val
	}
	return b
}
