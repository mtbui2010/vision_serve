package api

import (
	"strconv"
	"strings"
)

// EncodeMaskRLE encodes a ROW-MAJOR (len w*h) binary mask into the Mask.RLE format:
// space-separated run lengths over a COLUMN-MAJOR traversal (x outer, y inner) of the
// w×h grid, with runs alternating and STARTING with background (false). This matches the
// SAM models' encoder, so masks re-encoded here decode identically on the clients.
func EncodeMaskRLE(bin []bool, w, h int) string {
	if len(bin) == 0 || w <= 0 || h <= 0 {
		return ""
	}
	var counts []int
	prev := false // runs start with background
	run := 0
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			if bin[y*w+x] == prev {
				run++
			} else {
				counts = append(counts, run)
				prev = !prev
				run = 1
			}
		}
	}
	counts = append(counts, run)

	var sb strings.Builder
	for i, c := range counts {
		if i > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(strconv.Itoa(c))
	}
	return sb.String()
}

// DecodeMaskRLE inverts EncodeMaskRLE into a ROW-MAJOR (len w*h) binary mask. An empty
// string (or non-positive dims) yields an all-false mask.
func DecodeMaskRLE(rle string, w, h int) []bool {
	data := make([]bool, max0(w*h))
	if rle == "" || w <= 0 || h <= 0 {
		return data
	}
	val := false
	idx := 0
	total := w * h
	for _, f := range strings.Fields(rle) {
		n, err := strconv.Atoi(f)
		if err != nil || n < 0 {
			break
		}
		for k := 0; k < n && idx < total; k++ {
			x := idx / h
			y := idx % h
			data[y*w+x] = val
			idx++
		}
		val = !val
	}
	return data
}

func max0(n int) int {
	if n < 0 {
		return 0
	}
	return n
}
