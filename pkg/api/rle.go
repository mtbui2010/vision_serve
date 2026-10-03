package api

import "visionserve/internal/vision/mask"

// EncodeMaskRLE encodes a ROW-MAJOR (len w*h) binary mask into the Mask.RLE format:
// space-separated run lengths over a COLUMN-major traversal (x outer, y inner) of the
// w×h grid, with runs alternating and STARTING with background (false).
//
// It delegates to the single implementation, mask.EncodeRLE; note the argument order
// here is (w, h).
func EncodeMaskRLE(bin []bool, w, h int) string {
	return mask.EncodeRLE(mask.Bitmap{Data: bin, W: w, H: h})
}

// DecodeMaskRLE inverts EncodeMaskRLE into a ROW-MAJOR (len w*h) binary mask. An empty
// string (or non-positive dims) yields an all-false mask. Delegates to mask.DecodeRLE.
func DecodeMaskRLE(rle string, w, h int) []bool {
	return mask.DecodeRLE(rle, h, w).Data
}
