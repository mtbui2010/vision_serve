package mask

// ResizeNearest nearest-neighbour resizes b to h×w with UpsampleNearest's index rule (source
// index = dst·src/dst, integer division, clamped to the last row/column) — the bitmap
// counterpart of UpsampleNearest, for masks computed at a working resolution and returned at
// the original one. When the size is unchanged it returns b itself: the data is SHARED, not
// copied.
func ResizeNearest(b Bitmap, h, w int) Bitmap {
	if b.W == w && b.H == h {
		return b
	}
	out := New(h, w)
	if b.W <= 0 || b.H <= 0 {
		return out
	}
	for dy := 0; dy < h; dy++ {
		sy := dy * b.H / h
		if sy >= b.H {
			sy = b.H - 1
		}
		src := b.Data[sy*b.W : (sy+1)*b.W]
		row := out.Data[dy*w : (dy+1)*w]
		for dx := range row {
			sx := dx * b.W / w
			if sx >= b.W {
				sx = b.W - 1
			}
			row[dx] = src[sx]
		}
	}
	return out
}

// Or sets every pixel of dst that is set in src: the union of two same-size row-major masks,
// accumulated into dst.
func Or(dst, src []bool) {
	for i, v := range src {
		if v {
			dst[i] = true
		}
	}
}
