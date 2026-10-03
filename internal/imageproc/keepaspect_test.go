package imageproc

import "testing"

// Expected values were produced by the real HF function, transformers'
// models/dpt/image_processing_dpt.get_resize_output_image_size(img, (th, tw), keep_aspect_ratio=True,
// multiple=m), not re-derived by hand.
func TestDPTKeepAspectSizeMatchesHF(t *testing.T) {
	cases := []struct{ w, h, tw, th, m, wantW, wantH int }{
		{848, 480, 518, 518, 14, 910, 518},
		{480, 848, 518, 518, 14, 518, 910}, // portrait
		{640, 480, 518, 518, 14, 686, 518},
		{518, 518, 518, 518, 14, 518, 518},
		{1, 1, 518, 518, 14, 518, 518}, // tiny: upscaled
		{5, 3, 518, 518, 14, 518, 308},
		{10000, 7000, 518, 518, 14, 742, 518}, // huge
		{7000, 10000, 518, 518, 14, 518, 742},
		{1920, 1080, 518, 518, 14, 924, 518},
		{1080, 1920, 518, 518, 14, 518, 924},
		{640, 427, 518, 518, 14, 518, 350}, // width scale is closer to 1: short side ends BELOW 518
		{500, 375, 518, 518, 14, 518, 392},
		{800, 600, 518, 518, 14, 686, 518},
		{256, 256, 518, 518, 14, 518, 518},
		{1036, 518, 518, 518, 14, 1036, 518},
		{700, 1036, 518, 518, 14, 518, 770},
		{333, 500, 518, 518, 14, 350, 518},
		{640, 480, 384, 384, 32, 512, 384},
		{640, 480, 518, 518, 1, 691, 518},
		{640, 480, 518, 392, 14, 518, 392}, // non-square target
		{800, 400, 256, 256, 1, 512, 256},
		// Rounding ties (x.5 exactly): Python's round is half-to-even.
		{200, 201, 100, 100, 1, 100, 100},    // 100.5 -> 100
		{200, 203, 100, 100, 1, 100, 102},    // 101.5 -> 102
		{1036, 1050, 518, 518, 14, 518, 532}, // 525/14 = 37.5 -> 38
		{1036, 1078, 518, 518, 14, 518, 532}, // 539/14 = 38.5 -> 38
		{1050, 1036, 518, 518, 14, 532, 518}, // same, landscape
		{600, 200, 300, 300, 1, 900, 300},    // |1-sw| == |1-sh|: HF fits the HEIGHT
	}
	for _, c := range cases {
		gw, gh := DPTKeepAspectSize(c.w, c.h, c.tw, c.th, c.m)
		if gw != c.wantW || gh != c.wantH {
			t.Errorf("DPTKeepAspectSize(%d,%d -> %dx%d, m=%d) = %dx%d, HF gives %dx%d",
				c.w, c.h, c.tw, c.th, c.m, gw, gh, c.wantW, c.wantH)
		}
	}
}

// HF returns height 0 for a 3000x20 image (and then fails); we keep at least one multiple.
func TestDPTKeepAspectSizeNeverZero(t *testing.T) {
	gw, gh := DPTKeepAspectSize(3000, 20, 518, 518, 14)
	if gw != 518 || gh != 14 {
		t.Fatalf("3000x20 -> %dx%d, want 518x14 (HF: 518x0)", gw, gh)
	}
	if gw, gh := DPTKeepAspectSize(640, 480, 518, 518, 0); gw != 691 || gh != 518 {
		t.Fatalf("multiple 0 must mean 1: got %dx%d", gw, gh)
	}
}
