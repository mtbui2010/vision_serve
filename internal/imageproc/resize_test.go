package imageproc

import (
	"image"
	"math"
	"testing"
)

// The CLIP recipe, pinned to HuggingFace's rounding: 848x480 -> short side 224, long side
// int(224*848/480) = 395 (truncated), crop offset (395-224)//2 = 85 (floored).
func TestResizeShortCenterCropMatchesHFRounding(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 848, 480))
	out, sx, sy, offX, offY := ResizeShortCenterCrop(src, 224, 224)
	if b := out.Bounds(); b.Dx() != 224 || b.Dy() != 224 {
		t.Fatalf("crop is %v, want 224x224", b)
	}
	if offX != 85 || offY != 0 {
		t.Fatalf("offset (%d,%d), want (85,0)", offX, offY)
	}
	if math.Abs(sx-395.0/848) > 1e-12 || math.Abs(sy-224.0/480) > 1e-12 {
		t.Fatalf("scale (%v,%v), want (395/848, 224/480)", sx, sy)
	}
	// portrait: the width is the short side
	if _, _, _, ox, oy := ResizeShortCenterCrop(image.NewRGBA(image.Rect(0, 0, 300, 500)), 224, 224); ox != 0 || oy != (373-224)/2 {
		t.Fatalf("portrait offset (%d,%d), want (0,%d)", ox, oy, (373-224)/2)
	}
}
