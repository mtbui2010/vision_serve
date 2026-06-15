package background

import (
	"image"
	"image/color"
	"testing"

	"visionserve/internal/models"
)

// TestBackgroundCV_BottomSurface synthesizes an image with a uniform, smooth surface in the
// bottom region and a differently-colored, textured object near the top, then asserts the CV
// method selects the bottom surface and excludes the object.
func TestBackgroundCV_BottomSurface(t *testing.T) {
	const w, h = 200, 200
	img := image.NewNRGBA(image.Rect(0, 0, w, h))

	surfaceTop := h / 2 // bottom half is the support surface
	surface := color.NRGBA{R: 120, G: 140, B: 110, A: 255}
	top := color.NRGBA{R: 30, G: 30, B: 200, A: 255}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if y >= surfaceTop {
				img.SetNRGBA(x, y, surface)
			} else {
				// Top half: a different color with a high-frequency checker texture.
				c := top
				if (x/4+y/4)%2 == 0 {
					c = color.NRGBA{R: 220, G: 220, B: 40, A: 255}
				}
				img.SetNRGBA(x, y, c)
			}
		}
	}

	m := &backgroundModel{}
	mask, err := m.backgroundCV(img, models.Prompt{}, nil)
	if err != nil {
		t.Fatalf("backgroundCV error: %v", err)
	}
	if mask == nil {
		t.Fatal("expected a support-surface mask, got nil")
	}
	if len(mask) != w*h {
		t.Fatalf("mask length = %d, want %d", len(mask), w*h)
	}

	// A bottom-center point should be IN the surface.
	if !mask[(h-5)*w+w/2] {
		t.Error("bottom-center pixel not selected as surface")
	}
	// A top-center point (the textured object) should be EXCLUDED.
	if mask[5*w+w/2] {
		t.Error("top-center (object) pixel wrongly selected as surface")
	}

	// Most of the bottom half should be selected; almost none of the top half.
	var bottomSel, topSel int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mask[y*w+x] {
				if y >= surfaceTop {
					bottomSel++
				} else {
					topSel++
				}
			}
		}
	}
	bottomArea := (h - surfaceTop) * w
	if bottomSel < bottomArea*80/100 {
		t.Errorf("only %d/%d bottom pixels selected (<80%%)", bottomSel, bottomArea)
	}
	if topSel > bottomArea*5/100 {
		t.Errorf("too many top pixels selected: %d", topSel)
	}
}

// TestBackgroundCV_NoSurface checks that a fully textured/noisy image yields no support
// surface (nil), since nothing forms a large smooth border-touching region.
func TestBackgroundCV_NoSurface(t *testing.T) {
	const w, h = 120, 120
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// High-frequency checker everywhere -> high gradient -> no smooth region.
			if (x+y)%2 == 0 {
				img.SetNRGBA(x, y, color.NRGBA{R: 0, G: 0, B: 0, A: 255})
			} else {
				img.SetNRGBA(x, y, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
			}
		}
	}
	m := &backgroundModel{}
	mask, err := m.backgroundCV(img, models.Prompt{}, nil)
	if err != nil {
		t.Fatalf("backgroundCV error: %v", err)
	}
	if mask != nil && anySet(mask) {
		t.Error("expected no support surface for a fully textured image")
	}
}
