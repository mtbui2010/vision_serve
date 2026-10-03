package server

import (
	"bytes"
	"fmt"
	"image"
	"io"
	"net/http"

	"github.com/disintegration/imaging"
)

// maxImagePixels caps a decoded image. maxImageBytes only caps the COMPRESSED upload, and image
// formats compress flat regions extremely well: a 265 KB PNG measured 8000x8000, i.e. 244 MB once
// decoded (~950x), so 32 MiB of upload could ask for tens of GB. 40 MP covers every real camera
// frame these models take (all resize to ≤1333 px anyway).
const maxImagePixels = 40_000_000

// maxDepthSide bounds an uploaded depth map's declared width/height before anything is allocated.
const maxDepthSide = 16384

// decodeImage reads at most maxImageBytes, checks the declared pixel count from the image header
// BEFORE decoding, then decodes. It is the one place every handler decodes an upload, so the
// limits cannot be forgotten at a new call site.
func decodeImage(r io.Reader) (image.Image, error) {
	raw, err := io.ReadAll(io.LimitReader(r, maxImageBytes+1))
	if err != nil {
		return nil, fmt.Errorf("reading image: %w", err)
	}
	if len(raw) > maxImageBytes {
		return nil, fmt.Errorf("image is larger than %d MiB", maxImageBytes>>20)
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > maxImagePixels {
		return nil, fmt.Errorf("image is %dx%d; the limit is %d megapixels", cfg.Width, cfg.Height, maxImagePixels/1_000_000)
	}
	// AutoOrientation applies the EXIF orientation tag, so a phone photo is processed the way
	// every viewer shows it (and the way transformers' load_image feeds models), and the
	// returned boxes are in that frame. Without it, a portrait JPEG stored sideways was
	// detected sideways. Formats without EXIF decode exactly as image.Decode would.
	img, err := imaging.Decode(bytes.NewReader(raw), imaging.AutoOrientation(true))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}
	return img, nil
}

// limitBody caps the whole request body. ParseMultipartForm's argument only bounds the part kept
// in memory — the rest spills to temp files with no limit — so without this a client could fill
// the disk with one request.
func limitBody(n int64, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, n)
		next(w, r)
	}
}
