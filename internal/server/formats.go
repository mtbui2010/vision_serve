package server

// The image formats every upload is decoded from (decodeImage, the template store, explain), and
// `visionserve run`, which links this package. image.Decode finds a format by its registration,
// so one blank import here covers every caller. JPEG, PNG, BMP, GIF and TIFF come with
// disintegration/imaging; WebP is golang.org/x/image/webp (BSD-3-Clause, pure Go, decode only).
import (
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"
)
