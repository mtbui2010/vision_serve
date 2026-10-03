package server

import (
	"bytes"
	"encoding/hex"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func pngBytes(t *testing.T, w, h int) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewGray(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// A tiny PNG declaring a huge canvas must be refused from its header, before decoding allocates
// width*height bytes (a 265 KB PNG of 8000x8000 decodes to 244 MB).
func TestDecodeImageRefusesDecompressionBomb(t *testing.T) {
	bomb := pngBytes(t, 8000, 6000) // 48 MP of zeros: a few hundred KB compressed
	if len(bomb) > 1<<20 {
		t.Fatalf("test bomb is %d bytes, expected a small file", len(bomb))
	}
	if _, err := decodeImage(bytes.NewReader(bomb)); err == nil || !strings.Contains(err.Error(), "megapixels") {
		t.Fatalf("decodeImage(8000x6000) = %v, want a megapixel-limit error", err)
	}
	if img, err := decodeImage(bytes.NewReader(pngBytes(t, 64, 32))); err != nil || img.Bounds().Dx() != 64 {
		t.Fatalf("normal image refused: %v", err)
	}
}

func TestDecodeImageRefusesOversizeUpload(t *testing.T) {
	big := make([]byte, maxImageBytes+10)
	if _, err := decodeImage(bytes.NewReader(big)); err == nil || !strings.Contains(err.Error(), "larger than") {
		t.Fatalf("oversize upload: %v, want a size error (not a truncated-decode error)", err)
	}
}

// The declared depth size must be checked before allocating: 2 bytes declaring 1e6 x 1e6 used to
// request 4 TB and kill the process with an unrecoverable OOM; 3.04e9 x 3.04e9 overflowed into a
// makeslice panic.
func TestParseDepthRefusesHugeDeclaredSize(t *testing.T) {
	for _, d := range [][2]int{{1_000_000, 1_000_000}, {3_040_000_000, 3_040_000_000}, {maxDepthSide + 1, 1}} {
		if got, _, _ := parseDepth([]byte{1, 2}, "uint16", d[0], d[1], 640, 480); got != nil {
			t.Fatalf("parseDepth(%dx%d) accepted 2 bytes", d[0], d[1])
		}
	}
}

func TestLimitBodyCapsTheRequest(t *testing.T) {
	h := limitBody(10, func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusRequestEntityTooLarge)
		}
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/", strings.NewReader(strings.Repeat("x", 1000)))
	req.Header.Set("Content-Type", "multipart/form-data; boundary=b")
	h(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("over-limit body: status %d, want 413", rec.Code)
	}
}

// exif6JPEG is an 8x4 JPEG tagged EXIF orientation 6 ("rotate 90° CW to display"): viewers and
// transformers' load_image show it as 4x8.
const exif6JPEG = "ffd8ffe000104a46494600010100000100010000ffe100224578696600004d4d002a00000008000101120003000000010006000000000000ffdb004300100b0c0e0c0a100e0d0e1211101318281a181616183123251d283a333d3c3933383740485c4e404457453738506d51575f626768673e4d71797064785c656763ffdb0043011112121815182f1a1a2f634238426363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363ffc00011080004000803012200021101031101ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00c0a28a2b80fad3ffd9"

func TestDecodeImageAppliesEXIFOrientation(t *testing.T) {
	raw, _ := hex.DecodeString(exif6JPEG)
	img, err := decodeImage(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != 4 || b.Dy() != 8 {
		t.Fatalf("decoded %dx%d, want 4x8 (EXIF orientation 6 applied)", b.Dx(), b.Dy())
	}
}
