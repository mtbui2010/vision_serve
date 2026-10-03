package groundingdino

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"visionserve/internal/engine"
)

// TestTextLimitRealORT pins MaxTextLen to the real export and checks the packing end to end:
// a 257-id pass is rejected by ONNX Runtime itself, and Detect turns a prompt longer than that
// into several passes that all run. Skipped without ORT_DYLIB_PATH or the re-exported weights.
func TestTextLimitRealORT(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("ORT_DYLIB_PATH not set; the fake-session tests cover the packing logic")
	}
	if testing.Short() {
		t.Skip("runs GroundingDINO on CPU several times")
	}
	dir := filepath.Join("..", "..", "..", "models", "grounding-dino")
	onnx := filepath.Join(dir, "model-fixedmask.onnx")
	if _, err := os.Stat(onnx); err != nil {
		t.Skipf("%s not present", onnx)
	}
	tok := testTokenizer(t)
	sess, err := engine.NewSession(onnx, nil, nil, []engine.Provider{engine.ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	img := image.Image(image.NewRGBA(image.Rect(0, 0, 640, 480)))
	if f, err := os.Open(filepath.Join("..", "..", "..", "demo", "images", "000000000139.jpg")); err == nil {
		if decoded, _, err := image.Decode(f); err == nil {
			img = decoded
		}
		f.Close()
	}
	pv, pm := preprocessImage(img)

	// The premise: one id over the limit fails inside ORT.
	ids := make([]int64, MaxTextLen+1)
	ids[0], ids[len(ids)-1] = idCLS, idSEP
	for i := 1; i < len(ids)-1; i++ {
		ids[i] = 2417 // "red"
	}
	ones := make([]int64, len(ids))
	for i := range ones {
		ones[i] = 1
	}
	L := int64(len(ids))
	_, err = sess.RunNamed(map[string]engine.Tensor{
		"pixel_values": pv, "pixel_mask": pm,
		"input_ids":      engine.I64(ids, 1, L),
		"attention_mask": engine.I64(ones, 1, L),
		"token_type_ids": engine.I64(make([]int64, L), 1, L),
	})
	if err == nil {
		t.Fatalf("a %d-id pass ran; MaxTextLen=%d is no longer the export's limit", L, MaxTextLen)
	}
	t.Logf("L=%d rejected by ORT as expected: %v", L, err)

	var phrases []string
	for i := 0; i < 80; i++ {
		phrases = append(phrases, fmt.Sprintf("object %d thing", i))
	}
	phrases = append(phrases, "person", "couch")
	prompt := strings.Join(phrases, ". ") + "."
	if n := len(tok.Encode(prompt).InputIDs); n <= MaxTextLen {
		t.Fatalf("setup: prompt is %d ids, want > %d", n, MaxTextLen)
	}
	passes := 0
	run := func(in map[string]engine.Tensor) ([]engine.Tensor, error) {
		passes++
		return sess.RunNamed(in)
	}
	dets, err := Detect(img, prompt, tok, run, sess.OutputNames(), 0.3, 0.25, WithJointTextPass(true))
	if err != nil {
		t.Fatalf("Detect on a %d-phrase prompt: %v", len(phrases), err)
	}
	if passes < 2 {
		t.Errorf("made %d passes, want >= 2", passes)
	}
	t.Logf("%d passes, %d detections", passes, len(dets))
}
