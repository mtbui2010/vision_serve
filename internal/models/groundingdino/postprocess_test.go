package groundingdino

import (
	"image"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"visionserve/internal/engine"
)

// testTokenizer loads the real bert-base-uncased vocab shipped with the weights. Skips
// when the weights are not present (they are not committed — see models/grounding-dino).
func testTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := LoadTokenizer(filepath.Join("..", "..", "..", "models", "grounding-dino", "vocab.txt"))
	if err != nil {
		t.Skipf("vocab.txt unavailable: %v", err)
	}
	return tok
}

func TestPhraseSpansSplitsOnPeriod(t *testing.T) {
	tok := testTokenizer(t)
	ids := tok.Encode("cup. water bottle. remote.").InputIDs

	spans := phraseSpans(ids, tok, 256)

	want := []string{"cup", "water bottle", "remote"}
	if len(spans) != len(want) {
		t.Fatalf("got %d spans %v, want %d", len(spans), spans, len(want))
	}
	for i, w := range want {
		if spans[i].text != w {
			t.Errorf("span %d = %q, want %q", i, spans[i].text, w)
		}
		if spans[i].start >= spans[i].end {
			t.Errorf("span %d has empty range [%d,%d)", i, spans[i].start, spans[i].end)
		}
	}
}

func TestPhraseSpansEdgeCases(t *testing.T) {
	tok := testTokenizer(t)
	for _, tc := range []struct {
		name, prompt string
		want         []string
	}{
		{"no trailing period", "cup", []string{"cup"}},
		{"double period", "cup.. remote.", []string{"cup", "remote"}},
		{"single class", "water bottle.", []string{"water bottle"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spans := phraseSpans(tok.Encode(tc.prompt).InputIDs, tok, 256)
			if len(spans) != len(tc.want) {
				t.Fatalf("got %d spans, want %d: %v", len(spans), len(tc.want), spans)
			}
			for i, w := range tc.want {
				if spans[i].text != w {
					t.Errorf("span %d = %q, want %q", i, spans[i].text, w)
				}
			}
		})
	}
}

// A query that fires on ONE phrase must be labelled with exactly that phrase — not with a
// concatenation of every token that happened to clear text_threshold. This is the bug that
// produced labels like "chair tv vase bear" on multi-class prompts.
func TestPostprocessAssignsSingleBestPhrase(t *testing.T) {
	tok := testTokenizer(t)
	prompt := "cup. water bottle. remote."
	ids := tok.Encode(prompt).InputIDs
	spans := phraseSpans(ids, tok, 256)
	if len(spans) != 3 {
		t.Fatalf("setup: got %d spans, want 3", len(spans))
	}

	const (
		nq  = 2
		dim = 256
	)
	// Logit values chosen so sigmoid lands well above / below the thresholds used below.
	const hi, mid, lo = float32(3.0), float32(0.5), float32(-6.0) // ~0.95, ~0.62, ~0.002
	logitData := make([]float32, nq*dim)
	for i := range logitData {
		logitData[i] = lo
	}
	// Query 0 fires hardest on "water bottle" but also lights up "cup" above text_threshold
	// — the old code would have emitted "cup water bottle".
	logitData[0*dim+spans[1].start] = hi
	logitData[0*dim+spans[0].start] = mid
	// Query 1 fires on "remote" only.
	logitData[1*dim+spans[2].start] = hi

	logits := engine.F32(logitData, 1, nq, dim)
	boxes := engine.F32([]float32{
		0.5, 0.5, 0.2, 0.4, // cxcywh, normalized
		0.25, 0.25, 0.1, 0.1,
	}, 1, nq, 4)

	dets, err := postprocess(&logits, &boxes, ids, tok, 100, 200, 0.3, 0.25)
	if err != nil {
		t.Fatalf("postprocess: %v", err)
	}
	if len(dets) != 2 {
		t.Fatalf("got %d detections, want 2: %+v", len(dets), dets)
	}
	if dets[0].Class != "water bottle" {
		t.Errorf("query 0 label = %q, want %q (must not concatenate phrases)", dets[0].Class, "water bottle")
	}
	if dets[1].Class != "remote" {
		t.Errorf("query 1 label = %q, want %q", dets[1].Class, "remote")
	}

	// Boxes come back as xywh in ORIGINAL pixels (100x200 here): cx=0.5,cy=0.5,w=0.2,h=0.4
	// -> x=(0.5-0.1)*100=40, y=(0.5-0.2)*200=60, w=20, h=80.
	want := [4]float64{40, 60, 20, 80}
	for i, w := range want {
		if got := dets[0].BBox[i]; got < w-1e-3 || got > w+1e-3 { // float32 logits/boxes
			t.Errorf("bbox[%d] = %v, want %v", i, got, w)
		}
	}
}

// A query whose best phrase stays under the thresholds must be dropped entirely.
func TestPostprocessDropsBelowThreshold(t *testing.T) {
	tok := testTokenizer(t)
	ids := tok.Encode("cup. remote.").InputIDs

	const dim = 256
	logitData := make([]float32, dim)
	for i := range logitData {
		logitData[i] = -6.0 // sigmoid ~0.002
	}
	logits := engine.F32(logitData, 1, 1, dim)
	boxes := engine.F32([]float32{0.5, 0.5, 0.2, 0.2}, 1, 1, 4)

	dets, err := postprocess(&logits, &boxes, ids, tok, 100, 100, 0.3, 0.25)
	if err != nil {
		t.Fatalf("postprocess: %v", err)
	}
	if len(dets) != 0 {
		t.Errorf("got %d detections, want 0: %+v", len(dets), dets)
	}
}

func TestSplitPhrases(t *testing.T) {
	for _, tc := range []struct {
		name, prompt string
		want         []string
	}{
		{"multi class", "cup. water bottle. remote.", []string{"cup", "water bottle", "remote"}},
		{"no trailing period", "cup", []string{"cup"}},
		{"double period", "cup.. remote.", []string{"cup", "remote"}},
		{"extra spaces", "  chair .  tv .", []string{"chair", "tv"}},
		{"empty", "  . . ", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := SplitPhrases(tc.prompt); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("SplitPhrases(%q) = %v, want %v", tc.prompt, got, tc.want)
			}
		})
	}
}

// The community ONNX export only builds the text self-attention block for the FIRST
// "."-separated phrase (see the package doc), which makes a joint multi-class pass
// order-dependent. Detect must therefore issue ONE pass per phrase, each carrying exactly
// that phrase, and the merged result must not depend on the order of the phrases.
func TestDetectRunsOnePassPerPhraseAndIsOrderStable(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 100, 200))

	// Fake session: whatever single-class prompt it is handed, it lights up query 0 on the
	// first real token. A joint pass would instead have to score several phrases at once.
	newRun := func(seen *[]string) func(map[string]engine.Tensor) ([]engine.Tensor, error) {
		return func(in map[string]engine.Tensor) ([]engine.Tensor, error) {
			ids := in["input_ids"].DataI64
			if len(ids) != 4 { // [CLS] <word> . [SEP] — one phrase, one word
				t.Errorf("pass got %d input_ids %v, want a single-phrase prompt", len(ids), ids)
			}
			*seen = append(*seen, tok.Decode(ids[1:len(ids)-1]))

			const dim = 256
			logitData := make([]float32, dim)
			for i := range logitData {
				logitData[i] = -6.0 // sigmoid ~0.002
			}
			logitData[1] = 3.0 // ~0.95 on the first real token of this phrase
			logits := engine.F32(logitData, 1, 1, dim)
			boxes := engine.F32([]float32{0.5, 0.5, 0.2, 0.4}, 1, 1, 4)
			return []engine.Tensor{logits, boxes}, nil
		}
	}

	classesOf := func(prompt string) []string {
		var seen []string
		dets, err := Detect(img, prompt, tok, newRun(&seen), []string{"logits", "pred_boxes"}, 0.3, 0.25)
		if err != nil {
			t.Fatalf("Detect(%q): %v", prompt, err)
		}
		want := SplitPhrases(prompt)
		if len(seen) != len(want) {
			t.Fatalf("Detect(%q) made %d passes %v, want %d (one per phrase)", prompt, len(seen), seen, len(want))
		}
		for i := range want {
			if seen[i] != want[i]+" ." {
				t.Errorf("pass %d ran on %q, want phrase %q", i, seen[i], want[i])
			}
		}
		got := make([]string, 0, len(dets))
		for _, d := range dets {
			got = append(got, d.Class)
		}
		sort.Strings(got)
		return got
	}

	base := classesOf("cup. remote. book.")
	if want := []string{"book", "cup", "remote"}; !reflect.DeepEqual(base, want) {
		t.Fatalf("classes = %v, want %v", base, want)
	}
	for _, reordered := range []string{"remote. book. cup.", "book. cup. remote."} {
		if got := classesOf(reordered); !reflect.DeepEqual(got, base) {
			t.Errorf("classes for %q = %v, want %v (must not depend on phrase order)", reordered, got, base)
		}
	}
}

func TestDetectRejectsPromptWithoutPhrase(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	run := func(map[string]engine.Tensor) ([]engine.Tensor, error) {
		t.Fatal("session must not run for a prompt with no class phrase")
		return nil, nil
	}
	if _, err := Detect(img, " . . ", tok, run, nil, 0.3, 0.25); err == nil {
		t.Error("Detect with a punctuation-only prompt returned no error")
	}
}

// On a correctly re-exported graph the whole prompt is scored in ONE pass, and the labels
// still come out one class per detection (phraseSpans splits the logits).
func TestDetectJointTextPassRunsOncePerPrompt(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 100, 200))
	prompt := "cup. remote. book."

	var seen [][]int64
	run := func(in map[string]engine.Tensor) ([]engine.Tensor, error) {
		ids := in["input_ids"].DataI64
		seen = append(seen, append([]int64(nil), ids...))

		// One query per phrase, each firing on that phrase's first token.
		spans := phraseSpans(ids, tok, 256)
		const dim = 256
		logitData := make([]float32, len(spans)*dim)
		for i := range logitData {
			logitData[i] = -6.0
		}
		boxData := make([]float32, 0, len(spans)*4)
		for q, sp := range spans {
			logitData[q*dim+sp.start] = 3.0
			boxData = append(boxData, 0.5, 0.5, 0.2, 0.4)
		}
		logits := engine.F32(logitData, 1, int64(len(spans)), dim)
		boxes := engine.F32(boxData, 1, int64(len(spans)), 4)
		return []engine.Tensor{logits, boxes}, nil
	}

	dets, err := Detect(img, prompt, tok, run, []string{"logits", "pred_boxes"}, 0.3, 0.25,
		WithJointTextPass(true))
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(seen) != 1 {
		t.Fatalf("made %d passes, want 1 for a joint pass", len(seen))
	}
	if got, want := tok.Decode(seen[0]), "[CLS] cup . remote . book . [SEP]"; got != want {
		t.Errorf("joint pass ran on %q, want %q", got, want)
	}
	got := make([]string, 0, len(dets))
	for _, d := range dets {
		got = append(got, d.Class)
	}
	sort.Strings(got)
	if want := []string{"book", "cup", "remote"}; !reflect.DeepEqual(got, want) {
		t.Errorf("classes = %v, want %v", got, want)
	}
}

// The joint pass must not change what a single-phrase prompt does.
func TestDetectJointTextPassSinglePhrase(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	var passes int
	run := func(in map[string]engine.Tensor) ([]engine.Tensor, error) {
		passes++
		const dim = 256
		d := make([]float32, dim)
		for i := range d {
			d[i] = -6.0
		}
		logits := engine.F32(d, 1, 1, dim)
		boxes := engine.F32([]float32{0.5, 0.5, 0.2, 0.2}, 1, 1, 4)
		return []engine.Tensor{logits, boxes}, nil
	}
	for _, joint := range []bool{false, true} {
		passes = 0
		if _, err := Detect(img, "chair.", tok, run, nil, 0.3, 0.25, WithJointTextPass(joint)); err != nil {
			t.Fatalf("joint=%v: %v", joint, err)
		}
		if passes != 1 {
			t.Errorf("joint=%v made %d passes, want 1", joint, passes)
		}
	}
}
