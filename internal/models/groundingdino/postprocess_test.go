package groundingdino

import (
	"fmt"
	"image"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
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

	spans := phraseSpans(ids, tok, nil)

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
			spans := phraseSpans(tok.Encode(tc.prompt).InputIDs, tok, nil)
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
	spans := phraseSpans(ids, tok, nil)
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

	dets, err := postprocess(&logits, &boxes, ids, tok, nil, 100, 200, 0.3, 0.25)
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

	dets, err := postprocess(&logits, &boxes, ids, tok, nil, 100, 100, 0.3, 0.25)
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
		spans := phraseSpans(ids, tok, nil)
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

// fakeGDINO is a session stand-in that enforces the export's text limit the way ONNX Runtime
// does (L=257 fails with "invalid expand shape" on the real graph) and lights up one query per
// phrase span on that span's first token, so every phrase that reaches a pass yields one
// detection.
func fakeGDINO(t *testing.T, passes *[][]int64) func(map[string]engine.Tensor) ([]engine.Tensor, error) {
	return func(in map[string]engine.Tensor) ([]engine.Tensor, error) {
		ids := in["input_ids"].DataI64
		if len(ids) > MaxTextLen {
			return nil, fmt.Errorf("invalid expand shape (L=%d)", len(ids))
		}
		*passes = append(*passes, append([]int64(nil), ids...))
		var starts []int
		start := 1
		for i := 1; i < len(ids)-1; i++ {
			if ids[i] == idPeriod || ids[i] == idComma {
				if i > start {
					starts = append(starts, start)
				}
				start = i + 1
			}
		}
		if len(ids)-1 > start {
			starts = append(starts, start)
		}
		const dim = 256
		nq := len(starts)
		if nq == 0 {
			nq = 1
		}
		logitData := make([]float32, nq*dim)
		for i := range logitData {
			logitData[i] = -6.0
		}
		boxData := make([]float32, 0, nq*4)
		for q := 0; q < nq; q++ {
			if q < len(starts) {
				logitData[q*dim+starts[q]] = 3.0
			}
			boxData = append(boxData, 0.5, 0.5, 0.2, 0.4)
		}
		return []engine.Tensor{engine.F32(logitData, 1, int64(nq), dim), engine.F32(boxData, 1, int64(nq), 4)}, nil
	}
}

// B1: a prompt longer than the export's 256 text positions used to go to ORT as ONE pass and
// fail the request. On the joint path it must be packed into several passes of at most
// MaxTextLen ids each, and every phrase must still be scored exactly once.
func TestDetectJointPassChunksPromptOverTextLimit(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 100, 200))

	var phrases []string
	for i := 0; i < 80; i++ { // "object 7 thing." is 4 ids -> 322 ids in all
		phrases = append(phrases, fmt.Sprintf("object %d thing", i))
	}
	prompt := strings.Join(phrases, ". ") + "."
	if n := len(tok.Encode(prompt).InputIDs); n <= MaxTextLen {
		t.Fatalf("setup: prompt is only %d ids, want > %d", n, MaxTextLen)
	}

	var passes [][]int64
	dets, err := Detect(img, prompt, tok, fakeGDINO(t, &passes), []string{"logits", "pred_boxes"}, 0.3, 0.25,
		WithJointTextPass(true))
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(passes) < 2 {
		t.Fatalf("made %d passes, want the prompt split into >= 2", len(passes))
	}
	for i, p := range passes {
		if len(p) > MaxTextLen {
			t.Errorf("pass %d has %d ids, limit %d", i, len(p), MaxTextLen)
		}
	}
	got := make([]string, 0, len(dets))
	for _, d := range dets {
		got = append(got, d.Class)
	}
	want := append([]string(nil), phrases...)
	sort.Strings(got)
	sort.Strings(want)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("labels = %v\nwant %v (every phrase exactly once)", got, want)
	}
}

// The greedy packing must not change anything for a prompt that already fits: one pass, the
// same ids as before.
func TestDetectJointPassShortPromptIsOnePassWithSameIDs(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 100, 200))
	var passes [][]int64
	if _, err := Detect(img, "cup. water bottle. remote.", tok, fakeGDINO(t, &passes), nil, 0.3, 0.25,
		WithJointTextPass(true)); err != nil {
		t.Fatal(err)
	}
	if len(passes) != 1 {
		t.Fatalf("made %d passes, want 1", len(passes))
	}
	if want := tok.Encode("cup. water bottle. remote.").InputIDs; !reflect.DeepEqual(passes[0], want) {
		t.Errorf("ids = %v, want %v", passes[0], want)
	}
}

// A single phrase that cannot fit even on its own is a clear request error on BOTH paths,
// before any session runs — not an ORT shape error from deep inside the graph.
func TestDetectRejectsPhraseOverTextLimit(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	long := strings.TrimSpace(strings.Repeat("red ", MaxTextLen)) // MaxTextLen tokens + specials
	for _, joint := range []bool{false, true} {
		var passes [][]int64
		_, err := Detect(img, "cup. "+long+".", tok, fakeGDINO(t, &passes), nil, 0.3, 0.25, WithJointTextPass(joint))
		if err == nil {
			t.Errorf("joint=%v: no error for a %d-token phrase", joint, MaxTextLen)
			continue
		}
		if !strings.Contains(err.Error(), "tokens") {
			t.Errorf("joint=%v: error %q does not explain the token limit", joint, err)
		}
		if len(passes) != 0 {
			t.Errorf("joint=%v: %d passes ran before the error", joint, len(passes))
		}
	}
}

// The longest phrase that DOES fit (MaxTextLen-3 tokens: [CLS] phrase . [SEP]) must run.
func TestDetectAcceptsPhraseAtTextLimit(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	phrase := strings.TrimSpace(strings.Repeat("red ", MaxTextLen-3))
	for _, joint := range []bool{false, true} {
		var passes [][]int64
		if _, err := Detect(img, phrase+".", tok, fakeGDINO(t, &passes), nil, 0.3, 0.25, WithJointTextPass(joint)); err != nil {
			t.Errorf("joint=%v: %v", joint, err)
		}
		if len(passes) != 1 || len(passes[0]) != MaxTextLen {
			t.Errorf("joint=%v: passes %d, want 1 pass of %d ids", joint, len(passes), MaxTextLen)
		}
	}
}

// B4: labels are the caller's phrase text, not a WordPiece round trip ("t-shirt" used to come
// back as "t - shirt", "café" as "[UNK]").
func TestDetectLabelsAreCallerPhraseText(t *testing.T) {
	tok := testTokenizer(t)
	img := image.NewRGBA(image.Rect(0, 0, 100, 200))
	prompt := "t-shirt. café. Water  Bottle. red cup, blue cup."
	want := []string{"t-shirt", "café", "Water Bottle", "red cup", "blue cup"}
	for _, joint := range []bool{false, true} {
		var passes [][]int64
		dets, err := Detect(img, prompt, tok, fakeGDINO(t, &passes), nil, 0.3, 0.25, WithJointTextPass(joint))
		if err != nil {
			t.Fatalf("joint=%v: %v", joint, err)
		}
		got := make([]string, 0, len(dets))
		for _, d := range dets {
			got = append(got, d.Class)
		}
		sort.Strings(got)
		w := append([]string(nil), want...)
		sort.Strings(w)
		if !reflect.DeepEqual(got, w) {
			t.Errorf("joint=%v: labels %q, want %q", joint, got, w)
		}
	}
}

// B9: boxes are clamped to the image in ORIGINAL coordinates, like RF-DETR's.
func TestPostprocessClampsBoxesToImage(t *testing.T) {
	tok := testTokenizer(t)
	ids := tok.Encode("cup.").InputIDs
	const dim = 256
	logitData := make([]float32, 3*dim)
	for i := range logitData {
		logitData[i] = -6.0
	}
	for q := 0; q < 3; q++ {
		logitData[q*dim+1] = 3.0
	}
	logits := engine.F32(logitData, 1, 3, dim)
	boxes := engine.F32([]float32{
		0.05, 0.5, 0.2, 0.4, // spills off the left edge: x0 = -0.05
		0.95, 0.95, 0.2, 0.2, // spills off the right and bottom edges
		0.5, 0.5, 0.2, 0.2, // inside: unchanged
	}, 1, 3, 4)
	dets, err := postprocess(&logits, &boxes, ids, tok, []string{"cup"}, 100, 200, 0.3, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	if len(dets) != 3 {
		t.Fatalf("got %d detections, want 3", len(dets))
	}
	want := [][4]float64{
		{0, 60, 15, 80},   // x0 -5 -> 0, x1 15
		{85, 170, 15, 30}, // x1 105 -> 100, y1 210 -> 200
		{40, 80, 20, 40},
	}
	for i, w := range want {
		for k := range w {
			if g := dets[i].BBox[k]; g < w[k]-1e-3 || g > w[k]+1e-3 {
				t.Errorf("det %d bbox = %v, want %v", i, dets[i].BBox, w)
				break
			}
		}
	}
}
