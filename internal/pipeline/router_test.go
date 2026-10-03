package pipeline

import (
	"fmt"
	"image"
	"reflect"
	"sync"
	"testing"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

func TestParseClasses(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"   ", nil},
		{"cat. remote.", []string{"cat", "remote"}},
		{"Person.  DOG ", []string{"person", "dog"}},
		{"a..b.", []string{"a", "b"}},
		// B2: a repeated word reaches the rescorer ONCE (twice halves every rescored confidence).
		{"zebra. zebra.", []string{"zebra"}},
		{"Zebra. cup. ZEBRA. cup.", []string{"zebra", "cup"}},
		{"cup. zebra. cup.", []string{"cup", "zebra"}}, // first-seen order
		// B7: internal whitespace must not decide the route.
		{"dining  table.\tteddy \t bear.", []string{"dining table", "teddy bear"}},
	}
	for _, c := range cases {
		if got := ParseClasses(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("ParseClasses(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// The router reads text and dispatches; it never detects. The case that matters is the mixed
// prompt: it used to send the WHOLE request to GroundingDINO as soon as one word was out of
// vocabulary, which threw away the in-domain detector for the words it was trained on.
func TestPartition(t *testing.T) {
	rt := &Router{Vocab: ClosedVocab([]string{"N/A", "person", "Car", "dog", "dining  table", ""})}
	cases := []struct {
		name           string
		classes        []string
		known, unknown []string
	}{
		{"no prompt", nil, nil, nil},
		{"all in vocab", []string{"person", "car"}, []string{"person", "car"}, nil},
		{"all out of vocab", []string{"unicorn", "n/a"}, nil, []string{"unicorn", "n/a"}},
		{"mixed splits, order preserved", []string{"person", "unicorn", "dog", "zebra"},
			[]string{"person", "dog"}, []string{"unicorn", "zebra"}},
		{"normalised labels", ParseClasses("dining  table."), []string{"dining table"}, nil},
	}
	for _, c := range cases {
		known, unknown := rt.Partition(c.classes)
		if !reflect.DeepEqual(known, c.known) || !reflect.DeepEqual(unknown, c.unknown) {
			t.Errorf("%s: Partition = %v / %v, want %v / %v", c.name, known, unknown, c.known, c.unknown)
		}
	}
}

func TestJoinPhrases(t *testing.T) {
	for in, want := range map[string]string{"": "", "zebra": "zebra.", "zebra|snack bag": "zebra. snack bag."} {
		var words []string
		if in != "" {
			words = splitBar(in)
		}
		if got := joinPhrases(words); got != want {
			t.Errorf("joinPhrases(%v) = %q, want %q", words, got, want)
		}
	}
}

func splitBar(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == '|' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return out
}

// A phrase prompt survives the words round trip exactly: what GroundingDINO's Detect splits back
// out of joinPhrases(TextPhrases(text)) is what it split out of text.
func TestTextPhrasesRoundTrip(t *testing.T) {
	for _, text := range []string{"cat. remote.", " red  cup , blue cup . t-shirt", "a..b.", "café"} {
		words, err := TextPhrases(text)
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		again, err := TextPhrases(joinPhrases(words))
		if err != nil || !reflect.DeepEqual(again, words) {
			t.Errorf("%q: phrases %q, after the round trip %q (%v)", text, words, again, err)
		}
	}
	if _, err := TextPhrases(" . .. "); err == nil {
		t.Error("a prompt with no phrase was accepted")
	}
}

func TestFilterByClass(t *testing.T) {
	dets := []models.Detection{{Class: "person"}, {Class: "Car"}, {Class: "dog"}}
	got := FilterByClass(dets, []string{"person", "car"})
	if len(got) != 2 || got[0].Class != "person" || got[1].Class != "Car" {
		t.Fatalf("FilterByClass = %+v, want person and Car (case-insensitive)", got)
	}
}

// ---- Router behaviour, with fake stages ----------------------------------------------------

// fakeModel is a plain closed-set detector with fixed detections; it owns no session.
type fakeModel struct{ dets []models.Detection }

func (fakeModel) Name() string          { return "fake" }
func (fakeModel) Task() models.Task     { return models.TaskDetection }
func (fakeModel) InputName() string     { return "images" }
func (fakeModel) OutputNames() []string { return nil }
func (fakeModel) Preprocess(image.Image) (engine.Tensor, models.PreprocessMeta, error) {
	return engine.F32([]float32{0}, 1), models.PreprocessMeta{OrigWidth: 8, OrigHeight: 8}, nil
}
func (f fakeModel) Postprocess(outs []engine.Tensor, _ models.PreprocessMeta) (models.Result, error) {
	if len(outs) != 2 || outs[0].Dim(-1) != 4 {
		return models.Result{}, fmt.Errorf("postprocess got %d outputs, want exactly [boxes, logits]", len(outs))
	}
	return models.Result{Detections: append([]models.Detection(nil), f.dets...)}, nil
}

// detRunner answers the closed role with [boxes, logits, query_feats].
type detRunner struct{ calls int }

func (r *detRunner) Run(role string, _ map[string]engine.Tensor) ([]engine.Tensor, error) {
	r.calls++
	return []engine.Tensor{engine.F32(make([]float32, 4), 1, 1, 4), engine.F32(make([]float32, 3), 1, 1, 3),
		engine.F32(make([]float32, 8), 1, 1, 8)}, nil
}
func (r *detRunner) InputNames(string) []string  { return []string{"input"} }
func (r *detRunner) OutputNames(string) []string { return nil }

type recDetector struct {
	name  string
	words [][]string
	lock  *probeLock // asserted held while Detect runs, when set
}

func (d *recDetector) Detect(c Call, words []string) ([]models.Detection, error) {
	d.words = append(d.words, words)
	if d.lock != nil && !d.lock.held() {
		return nil, fmt.Errorf("%s ran without the open lock", d.name)
	}
	var out []models.Detection
	for _, w := range words {
		out = append(out, models.Detection{Class: w, Conf: 0.5})
	}
	return out, nil
}

type recHead struct{ recDetector }

func (h *recHead) DetectQueries(c Call, p *ClosedPass, words []string) ([]models.Detection, error) {
	if p == nil || p.Feats.Data == nil {
		return nil, fmt.Errorf("head without the closed pass's query features")
	}
	return h.Detect(c, words)
}

// halfRescorer halves every confidence and records what it was asked about.
type halfRescorer struct{ words [][]string }

func (h *halfRescorer) Rescore(c Call, dets []models.Detection, words []string) ([]models.Detection, error) {
	h.words = append(h.words, words)
	out := append([]models.Detection(nil), dets...)
	for i := range out {
		out[i].Conf /= 2
	}
	return out, nil
}

type countSegmenter struct{ n int }

func (s *countSegmenter) Segment(c Call, dets []models.Detection) ([]models.Mask, error) {
	s.n = len(dets)
	masks := make([]models.Mask, len(dets))
	for i, d := range dets {
		masks[i] = models.Mask{Conf: d.Conf}
	}
	return masks, nil
}

type probeLock struct {
	mu     sync.Mutex
	locked bool
	takes  int
}

func (p *probeLock) Lock()      { p.mu.Lock(); p.locked = true; p.takes++ }
func (p *probeLock) Unlock()    { p.locked = false; p.mu.Unlock() }
func (p *probeLock) held() bool { return p.locked }

func testRouter(head bool) (*Router, *recDetector, *recHead, *halfRescorer, *countSegmenter, *probeLock) {
	lock := &probeLock{}
	open := &recDetector{name: "open", lock: lock}
	rs := &halfRescorer{}
	seg := &countSegmenter{}
	rt := &Router{
		Name:     "router",
		Vocab:    ClosedVocab([]string{"cup", "person"}),
		Closed:   &Closed{Role: "rfdetr", Model: fakeModel{dets: []models.Detection{{Class: "cup", Conf: 0.9}, {Class: "person", Conf: 0.8}}}, DETR: true, Labels: 3, Prefix: "t"},
		Open:     open,
		Rescore:  rs,
		Segment:  seg,
		OpenLock: lock,
	}
	var h *recHead
	if head {
		h = &recHead{recDetector{name: "head"}}
		rt.Head = h
	}
	return rt, open, h, rs, seg, lock
}

func classes(dets []models.Detection) []string {
	var out []string
	for _, d := range dets {
		out = append(out, d.Class)
	}
	return out
}

func TestRouterClosedOnlyNeverTakesOpenLock(t *testing.T) {
	for _, prompt := range []string{"cup.", ""} {
		rt, open, _, rs, seg, lock := testRouter(false)
		res, err := rt.Infer(Call{Img: image.NewRGBA(image.Rect(0, 0, 8, 8)), Prompt: models.Prompt{Text: prompt}, Runner: &detRunner{}})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{"cup"}
		if prompt == "" {
			want = []string{"cup", "person"} // its own vocabulary IS the prompt
		}
		if !reflect.DeepEqual(classes(res.Detections), want) {
			t.Errorf("%q: detections %v, want %v", prompt, classes(res.Detections), want)
		}
		if lock.takes != 0 || len(open.words) != 0 || len(rs.words) != 0 {
			t.Errorf("%q: open lock taken %d×, open ran %v, rescorer ran %v — closed answers are untouched", prompt, lock.takes, open.words, rs.words)
		}
		if seg.n != len(want) || len(res.Masks) != len(want) {
			t.Errorf("%q: %d masks for %d detections", prompt, len(res.Masks), len(want))
		}
	}
}

func TestRouterMixedPromptSplitsAndRescoresOnlyOpen(t *testing.T) {
	rt, open, _, rs, _, lock := testRouter(false)
	res, err := rt.Infer(Call{Img: image.NewRGBA(image.Rect(0, 0, 8, 8)), Prompt: models.Prompt{Text: "zebra. Cup. lamp."}, Runner: &detRunner{}})
	if err != nil {
		t.Fatal(err)
	}
	if got := classes(res.Detections); !reflect.DeepEqual(got, []string{"cup", "zebra", "lamp"}) {
		t.Errorf("detections %v, want the closed cup then the open zebra, lamp", got)
	}
	if !reflect.DeepEqual(open.words, [][]string{{"zebra", "lamp"}}) || !reflect.DeepEqual(rs.words, open.words) {
		t.Errorf("open asked %v, rescorer asked %v; both want exactly the unknown words once", open.words, rs.words)
	}
	if res.Detections[0].Conf != 0.9 || res.Detections[1].Conf != 0.25 {
		t.Errorf("confidences %v / %v: the closed one must not be rescored", res.Detections[0].Conf, res.Detections[1].Conf)
	}
	if lock.takes != 1 || lock.held() {
		t.Errorf("open lock taken %d×, still held %v; want once, released", lock.takes, lock.held())
	}
}

func TestRouterHeadAnswersUnknownWordsFromTheSamePass(t *testing.T) {
	rt, open, h, rs, _, lock := testRouter(true)
	r := &detRunner{}
	res, err := rt.Infer(Call{Img: image.NewRGBA(image.Rect(0, 0, 8, 8)), Prompt: models.Prompt{Text: "hat. bread."}, Runner: r})
	if err != nil {
		t.Fatal(err)
	}
	if r.calls != 1 {
		t.Errorf("the closed detector ran %d×, want once (the head scores its queries)", r.calls)
	}
	if got := classes(res.Detections); !reflect.DeepEqual(got, []string{"hat", "bread"}) {
		t.Errorf("detections %v; the closed head has no answer for unknown words", got)
	}
	if len(open.words) != 0 || lock.takes != 0 {
		t.Error("GroundingDINO consulted although the head answered")
	}
	if !reflect.DeepEqual(h.words, [][]string{{"hat", "bread"}}) || !reflect.DeepEqual(rs.words, h.words) {
		t.Errorf("head asked %v, rescorer %v", h.words, rs.words)
	}
}

func TestRouterWithoutClosedDetectorNeedsAPrompt(t *testing.T) {
	rt, _, _, _, _, _ := testRouter(false)
	rt.Closed, rt.Vocab = nil, nil
	if _, err := rt.Infer(Call{Img: image.NewRGBA(image.Rect(0, 0, 8, 8)), Runner: &detRunner{}}); err == nil {
		t.Fatal("an empty prompt with no closed-set detector was accepted")
	}
	res, err := rt.Infer(Call{Img: image.NewRGBA(image.Rect(0, 0, 8, 8)), Prompt: models.Prompt{Text: "cup."}, Runner: &detRunner{}})
	if err != nil || !reflect.DeepEqual(classes(res.Detections), []string{"cup"}) {
		t.Errorf("gdino-siglip shape: %v, %v; want cup from the open detector", classes(res.Detections), err)
	}
}

func TestGroundedSkipsSegmentationWithoutDetections(t *testing.T) {
	seg := &countSegmenter{n: -1}
	g := Grounded{Detector: &recDetector{}, Segmenter: seg}
	res, err := g.Infer(Call{}, nil)
	if err != nil || len(res.Detections) != 0 || res.Masks != nil || seg.n != -1 {
		t.Errorf("no detections: %+v, %v, segmenter ran=%v", res, err, seg.n != -1)
	}
	res, err = g.Infer(Call{}, []string{"cup", "tv"})
	if err != nil || len(res.Masks) != 2 || seg.n != 2 {
		t.Errorf("two detections: %d masks, %v", len(res.Masks), err)
	}
}
