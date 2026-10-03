package pipeline

import (
	"fmt"
	"strings"
	"sync"

	"visionserve/internal/models"
)

// QueryHead answers words outside the closed detector's vocabulary from the closed detector's
// OWN forward pass — its query features — instead of a second detector (the distilled fast path,
// internal/models/hybrid). Boxes are in ORIGINAL image coordinates.
type QueryHead interface {
	DetectQueries(c Call, pass *ClosedPass, words []string) ([]models.Detection, error)
}

// Router answers each requested word with the detector that is the right tool for it: the
// closed-set detector (RF-DETR, fast, supervised) for the words it was trained on, an open-
// vocabulary detector (GroundingDINO) for the rest — or, when a QueryHead is wired, the head over
// the closed detector's own queries. The two answers are concatenated; open answers are rescored
// against exactly the words they were asked about; with a Segmenter every detection gets a mask.
//
// Configurations (internal/models/hybrid): rfdetr-gdino = Closed + Open (+ Rescore, + Head,
// + Segment); gdino-siglip = Open + Rescore (+ Segment), no closed detector.
//
// It reads text and dispatches; it never detects anything itself.
type Router struct {
	Name    string          // the model name, for error messages
	Vocab   map[string]bool // the closed detector's class names in NormClass form (the routing set)
	Closed  *Closed         // nil: no closed-set detector, every word goes to Open
	Head    QueryHead       // optional; replaces Open for unknown words (requires Rescore)
	Open    Detector        // the open-vocabulary detector
	Rescore Rescorer        // optional: rescores the open answers (Head's or Open's)
	Segment Segmenter       // optional: one mask per detection

	// OpenLock, when set, is held from just before the Open pass to the end of the request (its
	// rescoring and the masks after it): the GroundingDINO section. Requests the closed detector
	// or the head answer alone never take it and stay concurrent.
	OpenLock sync.Locker
}

// Infer runs one request.
func (rt *Router) Infer(c Call) (models.Result, error) {
	classes := ParseClasses(c.Prompt.Text)
	if rt.Closed == nil && len(classes) == 0 {
		return models.Result{}, fmt.Errorf("%s: a text prompt is required (e.g. \"cup. towel.\") — "+
			"there is no closed-set detector to answer an empty one", rt.Name)
	}
	known, unknown := rt.Partition(classes)

	// Each detector is asked ONLY about the words it is the right tool for. The alternative — the
	// whole request going to GroundingDINO as soon as one word is out of vocabulary — threw away
	// the in-domain specialist for the words it was trained on: "cup. zebra." lost RF-DETR's cup.
	pass, dets, err := rt.closedBranch(c, classes, known)
	if err != nil {
		return models.Result{}, err
	}
	if len(unknown) > 0 && rt.Head != nil {
		d, err := rt.headBranch(c, pass, unknown)
		if err != nil {
			return models.Result{}, err
		}
		dets = append(dets, d...)
		unknown = nil // answered; the open detector is not consulted
	}
	if len(unknown) > 0 {
		if rt.OpenLock != nil {
			rt.OpenLock.Lock()
			defer rt.OpenLock.Unlock()
		}
		d, err := rt.openBranch(c, unknown)
		if err != nil {
			return models.Result{}, err
		}
		dets = append(dets, d...)
	}

	res := models.Result{Detections: dets}
	if rt.Segment == nil || len(dets) == 0 {
		return res, nil
	}
	masks, err := rt.Segment.Segment(c, dets)
	if err != nil {
		return models.Result{}, err
	}
	res.Masks = masks
	return res, nil
}

// closedBranch runs the closed detector at most ONCE per request, when it has something to
// answer: an empty prompt (its own vocabulary IS the prompt then), a known word, or a head that
// will score its queries. The same pass serves both: that sharing is the whole latency argument
// for the head — the open branch costs a matmul rather than a second detector (ovd-edge FINDINGS
// §10-§11). Its detections are the known words' only; none when the caller asked only about
// unknown words.
func (rt *Router) closedBranch(c Call, classes, known []string) (*ClosedPass, []models.Detection, error) {
	if rt.Closed == nil || (len(classes) > 0 && len(known) == 0 && rt.Head == nil) {
		return nil, nil, nil
	}
	pass, err := rt.Closed.Forward(c)
	if err != nil {
		return nil, nil, err
	}
	switch {
	case len(classes) > 0 && len(known) == 0:
		return pass, nil, nil // the closed head has no answer for these words
	case len(known) > 0:
		return pass, FilterByClass(pass.Dets, known), nil
	}
	return pass, pass.Dets, nil
}

// headBranch answers the unknown words from the closed pass's queries, then rescores them. The
// head alone is a ranking (21.53 mAP); rescoring on top is what makes it usable (46.41).
func (rt *Router) headBranch(c Call, pass *ClosedPass, unknown []string) ([]models.Detection, error) {
	if pass == nil {
		return nil, fmt.Errorf("%s: the distilled head needs the closed detector's forward pass", rt.Name)
	}
	d, err := rt.Head.DetectQueries(c, pass, unknown)
	if err != nil {
		return nil, err
	}
	return rt.Rescore.Rescore(c, d, unknown)
}

// openBranch asks the open detector about the unknown words only, then rescores ONLY its
// detections, against only those words. The closed detector's are not touched: its supervised
// head already answered on one calibrated scale, and the 17-name control column must not move.
//
// Asking GroundingDINO about a subset is not a pure subset of the old behaviour: its fusion and
// decoder layers attend over the whole prompt, so dropping the in-vocabulary words removes them as
// distractors and can move the remaining scores slightly. Prompt splitting is a modelling
// decision, not just a dispatch optimisation.
func (rt *Router) openBranch(c Call, unknown []string) ([]models.Detection, error) {
	d, err := rt.Open.Detect(c, unknown)
	if err != nil {
		return nil, err
	}
	if rt.Rescore == nil {
		return d, nil
	}
	return rt.Rescore.Rescore(c, d, unknown)
}

// Partition splits the requested classes into the ones the closed detector was trained on and
// the rest, in request order. An empty prompt yields two empty slices.
func (rt *Router) Partition(classes []string) (known, unknown []string) {
	for _, c := range classes {
		if rt.Vocab[c] {
			known = append(known, c)
		} else {
			unknown = append(unknown, c)
		}
	}
	return known, unknown
}

// ParseClasses splits a GroundingDINO-style prompt ("cat. remote.") into class names in
// NormClass form ([cat, remote]), de-duplicated in first-seen order. An empty prompt yields nil.
//
// De-duplication is not cosmetic: the rescorer softmaxes over the word list, so "zebra. zebra."
// put two identical rows in the softmax and halved every rescored confidence (and asked
// GroundingDINO the same phrase twice). Whitespace collapsing keeps "dining  table" on
// RF-DETR's "dining table" route instead of sending it to GroundingDINO.
func ParseClasses(text string) []string {
	var out []string
	seen := make(map[string]bool)
	for _, p := range strings.Split(text, ".") {
		p = NormClass(p)
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

// NormClass is the one normal form class names are compared in: lowercase, single spaces.
func NormClass(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

// ClosedVocab is the routing set of a closed detector's labels: NormClass form, without the
// empty and "n/a" (background) entries.
func ClosedVocab(labels []string) map[string]bool {
	vocab := make(map[string]bool, len(labels))
	for _, l := range labels {
		l = NormClass(l)
		if l != "" && l != "n/a" {
			vocab[l] = true
		}
	}
	return vocab
}

// FilterByClass keeps the detections whose class (in NormClass form) is one of classes.
func FilterByClass(dets []models.Detection, classes []string) []models.Detection {
	want := make(map[string]bool, len(classes))
	for _, c := range classes {
		want[c] = true
	}
	out := make([]models.Detection, 0, len(dets))
	for _, d := range dets {
		if want[NormClass(d.Class)] {
			out = append(out, d)
		}
	}
	return out
}
