package textalign

import (
	"fmt"
	"math"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// # Why a third scoring mode exists
//
// Folding W = a·T̂P is exact for NAMING and wrong for SELECTION, and the two failure modes
// are separable. Writing ρ = ‖P f‖ and dropping the shared bias b, the folded score is
//
//	s_fold(c) − b = ρ · (s_exact(c) − b)
//
// an identity, verified to 4.8e-6 (float32 noise) on real features. Two consequences follow,
// and they point in opposite directions:
//
//   - ρ > 0 is ONE POSITIVE SCALAR SHARED BY EVERY CLASS c, so it cannot reorder the classes
//     of a query: argmax_c s_fold = argmax_c s_exact, for every input, exactly. Measured on
//     358 held-out ETRI rows: identical predictions on 100.0000% of them, 76.26% accuracy on
//     both paths. This is an algebraic fact, not a lucky benchmark.
//   - ρ is a DIFFERENT scalar for every query i, so it does reorder queries against each
//     other. Measured across queries: Spearman 0.713 between the exact and folded ranking,
//     and only 19 of the top 100 queries are shared. The threshold moves too: s+b > 0
//     becomes s > −b/ρ, which predicted all 382 ETRI disagreements exactly.
//
// internal/models/rfdetr's postprocess does BOTH jobs from one logits tensor — it takes the
// per-query argmax (naming) but also thresholds and sorts by that same value (selection). So
// handing it folded logits gets the naming right and the box selection wrong.
//
// modeGated splits the job along the line the algebra draws: the detector's own class head
// supplies the score used for thresholding and ranking, head B supplies only the name. Then
// the fold is exact for everything it is asked to decide, and the cheap path is also the
// correct one.
//
// The detector's objectness is the standard DETR proxy, max over the REAL classes of the
// frozen head. "Real" excludes the "N/A" background column, whose INDEX IS NOT PORTABLE: the
// COCO-91 export carries it at column 0, the 22-class ETRI export at column 22. It is
// therefore located by NAME from the manifest labels, never by a hard-coded index — reading
// the background column as a class silently corrupts every score.
type scoreMode int

const (
	// modeExact keeps the ‖P f‖ normalisation: logit = a⟨t̂_c, Pf⟩/‖Pf‖ + b.
	modeExact scoreMode = iota
	// modeFolded drops it, making the head a literal nn.Linear. Same names, different
	// calibration, and a query ranking that no longer matches modeExact.
	modeFolded
	// modeGated names with the folded head and selects with the detector's objectness.
	modeGated
	// modeDual splits naming between two heads — the detector's own supervised class head for
	// the words it was trained on, head B for the rest — while still selecting with the
	// detector's objectness. See dualLogits.
	modeDual
)

func (m scoreMode) String() string {
	switch m {
	case modeExact:
		return "exact"
	case modeFolded:
		return "folded"
	case modeGated:
		return "gated"
	case modeDual:
		return "dual"
	}
	return fmt.Sprintf("scoreMode(%d)", int(m))
}

// normalize reports whether this mode divides by ‖P f‖ when computing head B's logits.
// modeGated does not: it uses the folded head precisely because the fold is exact for the
// only decision it delegates to head B, the argmax.
func (m scoreMode) normalize() bool { return m == modeExact }

// gatedLogitFloor is the logit written into every class a query did NOT win. sigmoid(-30) is
// ~9e-14, far below any usable conf_threshold, so these columns can never be selected while
// staying finite (an -Inf would poison the sigmoid and any downstream arithmetic).
const gatedLogitFloor = -30

// dualClaimThresh is the default logit the closed head must reach on a requested word to claim a
// query. -1.7346 is sigmoid 0.15, chosen from the held-out-names sweep rather than by intuition:
// across claim thresholds 0.05-0.8 the open head's mAP on five genuinely unseen names peaks
// broadly (21.5-22.3), while closed-set mAP starts falling above 0.15. 0.4 buys 0.8 more open mAP
// — inside the ~1 mAP noise floor that sweep's control established — for 0.6 of real closed-set
// accuracy, so 0.15 is the better trade.
//
// It is deliberately NOT the manifest's conf_threshold. That threshold decides which detections
// are REPORTED and is routinely set to 0.001 for evaluation so the precision-recall curve is
// complete; reusing it would make every evaluation run claim every query and hide exactly the
// failure this constant exists to prevent.
//
// Override per request with `claim_threshold` (a probability in (0,1)).
const dualClaimThresh = -1.7346

// realClassCols marks which columns of the detector's own class head are real classes, i.e.
// everything except the "N/A" background column. Matching is by name and case-insensitive,
// mirroring baseVocab, so an export that has no such column simply yields all-true.
func realClassCols(labels []string) []bool {
	out := make([]bool, len(labels))
	for i, l := range labels {
		out[i] = !strings.EqualFold(strings.TrimSpace(l), "n/a")
	}
	return out
}

// objectness returns, per query, max over the real classes of the detector's own class
// logits — the DETR objectness proxy, kept in LOGIT space so postprocess's sigmoid is the
// only place a probability is formed.
//
// cls is the detector's [1, Q, C] class tensor. A query whose real columns are all masked
// out scores gatedLogitFloor rather than silently becoming +0.
func objectness(cls engine.Tensor, real []bool, q int) ([]float32, error) {
	c := int(cls.Dim(-1))
	if c != len(real) {
		return nil, fmt.Errorf("textalign: detector class head has %d columns but the manifest lists %d labels — the labels file does not match files.%s",
			c, len(real), roleDetector)
	}
	if len(cls.Data) < q*c {
		return nil, fmt.Errorf("textalign: detector class tensor has %d values, need %d (%d queries × %d classes)",
			len(cls.Data), q*c, q, c)
	}
	out := make([]float32, q)
	for i := 0; i < q; i++ {
		row := cls.Data[i*c : (i+1)*c]
		best := float32(math.Inf(-1))
		for k, v := range row {
			if real[k] && v > best {
				best = v
			}
		}
		if math.IsInf(float64(best), -1) {
			best = gatedLogitFloor
		}
		out[i] = best
	}
	return out, nil
}

// objectnessArgmax is objectness plus the index of the real class that won, which modeDual
// needs: that winning column IS the closed head's answer, so taking it in the same pass costs
// nothing. A query with no real class at all scores -1.
func objectnessArgmax(cls engine.Tensor, real []bool, q int) (vals []float32, args []int, err error) {
	c := int(cls.Dim(-1))
	if c != len(real) {
		return nil, nil, fmt.Errorf("textalign: detector class head has %d columns but the manifest lists %d labels — the labels file does not match files.%s",
			c, len(real), roleDetector)
	}
	if len(cls.Data) < q*c {
		return nil, nil, fmt.Errorf("textalign: detector class tensor has %d values, need %d (%d queries × %d classes)",
			len(cls.Data), q*c, q, c)
	}
	vals = make([]float32, q)
	args = make([]int, q)
	for i := 0; i < q; i++ {
		row := cls.Data[i*c : (i+1)*c]
		best := float32(math.Inf(-1))
		arg := -1
		for k, v := range row {
			if real[k] && v > best {
				best, arg = v, k
			}
		}
		if arg < 0 {
			best = gatedLogitFloor
		}
		vals[i], args[i] = best, arg
	}
	return vals, args, nil
}

// dualLogits is modeDual's [Q, C] tensor. It differs from gatedLogits in ONE decision — where
// the NAME comes from — and deliberately not at all in where the SCORE comes from.
//
// That is what makes it safe. gated.go exists because RF-DETR's postprocess reads the name and
// the confidence off one tensor, so handing it two quantities on different scales gets the
// selection wrong. The closed head's logits and head B's cosine scores ARE on different scales:
// dec1's class head sits low enough that its serving threshold had to be retuned to 0.35. Filling
// some columns from one and some from the other would reintroduce precisely that bug. Here every
// column carries the SAME quantity — that query's detector objectness — so thresholding and
// ranking stay on one scale, and the two heads only disagree about what to call the object.
//
// Each head is confined to the columns it owns, and SYMMETRICALLY so. An earlier version
// restricted head B to the open columns but let the closed head use its global argmax over all
// its labels, which produced a reproducible failure: the tabletop vocabulary contains both
// "coffee" and "coffee can", so on a request for "coffee can" the detector's argmax landed on
// "coffee", the router read that as "the caller did not ask for this", and the query fell through
// to head B — which, confined to the open columns, called a coffee can a "power strip". With no
// open words in the request it was dropped entirely. The closed head must be asked what it thinks
// of the words the caller ACTUALLY requested, not merely what it likes best overall.
//
// clsRow(i) returns query i's raw detector class logits. closedOfCol maps a requested column to
// the detector label that names it (-1 if none); openCol marks the columns only head B may name.
//
// claimThresh is the ABSOLUTE logit the closed head must reach, for a word the caller actually
// requested, to claim a query.
//
// An earlier version compared that score against the closed head's own best guess overall
// (bestC >= obj[i] - margin). That is a vacuous test whenever the prompt contains the whole base
// vocabulary, because then the two maxima are THE SAME NUMBER by construction: the condition is
// always true, head B is never consulted, and the mode silently stops doing open-vocabulary at
// all. Measured: at |V|=78 it named 0 of 300 queries with an open word, and its every statistic
// was byte-identical to |V|=22. Even on a small mixed prompt it claimed 298 of 300, because on
// low-objectness background queries all 22 class logits sit within a few logits of each other.
//
// An absolute threshold asks the question that was meant all along — "does the closed head
// actually recognise one of the words you asked for?" — and a background query, whose logits are
// all low, fails it and falls through to head B.
func dualLogits(nameLogits []float32, obj []float32, clsRow func(int) []float32,
	closedOfCol []int, openCol []bool, claimThresh float32, q, c int) ([]float32, error) {
	if c == 0 {
		return nil, fmt.Errorf("textalign: empty vocabulary")
	}
	if len(nameLogits) < q*c || len(obj) < q || len(openCol) < c || len(closedOfCol) < c {
		return nil, fmt.Errorf("textalign: dual inputs too short (names %d want %d, objectness %d want %d, open mask %d want %d, closed map %d want %d)",
			len(nameLogits), q*c, len(obj), q, len(openCol), c, len(closedOfCol), c)
	}
	out := make([]float32, q*c)
	for i := range out {
		out[i] = gatedLogitFloor
	}
	for i := 0; i < q; i++ {
		// The closed head, restricted to the requested words it owns.
		colC, bestC := -1, float32(math.Inf(-1))
		if row := clsRow(i); row != nil {
			for k := 0; k < c; k++ {
				if j := closedOfCol[k]; j >= 0 && j < len(row) && row[j] > bestC {
					bestC, colC = row[j], k
				}
			}
		}
		// The closed head's exclusivity is conditional on it HAVING an opinion. While it claims
		// the query, head B may not re-name it — head B is 16 points worse on those words. Once
		// it declines, that justification is gone with it, and confining head B to the open
		// columns only loses detections: measured on a real image, the detector called an object
		// "coffee" while head B called it "coffee can", both of which are labels in this
		// vocabulary. Asked for "coffee can", a confined head B answered "power strip".
		claimed := colC >= 0 && bestC >= claimThresh
		win := colC
		if !claimed {
			win = -1
			best := float32(math.Inf(-1))
			nrow := nameLogits[i*c : (i+1)*c]
			for k := 0; k < c; k++ {
				if (openCol[k] || !claimed) && nrow[k] > best {
					best, win = nrow[k], k
				}
			}
		}
		if win >= 0 {
			out[i*c+win] = obj[i]
		}
	}
	return out, nil
}

// gatedLogits builds the [Q, C] tensor handed to RF-DETR's postprocess under modeGated: for
// each query, the class head B named gets that query's detector objectness, every other
// class gets gatedLogitFloor.
//
// Postprocess then does exactly the right thing with no changes to it: its per-query argmax
// lands on head B's class (naming), and the bestScore it thresholds, sorts and cuts by is
// sigmoid(objectness) (selection).
func gatedLogits(nameLogits []float32, obj []float32, q, c int) ([]float32, error) {
	if c == 0 {
		return nil, fmt.Errorf("textalign: empty vocabulary")
	}
	if len(nameLogits) < q*c || len(obj) < q {
		return nil, fmt.Errorf("textalign: gated inputs too short (names %d want %d, objectness %d want %d)",
			len(nameLogits), q*c, len(obj), q)
	}
	out := make([]float32, q*c)
	for i := 0; i < q; i++ {
		row := nameLogits[i*c : (i+1)*c]
		best := 0
		for k := 1; k < c; k++ {
			if row[k] > row[best] {
				best = k
			}
		}
		base := i * c
		for k := 0; k < c; k++ {
			out[base+k] = gatedLogitFloor
		}
		out[base+best] = obj[i]
	}
	return out, nil
}

// decodeGated is decode's modeGated branch: name with the folded head, select with the
// detector's own objectness, then reuse RF-DETR's postprocess for geometry as usual.
func (m *textAlign) decodeGated(h *head, boxes, cls, feats engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
	q := int(boxes.Dim(1))
	names, err := h.logits(m.proj, feats.Data, q, false)
	if err != nil {
		return models.Result{}, err
	}
	obj, err := objectness(cls, realClassCols(m.cfg.Labels), q)
	if err != nil {
		return models.Result{}, err
	}
	logits, err := gatedLogits(names, obj, q, len(h.classes))
	if err != nil {
		return models.Result{}, err
	}
	return h.rf.Postprocess(
		[]engine.Tensor{boxes, engine.F32(logits, 1, int64(q), int64(len(h.classes)))}, meta)
}

// routeVocab is the router: it reads the requested words and decides which head owns each one.
// It detects nothing. A word the detector was trained on belongs to the closed head; every other
// word belongs to head B.
//
// closedCol is indexed by DETECTOR label, openCol by REQUESTED column, because that is how each
// is consumed — closedCol answers "the detector said class j, did the caller ask for it?", openCol
// answers "may head B name column k?".
// Both outputs are indexed by REQUESTED column, which is what dualLogits consumes: closedOfCol
// answers "which detector label names this word, if any", openCol answers "may head B name it".
func routeVocab(labels, classes []string) (closedOfCol []int, openCol []bool) {
	label := make(map[string]int, len(labels))
	for j, l := range labels {
		l = strings.ToLower(strings.TrimSpace(l))
		if l == "" || l == "n/a" {
			continue // the background column names nothing
		}
		if _, dup := label[l]; !dup {
			label[l] = j
		}
	}
	closedOfCol = make([]int, len(classes))
	openCol = make([]bool, len(classes))
	for k, c := range classes {
		closedOfCol[k] = -1
		openCol[k] = true
		if j, ok := label[strings.ToLower(strings.TrimSpace(c))]; ok {
			closedOfCol[k] = j
			openCol[k] = false // the closed head owns this word; head B must not re-name it
		}
	}
	return closedOfCol, openCol
}

// decodeDual is decode's modeDual branch: route the vocabulary between the two heads, name each
// query with whichever head owns it, and select with the detector's objectness as modeGated does.
// claimThreshold converts a per-request probability into the logit dualLogits compares against,
// falling back to the package default. Out-of-range values are ignored rather than clamped: 0 and
// 1 are not thresholds a caller can have meant, and silently turning them into ±inf would make
// the closed head claim everything or nothing with no error.
func claimThreshold(p float64) float32 {
	if p <= 0 {
		// 0 is the zero value of an omitted field, so it has to mean "unset". A caller who
		// genuinely wants the closed head to claim everything can pass a small positive number.
		return dualClaimThresh
	}
	if p >= 1 {
		// p >= 1 was ALSO treated as unset, which silently did the opposite of what it asks:
		// "require certainty before claiming" came back as the default threshold. Nobody's zero
		// value is 1, so there is no ambiguity to protect against, and "the closed head never
		// claims" is both the plain reading and the configuration needed to measure the open
		// head's ceiling. A finite sentinel rather than +Inf keeps every comparison well-defined.
		return neverClaim
	}
	return float32(math.Log(p / (1 - p)))
}

// neverClaim is a logit no class score can reach, so the closed head declines every query.
const neverClaim = float32(1e9)

func (m *textAlign) decodeDual(h *head, boxes, cls, feats engine.Tensor, meta models.PreprocessMeta, claim float32) (models.Result, error) {
	q := int(boxes.Dim(1))
	names, err := h.logits(m.proj, feats.Data, q, false)
	if err != nil {
		return models.Result{}, err
	}
	obj, _, err := objectnessArgmax(cls, realClassCols(m.cfg.Labels), q)
	if err != nil {
		return models.Result{}, err
	}
	closedOfCol, openCol := routeVocab(m.cfg.Labels, h.classes)
	nCls := int(cls.Dim(-1))
	clsRow := func(i int) []float32 {
		if (i+1)*nCls > len(cls.Data) {
			return nil
		}
		return cls.Data[i*nCls : (i+1)*nCls]
	}
	logits, err := dualLogits(names, obj, clsRow, closedOfCol, openCol, claim, q, len(h.classes))
	if err != nil {
		return models.Result{}, err
	}
	return h.rf.Postprocess(
		[]engine.Tensor{boxes, engine.F32(logits, 1, int64(q), int64(len(h.classes)))}, meta)
}
