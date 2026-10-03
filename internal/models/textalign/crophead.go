package textalign

import (
	"fmt"
	"image"
	"math"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/siglip"
)

// roleCrop is an OPTIONAL SigLIP vision tower. Declaring files.crop switches `method: dual`'s
// open head from head B (which reads the detector's query features) to SigLIP-crop (which reads
// the image). On five names held out of every trained component, SigLIP-crop scores 85.3 against
// head B's 46.7 — head B loses to simply running the teacher on the crop (docs/FINDINGS.md §8-§9).
//
// When files.crop is absent the open head stays head B, so this is additive.
const roleCrop = "crop"

// openSentinel is the internal class name given to a query the closed head declined. It exists
// because the crop namer CANNOT run inside the per-query decode: it crops real pixels, and at
// ~1.4 ms per crop, naming all 300 object queries would cost ~420 ms against a 20 ms detector.
//
// So naming happens in two passes. The sentinel lets an unclaimed query keep the detector's
// objectness and survive postprocess — thresholding, sorting and the max_detections cut — and
// only the handful that survive (3-15 in a typical scene) are ever cropped. That is also exactly
// the batch size the latency measurements assume.
//
// It carries a NUL so it cannot collide with a real class name from a user prompt.
const openSentinel = "\x00open"

// dualLogitsSentinel is dualLogits for the crop path: same claim rule, but a query the closed
// head declines goes to the sentinel column instead of being named by head B. The output is
// [Q, C+1]; column C is the sentinel.
//
// Like dualLogits, every column carries the detector's objectness, so postprocess still
// thresholds and ranks on one scale.
func dualLogitsSentinel(obj []float32, clsRow func(int) []float32, closedOfCol []int,
	claimThresh float32, q, c int) ([]float32, error) {
	if len(obj) < q || len(closedOfCol) < c {
		return nil, fmt.Errorf("textalign: sentinel inputs too short (objectness %d want %d, closed map %d want %d)",
			len(obj), q, len(closedOfCol), c)
	}
	width := c + 1
	out := make([]float32, q*width)
	for i := range out {
		out[i] = gatedLogitFloor
	}
	for i := 0; i < q; i++ {
		colC, bestC := -1, float32(math.Inf(-1))
		if row := clsRow(i); row != nil {
			for k := 0; k < c; k++ {
				if j := closedOfCol[k]; j >= 0 && j < len(row) && row[j] > bestC {
					bestC, colC = row[j], k
				}
			}
		}
		win := c // the sentinel
		if colC >= 0 && bestC >= claimThresh {
			win = colC
		}
		out[i*width+win] = obj[i]
	}
	return out, nil
}

// decodeDualCrop is decodeDual with SigLIP-crop as the open head.
//
// Pass 1 names with the closed head and marks the rest, then lets RF-DETR's own postprocess do
// the geometry, thresholding and top-k as usual. Pass 2 crops only the marked survivors and names
// them in ONE batched vision-tower call.
func (m *textAlign) decodeDualCrop(h *head, img image.Image, boxes, cls engine.Tensor,
	meta models.PreprocessMeta, claim float32, temp float64, r models.Runner) (models.Result, error) {
	q := int(boxes.Dim(1))
	c := len(h.classes)

	obj, _, err := objectnessArgmax(cls, realClassCols(m.cfg.Labels), q)
	if err != nil {
		return models.Result{}, err
	}
	closedOfCol, _ := routeVocab(m.cfg.Labels, h.classes)

	nCls := int(cls.Dim(-1))
	clsRow := func(i int) []float32 {
		if (i+1)*nCls > len(cls.Data) {
			return nil
		}
		return cls.Data[i*nCls : (i+1)*nCls]
	}
	logits, err := dualLogitsSentinel(obj, clsRow, closedOfCol, claim, q, c)
	if err != nil {
		return models.Result{}, err
	}

	// A sub-model whose labels carry the sentinel, so postprocess can name it. Building one is
	// free — it owns no session, only pre/postprocess.
	rf, err := newRFDETR(m.cfg, append(append([]string{}, h.classes...), openSentinel))
	if err != nil {
		return models.Result{}, err
	}
	res, err := rf.Postprocess(
		[]engine.Tensor{boxes, engine.F32(logits, 1, int64(q), int64(c+1))}, meta)
	if err != nil {
		return models.Result{}, err
	}

	named, err := m.nameOpenCrops(img, res.Detections, h.classes, temp, r)
	if err != nil {
		return models.Result{}, err
	}
	res.Detections = named
	return res, nil
}

// nameOpenCrops replaces every sentinel-labelled detection with a SigLIP-crop name. Detections
// the closed head named pass through untouched, and the surviving order is preserved: callers
// index masks against detections.
//
// It scores against ALL requested words, closed ones included, and that is the correction to an
// earlier version that confined it to the open words. Confining it looked like the right way to
// stop the crop head second-guessing the supervised head — but the closed head has already had
// first refusal on these queries, so there is nothing left to protect, and the confinement forced
// every declined background query to carry one of the open names. Measured on the held-out-names
// protocol: the crop head named 21144 detections with a held-out name against head B's 7515, and
// its precision at the top of the ranking was WORSE (16.5% vs 26.7%) despite naming detected
// objects far better (91.3% vs 51.1% top-1). Head B was free to call those queries "cup" and be
// scored as a cup; the crop head could only call them "hat".
//
// This is the same rule head B already follows — once the closed head declines, exclusivity
// lapses — and the inconsistency was an oversight, not a design.
func (m *textAlign) nameOpenCrops(img image.Image, dets []models.Detection, classes []string,
	temp float64, r models.Runner) ([]models.Detection, error) {
	idx := make([]int, 0, len(dets))
	boxes := make([][4]float64, 0, len(dets))
	var unusable []int // marked detections whose box has no pixels: nothing to name them from
	for i, d := range dets {
		if d.Class != openSentinel {
			continue
		}
		if len(classes) > 0 && !siglip.UsableBox(img, d.BBox) {
			unusable = append(unusable, i)
			continue
		}
		idx = append(idx, i)
		boxes = append(boxes, d.BBox)
	}
	if len(idx) == 0 {
		if len(unusable) > 0 {
			// Every marked box is degenerate (e.g. one sub-pixel sliver on the frame edge): drop
			// them and return the rest. Handing CropTensor nothing usable is an error by its
			// contract, and it used to fail the whole request here.
			return dropIndices(dets, unusable), nil
		}
		return dets, nil
	}
	if len(classes) == 0 {
		// No vocabulary at all: these boxes have no name available, and emitting them under the
		// sentinel would leak an internal label into a client's JSON.
		return dropIndices(dets, idx), nil
	}

	// Degenerate boxes were filtered above, but `kept` is still honoured: it says which of
	// `boxes` actually produced a row, and losing that mapping would rename detections with
	// another box's embedding.
	crops, kept, err := siglip.EmbedCrops(img, boxes,
		func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(roleCrop, in) },
		r.InputNames(roleCrop))
	if err != nil {
		return nil, fmt.Errorf("textalign: crop namer: %w", err)
	}

	// Cached: the text tower is ~4.3 ms per word per request, and a 78-word vocabulary re-embedded
	// on every image was costing more than the crops it exists to name.
	text, err := m.openVocabEmbeddings(classes, r)
	if err != nil {
		return nil, err
	}

	scores, err := siglip.ScoreCrops(crops, text)
	if err != nil {
		return nil, err
	}

	n := len(classes)
	// Every marked detection is dropped unless the loop below names it. A box that was skipped as
	// degenerate never reaches the namer, and must not survive carrying the sentinel.
	drop := make([]int, 0, len(idx)+len(unusable))
	drop = append(drop, unusable...)
	named := make(map[int]bool, len(kept))
	for _, k := range kept {
		named[idx[k]] = true
	}
	for _, di := range idx {
		if !named[di] {
			drop = append(drop, di)
		}
	}
	for j, k := range kept {
		di := idx[k]
		best, bestK := float32(math.Inf(-1)), -1
		for k := 0; k < n; k++ {
			if s := scores[j*n+k]; s > best {
				best, bestK = s, k
			}
		}
		if bestK < 0 || best < cropNameFloor {
			drop = append(drop, di)
			continue
		}
		dets[di].Class = classes[bestK]
		// Conf becomes P(object) x P(this name | crop).
		//
		// It used to stay the detector's raw objectness, on the reasoning that the crop head
		// decided the NAME and not whether the object is there. That reasoning conflated two
		// decisions. Objectness is the detector's confidence that the box is one of the classes
		// IT knows — and for a query the closed head declined, none of those is the name being
		// reported. A hat was going out carrying the detector's confidence that it is a towel.
		//
		// Measured consequence: `hat` was named correctly 88% of the time and scored 3.82 AP,
		// because correctly-named hats ranked below confidently-wrong queries.
		//
		// This does NOT reintroduce what gated.go warns about. Selection — thresholding, ranking
		// and the top-k cut inside postprocess — still runs on pure objectness, one scale, and
		// has already happened by the time this line executes. What changes is the number
		// REPORTED for a detection whose name came from elsewhere, and reporting the joint
		// quantity is the honest answer to "how sure are you this is a hat".
		dets[di].Conf *= float64(softmaxAt(scores[j*n:(j+1)*n], bestK, temp))
	}
	if len(drop) > 0 {
		dets = dropIndices(dets, drop)
	}
	return dets, nil
}

// openVocabEmbeddings is embedVocab with a cache, keyed the same way head B's is: by the prompt
// templates plus the class list, so two requests asking for the same open words share one text
// tower call.
//
// Without it the tower ran on EVERY request. Measured, latency scaled with the number of open
// WORDS rather than the number of crops — about 4.3 ms per word, 12 templates each — so a 78-word
// vocabulary cost ~306 ms per image, far more than the crops the head exists to name.
func (m *textAlign) openVocabEmbeddings(classes []string, r models.Runner) ([][]float32, error) {
	key := vocabKey(m.tmpl, classes)

	m.mu.RLock()
	rows := m.textCache[key]
	m.mu.RUnlock()
	if rows != nil {
		return rows, nil
	}

	rows, err := m.embedVocab(classes, r) // already template-averaged and L2-normalised
	if err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if existing := m.textCache[key]; existing != nil {
		return existing, nil // lost a race; both are identical, keep the published one
	}
	if m.textCache == nil {
		m.textCache = map[string][][]float32{}
	}
	if len(m.textOrder) >= maxVocabCache {
		delete(m.textCache, m.textOrder[0])
		m.textOrder = m.textOrder[1:]
	}
	m.textCache[key] = rows
	m.textOrder = append(m.textOrder, key)
	return rows, nil
}

// cropNameFloor is the cosine below which no requested open word describes the crop well enough
// to name it. SigLIP cosines are small in absolute terms; 0.0 keeps anything positively aligned.
// Like dualClaimThresh this is a starting value that has not been swept.
const cropNameFloor = 0.0

// dropIndices removes the given detection indices, preserving order.
func dropIndices(dets []models.Detection, drop []int) []models.Detection {
	kill := make(map[int]bool, len(drop))
	for _, i := range drop {
		kill[i] = true
	}
	out := dets[:0]
	for i, d := range dets {
		if !kill[i] {
			out = append(out, d)
		}
	}
	return out
}

// hasCropHead reports whether the manifest wired a SigLIP vision tower.
func (m *textAlign) hasCropHead() bool { return strings.TrimSpace(m.cfg.Files[roleCrop]) != "" }

// cropTemp is the softmax temperature that turns SigLIP cosines into a distribution over the
// requested words. Lower is more decisive.
//
// 0.02 is NOT a verified optimum any more. The sweeps that chose it (4 vocabulary sizes x 9
// temperatures on the held-out-names protocol) all ran text through the Go SigLIP tower BEFORE
// the tokenizer padded with </s> instead of <pad> (BUGS_TO_FIX.md #5), i.e. on embeddings 0.706
// cosine away from the correct ones, so the argument attached to this value does not stand.
// Re-derived with correct embeddings on the hybrid router's rescorer, the vertex is 0.05
// (hybrid/rescore.go, ovd-edge/docs/FINDINGS.md §16); this head has not been re-swept, and the
// optimum depends on the detection budget (FINDINGS §18), so the router's number is not copied
// here. Re-sweep before relying on it.
//
// `crop_temp` remains a per-request field, but for a different reason than a different vocabulary
// SIZE: a different namer, or a domain whose cosines are distributed differently, would move this.
// No temperature is "true" in any case, since this export drops SigLIP's learned scale and bias.
const cropTemp = 0.02

// softmaxAt returns the softmax probability of index k, computed in a numerically stable way. A
// non-positive temperature would divide by zero or invert the ordering, so it falls back to the
// package default rather than producing silent nonsense.
func softmaxAt(row []float32, k int, temp float64) float32 {
	if k < 0 || k >= len(row) || len(row) == 0 {
		return 0
	}
	if len(row) == 1 {
		return 1
	}
	if temp <= 0 {
		temp = cropTemp
	}
	max := row[0]
	for _, v := range row[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v-max) / temp)
	}
	if sum == 0 {
		return 0
	}
	return float32(math.Exp(float64(row[k]-max)/temp) / sum)
}
