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
	meta models.PreprocessMeta, claim float32, r models.Runner) (models.Result, error) {
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

	named, err := m.nameOpenCrops(img, res.Detections, h.classes, r)
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
	r models.Runner) ([]models.Detection, error) {
	idx := make([]int, 0, len(dets))
	boxes := make([][4]float64, 0, len(dets))
	for i, d := range dets {
		if d.Class == openSentinel {
			idx = append(idx, i)
			boxes = append(boxes, d.BBox)
		}
	}
	if len(idx) == 0 {
		return dets, nil
	}
	if len(classes) == 0 {
		// No vocabulary at all: these boxes have no name available, and emitting them under the
		// sentinel would leak an internal label into a client's JSON.
		return dropIndices(dets, idx), nil
	}

	// Degenerate boxes are skipped rather than fatal, so `kept` says which of `boxes` actually
	// produced a row. Losing that mapping would rename detections with another box's embedding.
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
	drop := make([]int, 0, len(idx))
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
		// Conf stays the DETECTOR's objectness. The crop similarity decided the name, not
		// whether the object is there, and mixing a cosine into a column postprocess already
		// thresholded on a different scale is the bug gated.go exists to prevent.
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
