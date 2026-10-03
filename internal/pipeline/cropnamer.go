package pipeline

import (
	"fmt"
	"math"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/siglip"
)

// CropNamer names boxes from their pixels: crop → embed (one batched vision-tower call) → cosine
// against each requested word's text embedding → softmax(cosines / T) → floor. It is the shared
// core of the router's SigLIP rescorer (hybrid) and textalign's crop head, which used to carry a
// copy each; each keeps ITS OWN default temperature and floor, which were measured separately and
// are deliberately not unified (BUGS_TO_FIX.md #2).
type CropNamer struct {
	CropRole string        // the vision tower's session role
	Text     *TextEmbedder // the matching text tower (same checkpoint family)
	Temp     float64       // softmax temperature when the request passes none (<= 0)
	Floor    float32       // best cosine below this rejects the crop
}

// Naming is one box's verdict: the index of the winning word and its softmax probability, or
// Word = -1 when the box is rejected (no pixels after clamping, or no word clears the floor).
type Naming struct {
	Word int
	P    float32
}

// Name names every box against words, index-aligned with boxes.
//
// Boxes that keep no pixel after clamping to the image are rejected before anything runs: they
// carry no embedding, and an unranked box would rejoin the flood a rescorer exists to drain. When
// no box is usable no tower runs at all and every box comes back rejected — a request whose only
// detection is a sub-pixel sliver must come back empty, not fail. siglip.EmbedCrops' `kept` is
// still honoured, so a row is never attributed to another box.
//
// A failing tower fails the request rather than falling back to unrescored detections: plausible
// output from a broken tower is how this project has been wrong before (a silent CPU fallback
// invalidated a whole sweep).
func (n *CropNamer) Name(c Call, boxes [][4]float64, words []string, temp float64) ([]Naming, error) {
	out := make([]Naming, len(boxes))
	usable := make([][4]float64, 0, len(boxes))
	idx := make([]int, 0, len(boxes)) // usable[j] is boxes[idx[j]]
	for i, b := range boxes {
		out[i].Word = -1
		if siglip.UsableBox(c.Img, b) {
			usable = append(usable, b)
			idx = append(idx, i)
		}
	}
	if len(usable) == 0 || len(words) == 0 {
		return out, nil
	}
	if n == nil || n.Text == nil {
		return nil, fmt.Errorf("crop namer: no towers configured")
	}
	if temp <= 0 {
		temp = n.Temp
	}
	crops, kept, err := siglip.EmbedCrops(c.Img, usable,
		func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return c.Runner.Run(n.CropRole, in) },
		c.Runner.InputNames(n.CropRole))
	if err != nil {
		return nil, fmt.Errorf("SigLIP crop tower: %w", err)
	}
	text, err := n.Text.Embed(words, c.Runner)
	if err != nil {
		return nil, err
	}
	scores, err := siglip.ScoreCrops(crops, text)
	if err != nil {
		return nil, err
	}
	nw := len(words)
	for j, k := range kept {
		row := scores[j*nw : (j+1)*nw]
		best, w := float32(math.Inf(-1)), -1
		for c, s := range row {
			if s > best {
				best, w = s, c
			}
		}
		if w < 0 || best < n.Floor {
			continue
		}
		out[idx[k]] = Naming{Word: w, P: SoftmaxAt(row, w, temp)}
	}
	return out, nil
}

// SoftmaxAt returns softmax(row / temp)[k], computed stably (max subtracted, accumulated in
// float64). It returns 0 for an index out of range and 1 for a single-word row; temp must be > 0
// (Name resolves the default first).
func SoftmaxAt(row []float32, k int, temp float64) float32 {
	if k < 0 || k >= len(row) || len(row) == 0 {
		return 0
	}
	if len(row) == 1 {
		return 1
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

// CropRescorer is the Rescorer stage over a CropNamer: every detection is renamed to its crop's
// best word and its confidence multiplied by that word's probability; rejected detections are
// dropped. The request's crop_temp, when > 0, overrides the namer's temperature.
//
// Measured on the held-out-names protocol (ovd-edge/docs/FINDINGS.md §1–§2): renaming alone is
// worth +1.4 mAP, conf × P alone +8.8, both +12.6 — SigLIP earns its place as a REJECTOR, its
// softmax flattening on background crops.
type CropRescorer struct{ Namer *CropNamer }

// Rescore implements Rescorer. `words` must be exactly the words the detections were asked
// about: scoring against a wider list is a different, worse condition (59.96 against 62.16).
func (cr CropRescorer) Rescore(c Call, dets []models.Detection, words []string) ([]models.Detection, error) {
	if len(dets) == 0 || len(words) == 0 {
		return dets, nil
	}
	boxes := make([][4]float64, len(dets))
	for i, d := range dets {
		boxes[i] = d.BBox
	}
	names, err := cr.Namer.Name(c, boxes, words, c.Prompt.CropTemp)
	if err != nil {
		return nil, err
	}
	var out []models.Detection
	for i, nm := range names {
		if nm.Word < 0 {
			continue
		}
		d := dets[i]
		d.Class = words[nm.Word]
		d.Conf *= float64(nm.P)
		out = append(out, d)
	}
	return out, nil
}
