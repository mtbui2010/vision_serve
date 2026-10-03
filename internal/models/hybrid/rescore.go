package hybrid

import (
	"fmt"
	"image"
	"path/filepath"
	"strings"

	"visionserve/internal/models"
	"visionserve/internal/models/promptens"
	"visionserve/internal/models/siglip"
	"visionserve/internal/pipeline"
)

// roleCrop and roleText are an OPTIONAL pair of SigLIP towers. Declaring BOTH files.crop and
// files.text makes the router rescore GroundingDINO's detections with SigLIP-crop before
// returning them; declaring neither leaves the router byte-identical to what it does today.
// Declaring exactly one is a manifest error rather than a silent half-feature — a crop tower
// with no text tower has nothing to score against.
const (
	roleCrop = "crop"
	roleText = "text"
)

// templatesFile optionally overrides the prompt ensemble, next to the manifest. It is part of
// the measured contract, not a cosmetic knob: the +12.6 below was measured with the ten
// templates in measuredTemplates, and a different ensemble moves every cosine.
const templatesFile = "templates.txt"

// cropTemp turns SigLIP cosines into a distribution over the words GroundingDINO was asked
// about. The reported confidence is multiplied by that probability, so a crop SigLIP cannot
// place among the requested words sinks below the true positives it used to outrank.
//
// 0.05, the vertex of a sweep SERVED through this code on the held-out-names protocol, after the
// SigLIP padding fix (BUGS_TO_FIX.md #5) and the RF-DETR squash fix (#1), 29 Sep 2026:
//
//	T          0.02   0.03   0.05   0.07   0.1
//	5-held-out 60.94  62.01  62.47  61.92  60.90      (17-trained control 89.75 throughout)
//
// ovd-edge/docs/FINDINGS.md §16 found the same vertex in-process with embeddings that never
// touched the Go tokenizer (0.02 62.23, 0.05 63.90). The earlier choice of 0.02 rested on sweeps
// run through the Go SigLIP tower BEFORE the padding fix, on embeddings 0.706 cosine from the
// right ones, so its argument did not survive. On the fast path's 22-crop budget 0.02 and 0.05
// are within 0.02 mAP (FINDINGS §18), so this only moves the router.
//
// CAVEAT, still true: every point above is on the same 247 images; there is no independent split
// to certify 0.05 over its neighbours. It stays a per-request field (`crop_temp`) for a domain
// whose cosines are distributed differently. textalign/crophead.go keeps its own constant: that
// head was never re-swept after the tokenizer fix.
const cropTemp = 0.05

// cropNameFloor is the cosine below which no requested word describes the crop well enough to
// keep the detection at all. Same value and reasoning as textalign/crophead.go: SigLIP cosines
// are small in absolute terms, so 0.0 keeps anything positively aligned and drops the rest.
const cropNameFloor = 0.0

// measuredTemplates is the ten-template ensemble every number in FINDINGS §2 was measured with
// (headb/targets.py::TEMPLATES, shipped as reg_final/map-ta-a/templates.txt). It is the fallback
// rather than the single "a photo of a {}." textalign falls back to, because a manifest that
// simply forgets templates.txt would otherwise serve a DIFFERENT head than the one measured and
// no error would say so.
var measuredTemplates = []string{
	"a photo of a {}.",
	"a photo of the {}.",
	"a close-up photo of a {}.",
	"a cropped photo of a {}.",
	"a photo of a {} on a table.",
	"a bad photo of a {}.",
	"a blurry photo of a {}.",
	"itap of a {}.",
	"a photo of one {}.",
	"there is a {} in the scene.",
}

// newRescorer builds the optional SigLIP crop namer — the crop tower, the text tower with its
// tokenizer and prompt ensemble, and a per-word cache of the embedded words (without it the text
// tower runs on EVERY request and latency scales with the number of unknown WORDS rather than the
// number of crops, ~4.3 ms per word per request) — or returns nil when the manifest declares
// neither tower.
func newRescorer(cfg models.Config) (*pipeline.CropNamer, error) {
	crop, text := strings.TrimSpace(cfg.Files[roleCrop]), strings.TrimSpace(cfg.Files[roleText])
	if crop == "" && text == "" {
		return nil, nil
	}
	if crop == "" || text == "" {
		return nil, fmt.Errorf("hybrid: SigLIP rescoring needs BOTH files.%s and files.%s "+
			"(got crop=%q text=%q) — a crop tower has nothing to score against on its own",
			roleCrop, roleText, crop, text)
	}
	tok, err := siglip.LoadTokenizer(filepath.Dir(text))
	if err != nil {
		return nil, fmt.Errorf("hybrid: SigLIP text tower tokenizer: %w", err)
	}
	tmpl, err := promptens.Load(filepath.Join(cfg.Dir, templatesFile), measuredTemplates)
	if err != nil {
		return nil, err
	}
	return newNamer(pipeline.SigLIPTokenizer{T: tok}, tmpl), nil
}

// newNamer is the router's configuration of the shared crop namer: its own temperature and floor.
func newNamer(tok pipeline.TextTokenizer, tmpl []string) *pipeline.CropNamer {
	return &pipeline.CropNamer{
		CropRole: roleCrop,
		Text:     pipeline.NewTextEmbedder(roleText, tok, tmpl, 0),
		Temp:     cropTemp,
		Floor:    cropNameFloor,
	}
}

// rescore renames and re-weights GroundingDINO's detections with SigLIP-crop, and is the whole
// of the +12.6 mAP this file exists for.
//
// The decomposition matters, because it is not what a namer is normally added for. On the
// held-out-names protocol (ovd-edge/docs/FINDINGS.md §1–§2), GroundingDINO already covers 90.2 %
// of the held-out ground-truth boxes and already names 83.6 % of the landing ones correctly.
// Perfect naming would be worth +7.6 mAP. Perfect REJECTION would be worth +27.6: 538 of its 849
// boxes land on nothing, and that flood is the failure. Measured separately:
//
//	renaming only                       49.54 -> 50.93   (+1.4)
//	GroundingDINO's name, conf x P      49.54 -> 58.32   (+8.8)
//	both (this function)                49.54 -> 62.16   (+12.6)
//
// So SigLIP earns its place as a RESCORER. The mechanism is its uncertainty, not its knowledge:
// on a background crop it is confident about no word, the softmax flattens, and the detection
// sinks. Its 96.8 % naming accuracy is nearly worthless by comparison.
//
// `words` must be exactly the words routed to GroundingDINO. Scoring against the whole prompt
// instead is a different, WORSE condition (59.96 against 62.16) — and a wrong one here anyway,
// since RF-DETR has already answered for the known words and these boxes were never candidates
// for them.
//
// Detections whose box is degenerate are DROPPED: zero width or height after clamping to the
// image, which a low threshold produces for real (zero-width queries, slivers on the frame edge).
// They carry no pixels and so no embedding, and returning one with an unrescored confidence
// would put an unranked box back into the flood this exists to drain. The shared namer
// (pipeline.CropNamer, textalign's crop head is the same code) filters them before the crop
// tower, so a request whose only detection is such a sliver comes back empty instead of failing,
// and keeps the crop rows and the detection list from drifting apart.
func (m *hybrid) rescore(img image.Image, dets []models.Detection, words []string,
	temp float64, r models.Runner) ([]models.Detection, error) {
	if m.rs == nil || len(dets) == 0 || len(words) == 0 {
		return dets, nil
	}
	out, err := pipeline.CropRescorer{Namer: m.rs}.Rescore(
		pipeline.Call{Img: img, Prompt: models.Prompt{CropTemp: temp}, Runner: r}, dets, words)
	if err != nil {
		return nil, fmt.Errorf("hybrid: %w", err)
	}
	return out, nil
}
