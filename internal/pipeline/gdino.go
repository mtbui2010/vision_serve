package pipeline

import (
	"fmt"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/groundingdino"
)

// GDINO is the GroundingDINO Detector stage: open-vocabulary, text-prompted. Each word is one
// "."-separated phrase of the prompt the model sees; detections are labelled with the word as
// given.
type GDINO struct {
	Role string                   // the GroundingDINO session's role
	Tok  *groundingdino.Tokenizer // loaded from the vocab.txt next to the weights
	// Joint scores the whole prompt in one pass; set from groundingdino.JointTextPassOrSafe at
	// load (the defective community export must be run once per phrase).
	Joint bool
	// BoxThresh and TextThresh are the manifest's thresholds, or 0 for the built-in defaults; the
	// request's overrides win either way (groundingdino.Thresholds). A composite whose manifest
	// thresholds belong to another detector leaves them 0.
	BoxThresh, TextThresh float64
}

// NewGDINO loads the tokenizer next to the GroundingDINO weights (groundingdino.VocabPath) and
// probes which text-mask regime the weights implement; box/text are the manifest's thresholds
// (0 = default).
func NewGDINO(role, weights, dir string, box, text float64) (*GDINO, error) {
	tok, err := groundingdino.LoadTokenizer(groundingdino.VocabPath(weights, dir))
	if err != nil {
		return nil, err
	}
	return &GDINO{
		Role: role, Tok: tok, BoxThresh: box, TextThresh: text,
		Joint: groundingdino.JointTextPassOrSafe(weights),
	}, nil
}

// Detect implements Detector. The words are joined back into a "w1. w2." prompt, which
// groundingdino.Detect splits into exactly these phrases again.
func (g *GDINO) Detect(c Call, words []string) ([]models.Detection, error) {
	box, text := groundingdino.Thresholds(g.BoxThresh, g.TextThresh, c.Prompt)
	run := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return c.Runner.Run(g.Role, in) }
	return groundingdino.Detect(c.Img, joinPhrases(words), g.Tok, run, c.Runner.OutputNames(g.Role), box, text,
		groundingdino.WithJointTextPass(g.Joint))
}

// TextPhrases splits a GroundingDINO prompt into the phrases a GDINO stage takes as words
// (groundingdino.SplitPhrases), failing as groundingdino.Detect does on a prompt with none.
// Callers reject an empty prompt first, with their own message.
func TextPhrases(text string) ([]string, error) {
	words := groundingdino.SplitPhrases(text)
	if len(words) == 0 {
		return nil, fmt.Errorf("grounding-dino: prompt %q holds no class phrase", text)
	}
	return words, nil
}

// joinPhrases rebuilds a GroundingDINO-style prompt from words ([cat, remote] -> "cat. remote.").
func joinPhrases(words []string) string {
	var b strings.Builder
	for _, w := range words {
		b.WriteString(w)
		b.WriteString(". ")
	}
	return strings.TrimSpace(b.String())
}
