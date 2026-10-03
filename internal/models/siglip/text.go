package siglip

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/clip"
	"visionserve/internal/vision/util"
)

func init() {
	models.Register("siglip-text", NewText)
}

// roleModel is the single ONNX session the text tower needs (manifest `files: model:`).
const roleModel = "model"

// inputIDsName is the graph input name of a standard SigLIP text export. If the manifest points
// at a differently-named export, Infer falls back to the session's first declared input — the
// same accommodation internal/models/clip makes.
const inputIDsName = "input_ids"

// textModel is the SigLIP text tower (Apache-2.0).
//
// A PipelineModel rather than a plain Model because it is PROMPT-driven: its input is a string,
// not an image (CLAUDE.md). It owns no session — lifecycle.Manager loads and keeps role "model"
// alive; Infer only calls it through the Runner. The image argument is ignored.
//
// Deliberately a sibling of internal/models/clip's textModel rather than a generalisation of it:
// the two differ in tokenizer (SentencePiece Unigram vs byte-level BPE), context length (64 vs
// 77) and embedding width (768 vs 512), and folding those into one type would hide exactly the
// details that matter when a head is distilled against one of them and served with the other.
type textModel struct {
	cfg models.Config
	tok *Tokenizer
}

// NewText builds the text tower, loading tokenizer.json once from the model directory. The
// resulting Tokenizer is immutable, so concurrent Infer calls need no locking.
func NewText(cfg models.Config) (models.Base, error) {
	if cfg.Dir == "" {
		return nil, fmt.Errorf("siglip-text: model directory unknown, cannot load tokenizer.json")
	}
	tok, err := LoadTokenizer(cfg.Dir)
	if err != nil {
		return nil, err
	}
	return &textModel{cfg: cfg, tok: tok}, nil
}

func (m *textModel) Name() string      { return m.cfg.Name }
func (m *textModel) Task() models.Task { return models.TaskEmbed }

// Roles declares the single ONNX session this model needs.
func (m *textModel) Roles() []string { return []string{roleModel} }

// Infer embeds prompt.Text and returns one L2-normalised row per phrase.
//
// Phrase splitting is clip.SplitPhrases, so a prompt means the same thing whichever text tower a
// head was distilled against ("cup. remote. water bottle." → three embeddings, in that order).
func (m *textModel) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	phrases := clip.SplitPhrases(prompt.Text)
	if len(phrases) == 0 {
		return models.Result{}, models.BadPrompt(fmt.Errorf(
			"siglip-text: empty prompt — pass the text to embed, e.g. --prompt \"cup. remote.\""))
	}

	ids, err := m.tok.EncodeBatch(phrases)
	if err != nil {
		return models.Result{}, err
	}

	name := inputIDsName
	if in := r.InputNames(roleModel); len(in) > 0 {
		found := false
		for _, n := range in {
			if n == inputIDsName {
				found = true
				break
			}
		}
		if !found {
			name = in[0]
		}
	}

	outs, err := r.Run(roleModel, map[string]engine.Tensor{
		name: engine.I64(ids, int64(len(phrases)), int64(m.tok.MaxLen())),
	})
	if err != nil {
		return models.Result{}, fmt.Errorf("siglip-text: inference: %w", err)
	}

	embs, err := decodeTextEmbeddings(outs, len(phrases))
	if err != nil {
		return models.Result{}, err
	}
	return models.Result{Task: models.TaskEmbed, Embeddings: embs}, nil
}

// decodeTextEmbeddings reshapes the [N, D] output into N L2-normalised rows. D is NOT asserted
// to be any particular width: 768 for SigLIP base, 1152 for so400m, and textalign validates it
// against the projection's own d_text (see textalign.go's embedVocab).
func decodeTextEmbeddings(outs []engine.Tensor, n int) ([][]float32, error) {
	if len(outs) == 0 {
		return nil, fmt.Errorf("siglip-text: expected 1 output tensor, got 0")
	}
	t := outs[0]
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("siglip-text: unexpected output shape %v (expected [N,D])", t.Shape)
	}
	if int(t.Shape[0]) != n {
		return nil, fmt.Errorf("siglip-text: output batch %d != %d prompts", t.Shape[0], n)
	}
	dim := int(t.Shape[1])
	if dim <= 0 || len(t.Data) < n*dim {
		return nil, fmt.Errorf("siglip-text: output data length %d < %d*%d", len(t.Data), n, dim)
	}
	embs := make([][]float32, n)
	for i := 0; i < n; i++ {
		embs[i] = util.L2Normalized(t.Data[i*dim : (i+1)*dim])
	}
	return embs, nil
}
