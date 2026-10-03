package clip

import (
	"fmt"
	"image"
	"math"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
)

// Compile-time checks of the interfaces lifecycle type-asserts at load: a signature drift
// fails the build instead of silently changing how the model is run.
var _ models.PipelineModel = (*textModel)(nil)

func init() {
	models.Register("clip-text", NewText)
}

// roleModel is the single ONNX session the text tower needs (manifest `files: model:`).
const roleModel = "model"

// inputIDsName is the graph input name of models/clip-text/model.onnx. If the manifest
// ever points at a differently-named export, Infer falls back to the session's first
// declared input.
const inputIDsName = "input_ids"

// textModel is the CLIP TEXT tower (ViT-B/32 text branch, MIT).
//
// It is a PipelineModel rather than a plain Model because it is PROMPT-driven: its input
// is a string, not an image (CLAUDE.md: "Prompted / multi-session models implement the
// PipelineModel interface instead"). It owns no session — lifecycle.Manager loads and
// keeps role "model" alive; Infer only calls it through the Runner.
//
// The image argument to Infer is IGNORED (there is no image branch in this graph).
type textModel struct {
	cfg models.Config
	tok *Tokenizer
}

// NewText builds the text tower. The tokenizer assets (vocab.json + merges.txt) are
// loaded ONCE here, from the model directory, and the resulting Tokenizer is immutable —
// so concurrent Infer calls need no locking.
func NewText(cfg models.Config) (models.Base, error) {
	if cfg.Dir == "" {
		return nil, fmt.Errorf("clip-text: model directory unknown, cannot load %s/%s", VocabFile, MergesFile)
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

// Tokenizer exposes the loaded tokenizer (handy for callers that want ids without
// running inference, e.g. tests).
func (m *textModel) Tokenizer() *Tokenizer { return m.tok }

// Infer embeds prompt.Text and returns one L2-normalised 512-d vector per phrase.
//
// The prompt is split on "." into phrases, the same convention GroundingDINO uses
// ("cup. remote. water bottle." → three embeddings). Result.Embeddings preserves that
// order, so a caller building a text-aligned class matrix knows which row is which.
// The img argument is ignored — this is the text branch.
func (m *textModel) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	phrases := SplitPhrases(prompt.Text)
	if len(phrases) == 0 {
		return models.Result{}, models.BadPrompt(fmt.Errorf(
			"clip-text: empty prompt — pass the text to embed, e.g. --prompt \"cup. remote.\""))
	}

	ids := m.tok.EncodeBatch(phrases)
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
		name: engine.I64(ids, int64(len(phrases)), int64(ContextLength)),
	})
	if err != nil {
		return models.Result{}, fmt.Errorf("clip-text: inference: %w", err)
	}

	embs, err := decodeTextEmbeddings(outs, len(phrases))
	if err != nil {
		return models.Result{}, err
	}
	return models.Result{Task: models.TaskEmbed, Embeddings: embs}, nil
}

// decodeTextEmbeddings reshapes the [N, D] output into N L2-normalised rows, so cosine
// similarity against a CLIP IMAGE embedding (which internal/models/clip's image
// postprocess also L2-normalises) is a plain dot product.
func decodeTextEmbeddings(outs []engine.Tensor, n int) ([][]float32, error) {
	if len(outs) == 0 {
		return nil, fmt.Errorf("clip-text: expected 1 output tensor, got 0")
	}
	t := outs[0]
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("clip-text: unexpected output shape %v (expected [N,D])", t.Shape)
	}
	if int(t.Shape[0]) != n {
		return nil, fmt.Errorf("clip-text: output batch %d != %d prompts", t.Shape[0], n)
	}
	dim := int(t.Shape[1])
	if dim <= 0 || len(t.Data) < n*dim {
		return nil, fmt.Errorf("clip-text: output data length %d < %d*%d", len(t.Data), n, dim)
	}

	embs := make([][]float32, n)
	for i := 0; i < n; i++ {
		embs[i] = l2Normalize(t.Data[i*dim : (i+1)*dim])
	}
	return embs, nil
}

// l2Normalize returns a normalised COPY of v (a zero vector is returned unchanged).
func l2Normalize(v []float32) []float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	norm := math.Sqrt(sumSq)
	if norm == 0 {
		copy(out, v)
		return out
	}
	inv := float32(1.0 / norm)
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// SplitPhrases splits a prompt into its "."-separated phrases, trimmed, with empty
// pieces dropped ("cup.. remote." → ["cup", "remote"]). A prompt without any "." is a
// single phrase. Exported so callers can reproduce the row order of Result.Embeddings.
func SplitPhrases(text string) []string {
	var out []string
	for _, p := range strings.Split(text, ".") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
