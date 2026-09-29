package siglip

import (
	"fmt"

	"visionserve/internal/engine"
)

// EmbedTexts runs the SigLIP text tower over every string in ONE batched call and returns one
// L2-normalised row each, in the order given.
//
// It exists so a caller outside this package can embed a vocabulary without going through
// textModel.Infer, whose entrypoint splits its prompt on "." — that would strip the trailing
// period of a prompt template like "a photo of a {}." and embed a different string than the one
// every measurement used. (internal/models/textalign avoids the same trap for the CLIP tower.)
//
// Tokenization is the tokenizer's own padded contract (max_length 64), because SigLIP is trained
// with padding="max_length" and dynamic padding changes the embedding.
func EmbedTexts(texts []string, tok *Tokenizer,
	run func(map[string]engine.Tensor) ([]engine.Tensor, error), inputNames []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("siglip-text: nothing to embed")
	}
	if tok == nil {
		return nil, fmt.Errorf("siglip-text: no tokenizer")
	}
	ids, err := tok.EncodeBatch(texts)
	if err != nil {
		return nil, err
	}
	outs, err := run(map[string]engine.Tensor{
		pickInput(inputNames, inputIDsName): engine.I64(ids, int64(len(texts)), int64(tok.MaxLen())),
	})
	if err != nil {
		return nil, fmt.Errorf("siglip-text: inference: %w", err)
	}
	return decodeTextEmbeddings(outs, len(texts))
}

// pickInput returns want when the session declares it, otherwise the session's first input —
// the same accommodation for differently-named exports that image.go and text.go already make.
func pickInput(names []string, want string) string {
	for _, n := range names {
		if n == want {
			return want
		}
	}
	if len(names) > 0 {
		return names[0]
	}
	return want
}
