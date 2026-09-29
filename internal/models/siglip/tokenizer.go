// Package siglip implements the SigLIP text tower (Apache-2.0) as a VisionServe model, so a
// head B distilled from SigLIP can be served without pulling Python into the runtime.
//
// # Why a second text tower at all
//
// Head B is a student of whichever teacher produced its distillation targets, and it cannot beat
// that teacher on words it has never seen. Measured on names held out of every trained component,
// SigLIP-crop scores 85.3% where CLIP-crop scores 65.3%, and a head distilled from SigLIP on
// in-domain crops reaches 69.3% where the CLIP-distilled one reaches 46.7%. Raising the teacher is
// the only lever that moves that ceiling — see paper/FINDINGS-2026-08.md §9–§10.
//
// The projection side already supports it: `d_text` is a field of proj.bin, not a constant, and
// textalign's Fold/ProjNorm/gated path are exercised at 512, 768 and 1152 in
// internal/models/textalign/dim_test.go. The only missing piece was this tokenizer, because
// SigLIP uses SentencePiece Unigram where CLIP uses byte-level BPE.
//
// # The tokenizer, and exactly how far it goes
//
// google/siglip-base-patch16-224's tokenizer.json declares:
//
//	model         Unigram, 32000 pieces, unk_id 2, byte_fallback true
//	normalizer    Lowercase → strip a fixed punctuation set → collapse whitespace → trim
//	              → Precompiled (SentencePiece's compiled NFKC-ish charsmap)
//	pre_tokenizer Metaspace: prepend "▁", spaces become "▁"
//	post          append "</s>"
//
// Everything above is implemented here EXCEPT the Precompiled charsmap, which is a compiled trie
// shipped as a base64 blob and is a substantial piece of work on its own. On ASCII input that map
// is the identity, so this tokenizer is EXACT for ASCII and only for ASCII — which covers class
// names and prompt templates, the entire vocabulary path. Non-ASCII input is REJECTED with an
// error rather than silently tokenized differently from the reference; a wrong-but-plausible
// token sequence would surface as a quietly worse vocabulary, which is the failure mode this
// project has been bitten by three times.
package siglip

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

const (
	// metaspace is SentencePiece's word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
	metaspace = "▁"
	// eosPiece terminates every sequence (TemplateProcessing "single": A </s>) AND fills the
	// padding, because this checkpoint's tokenizer_config.json declares pad_token = "</s>".
	//
	// The vocabulary DOES contain a separate "<pad>" piece at id 0, and reading the id off that
	// piece is the obvious thing to do — it is also wrong, and wrong silently: every prompt
	// tokenizes to the correct ids followed by 64-n zeros instead of 64-n ones, the text tower
	// happily embeds it, and the result is a plausible embedding that is not the one the
	// reference produces. Measured through the hybrid router's SigLIP rescorer on the
	// held-out-names protocol, that cost 4.3 mAP (57.85 against Python's 62.16), flipped the
	// argmax on 9.9 % of crops and pushed 29 of 849 below the rejection floor.
	eosPiece = "</s>"
	// maxLen is SigLIP's text context. The HF processor pads to exactly this.
	maxLen = 64
	// unkPenalty is the score charged for an unknown character when byte fallback cannot
	// cover it; large enough that any real piece wins, finite so Viterbi stays defined.
	unkPenalty = -1e4
)

// punctuation is the set the normalizer deletes: Python's `string.punctuation`, because that is
// what transformers' SiglipTokenizer.remove_punctuation uses.
//
// THE TWO REFERENCE TOKENIZERS DISAGREE, and this constant is the difference. tokenizer.json's
// normalizer carries an explicit character class that omits "/", "<" and ">"; the slow
// SiglipTokenizer strips all of string.punctuation. On "a photo of a N/A." the fast path keeps
// the slash and the slow path drops it — 16 of 525 reference strings, all of them containing a
// slash.
//
// We follow the SLOW one deliberately: `AutoProcessor.from_pretrained(...).tokenizer` returns it
// when sentencepiece is installed, so it is the tokenizer that produced the SigLIP embeddings
// head B was distilled against. Matching the other one would put the text tower in a different
// normalization from the targets, which is exactly the kind of silent mismatch this project has
// already paid for three times.
const punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

// Tokenizer is a SentencePiece Unigram tokenizer: Viterbi over a piece vocabulary scored by
// log-probability, with byte fallback for characters no piece covers.
type Tokenizer struct {
	vocab  map[string]int32   // piece -> id
	score  map[string]float64 // piece -> log-prob
	pieces []string           // id -> piece, for Decode
	unkID  int32
	eosID  int32
	padID  int32
	byteID [256]int32 // byte fallback: 0xNN -> id of "<0xNN>", -1 when absent
	maxLen int        // longest piece in bytes, bounds the Viterbi inner loop
}

// tokenizerJSON is the subset of HF's tokenizer.json this needs. Fields that are asserted
// rather than used are still parsed, so an unexpected file fails loudly at load.
type tokenizerJSON struct {
	Model struct {
		Type         string          `json:"type"`
		UnkID        int32           `json:"unk_id"`
		ByteFallback bool            `json:"byte_fallback"`
		Vocab        [][]json.Number `json:"-"`
		RawVocab     json.RawMessage `json:"vocab"`
	} `json:"model"`
	PreTokenizer struct {
		Type        string `json:"type"`
		Replacement string `json:"replacement"`
	} `json:"pre_tokenizer"`
}

// LoadTokenizer reads tokenizer.json from dir (the directory holding the text tower's weights,
// resolved the same way internal/models/clip does).
func LoadTokenizer(dir string) (*Tokenizer, error) {
	path := filepath.Join(dir, "tokenizer.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("siglip: read %s: %w", path, err)
	}
	return ParseTokenizer(raw)
}

// ParseTokenizer builds a Tokenizer from tokenizer.json bytes (split out so tests need no files).
func ParseTokenizer(raw []byte) (*Tokenizer, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("siglip: parse tokenizer.json: %w", err)
	}
	if tj.Model.Type != "Unigram" {
		return nil, fmt.Errorf("siglip: tokenizer model is %q, this implements Unigram only",
			tj.Model.Type)
	}
	if tj.PreTokenizer.Type != "" && tj.PreTokenizer.Type != "Metaspace" {
		return nil, fmt.Errorf("siglip: pre_tokenizer is %q, this implements Metaspace only",
			tj.PreTokenizer.Type)
	}
	if r := tj.PreTokenizer.Replacement; r != "" && r != metaspace {
		return nil, fmt.Errorf("siglip: metaspace replacement is %q, expected %q", r, metaspace)
	}

	// vocab is [[piece, score], ...] with mixed types, so decode it as raw pairs.
	var pairs []([]interface{})
	if err := json.Unmarshal(tj.Model.RawVocab, &pairs); err != nil {
		return nil, fmt.Errorf("siglip: parse vocab: %w", err)
	}
	if len(pairs) == 0 {
		return nil, fmt.Errorf("siglip: tokenizer.json carries an empty vocabulary")
	}

	t := &Tokenizer{
		vocab:  make(map[string]int32, len(pairs)),
		score:  make(map[string]float64, len(pairs)),
		pieces: make([]string, len(pairs)),
		unkID:  tj.Model.UnkID,
		eosID:  -1,
		padID:  -1,
	}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	for i, p := range pairs {
		if len(p) != 2 {
			return nil, fmt.Errorf("siglip: vocab entry %d has %d fields, want 2", i, len(p))
		}
		piece, ok := p[0].(string)
		if !ok {
			return nil, fmt.Errorf("siglip: vocab entry %d has a non-string piece", i)
		}
		sc, ok := p[1].(float64)
		if !ok {
			return nil, fmt.Errorf("siglip: vocab entry %d (%q) has a non-numeric score", i, piece)
		}
		id := int32(i)
		t.vocab[piece] = id
		t.score[piece] = sc
		t.pieces[i] = piece
		if n := len(piece); n > t.maxLen {
			t.maxLen = n
		}
		if piece == eosPiece {
			t.eosID = id
		}
		// byte-fallback pieces are spelled "<0xNN>"
		if len(piece) == 6 && strings.HasPrefix(piece, "<0x") && strings.HasSuffix(piece, ">") {
			var b int
			if _, err := fmt.Sscanf(piece, "<0x%02X>", &b); err == nil && b >= 0 && b < 256 {
				t.byteID[b] = id
			}
		}
	}
	if t.eosID < 0 {
		return nil, fmt.Errorf("siglip: vocabulary has no %q piece", eosPiece)
	}
	t.padID = t.eosID
	if int(t.unkID) < 0 || int(t.unkID) >= len(t.pieces) {
		return nil, fmt.Errorf("siglip: unk_id %d out of range", t.unkID)
	}
	return t, nil
}

// Normalize applies the tokenizer.json normalizer sequence: lowercase, delete the punctuation
// set, collapse runs of whitespace to one space, trim.
//
// The Precompiled charsmap that follows in the reference is NOT applied — see the package
// comment. It is the identity on ASCII, which is why Encode rejects anything else.
func Normalize(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r < 128 && strings.ContainsRune(punctuation, r) {
			continue
		}
		b.WriteRune(r)
	}
	fields := strings.FieldsFunc(b.String(), unicode.IsSpace)
	return strings.Join(fields, " ")
}

// Encode turns text into token ids, terminated by </s>.
//
// Returns an error for non-ASCII input rather than guessing: the reference normalizer would put
// such text through a compiled charsmap this package does not implement, and a plausible-looking
// but different token sequence is worse than a refusal.
func (t *Tokenizer) Encode(text string) ([]int64, error) {
	norm := Normalize(text)
	for i := 0; i < len(norm); i++ {
		if norm[i] >= 0x80 {
			return nil, fmt.Errorf("siglip: %q contains non-ASCII; this tokenizer implements "+
				"the SentencePiece normalizer only for ASCII (see package doc)", text)
		}
	}
	// Text that normalises away entirely ("...", "!!!", whitespace) yields NO pieces at all —
	// not a bare metaspace marker. The reference agrees, and getting this wrong showed up as
	// five reference strings emitting one spurious leading token.
	if norm == "" {
		return []int64{int64(t.eosID)}, nil
	}
	// Metaspace: a leading marker, and every space becomes one.
	spaced := metaspace + strings.ReplaceAll(norm, " ", metaspace)
	ids := t.viterbi(spaced)
	return append(ids, int64(t.eosID)), nil
}

// EncodePadded encodes and pads (or truncates) to the model's fixed context length, which is
// what the ONNX graph's static input shape requires.
func (t *Tokenizer) EncodePadded(text string) ([]int64, error) {
	ids, err := t.Encode(text)
	if err != nil {
		return nil, err
	}
	if len(ids) > maxLen {
		ids = ids[:maxLen]
		ids[maxLen-1] = int64(t.eosID) // never drop the terminator
	}
	for len(ids) < maxLen {
		ids = append(ids, int64(t.padID))
	}
	return ids, nil
}

// EncodeBatch encodes each text to maxLen and concatenates them row-major, the layout the text
// tower's [N, maxLen] input expects.
func (t *Tokenizer) EncodeBatch(texts []string) ([]int64, error) {
	out := make([]int64, 0, len(texts)*maxLen)
	for _, s := range texts {
		ids, err := t.EncodePadded(s)
		if err != nil {
			return nil, err
		}
		out = append(out, ids...)
	}
	return out, nil
}

// MaxLen reports the fixed context length EncodePadded produces.
func (t *Tokenizer) MaxLen() int { return maxLen }

// viterbi finds the segmentation maximising the summed piece log-probabilities — the standard
// SentencePiece Unigram decode. best[i] is the score of the best segmentation of s[:i].
func (t *Tokenizer) viterbi(s string) []int64 {
	n := len(s)
	best := make([]float64, n+1)
	prev := make([]int, n+1)
	pieceAt := make([]string, n+1)
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
		prev[i] = -1
	}
	for i := 0; i < n; i++ {
		if math.IsInf(best[i], -1) {
			continue
		}
		lim := i + t.maxLen
		if lim > n {
			lim = n
		}
		for j := i + 1; j <= lim; j++ {
			sub := s[i:j]
			sc, ok := t.score[sub]
			if !ok {
				continue
			}
			if v := best[i] + sc; v > best[j] {
				best[j], prev[j], pieceAt[j] = v, i, sub
			}
		}
		// Fallback for a position no piece covers: consume one byte. Charged heavily so it is
		// only ever chosen when nothing else reaches this position.
		j := i + 1
		if v := best[i] + unkPenalty; v > best[j] {
			best[j], prev[j], pieceAt[j] = v, i, s[i:j]
		}
	}

	var rev []string
	for i := n; i > 0; {
		p := prev[i]
		if p < 0 { // unreachable in practice: the byte fallback above always advances
			p = i - 1
			pieceAt[i] = s[p:i]
		}
		rev = append(rev, pieceAt[i])
		i = p
	}
	ids := make([]int64, 0, len(rev))
	for i := len(rev) - 1; i >= 0; i-- {
		ids = append(ids, t.idsFor(rev[i])...)
	}
	return ids
}

// idsFor maps one segment to ids: its own id when it is a piece, otherwise byte fallback, and
// <unk> only when even that is unavailable.
func (t *Tokenizer) idsFor(seg string) []int64 {
	if id, ok := t.vocab[seg]; ok {
		return []int64{int64(id)}
	}
	out := make([]int64, 0, len(seg))
	for i := 0; i < len(seg); i++ {
		if id := t.byteID[seg[i]]; id >= 0 {
			out = append(out, int64(id))
		} else {
			out = append(out, int64(t.unkID))
		}
	}
	return out
}

// Decode reverses Encode for debugging: pieces joined, metaspace back to spaces, specials
// dropped.
func (t *Tokenizer) Decode(ids []int64) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.pieces) {
			continue
		}
		p := t.pieces[id]
		if p == eosPiece || p == "<pad>" || int32(id) == t.unkID {
			continue
		}
		b.WriteString(p)
	}
	return strings.TrimSpace(strings.ReplaceAll(b.String(), metaspace, " "))
}
