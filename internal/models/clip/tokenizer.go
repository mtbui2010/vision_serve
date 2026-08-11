package clip

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// CLIP text-branch constants (openai/clip-vit-base-patch32, MIT).
const (
	// TokenBOS is <|startoftext|>, prepended to every sequence.
	TokenBOS int64 = 49406
	// TokenEOS is <|endoftext|>. It terminates the sequence AND doubles as the padding
	// token — which is why padding is harmless: the text transformer is causally masked
	// and pooling happens at the FIRST EOS, so pad positions can never influence it.
	TokenEOS int64 = 49407
	// ContextLength is CLIP's fixed sequence length. The exported graph
	// (models/clip-text/model.onnx) takes input_ids of shape [batch, 77].
	ContextLength = 77

	bosToken = "<|startoftext|>"
	eosToken = "<|endoftext|>"
	// endOfWord is the BPE end-of-word suffix ("end_of_word_suffix" in tokenizer.json).
	endOfWord = "</w>"
)

// VocabFile / MergesFile are the tokenizer asset names expected next to model.onnx
// in the model directory (copied verbatim from the HuggingFace repo).
const (
	VocabFile  = "vocab.json"
	MergesFile = "merges.txt"
)

// clipPattern is CLIP's pre-tokenizer regex, copied verbatim from tokenizer.json
// ("pre_tokenizer" → Split, invert=true): each match is one pre-token, everything in
// between (i.e. whitespace) is discarded. Note `[\p{N}]` without `+`: digits are split
// one by one ("42" → "4", "2").
//
// Go's regexp is leftmost-first over alternations (same as the Rust regex crate that
// HuggingFace tokenizers uses), so the "'s|'t|..." contractions win over `[\p{L}]+`.
const clipPattern = `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`

var (
	splitRE = regexp.MustCompile(clipPattern)
	wsRE    = regexp.MustCompile(`\s+`)
)

// Tokenizer is a pure-Go implementation of CLIP's byte-pair-encoding tokenizer
// (no Python at runtime — see CLAUDE.md). It reproduces HuggingFace's
// CLIPTokenizer(Fast) for openai/clip-vit-base-patch32:
//
//	normalize (collapse whitespace + lowercase) → regex pre-tokenize → byte-level
//	encode → BPE merges with the "</w>" end-of-word suffix → vocab lookup →
//	wrap with <|startoftext|> / <|endoftext|> → pad or truncate to 77.
//
// This is a DIFFERENT algorithm from the BERT WordPiece tokenizer used by
// GroundingDINO (internal/models/groundingdino/tokenizer.go).
//
// Known deviation: HuggingFace applies Unicode NFC normalization first. We do not
// (it would pull in golang.org/x/text for a case that never arises with ASCII class
// names). Input that is already NFC — which all ASCII text is — tokenizes identically;
// decomposed input (e.g. "e" + U+0301 instead of "é") may differ.
//
// A Tokenizer is immutable after loading and therefore safe for concurrent use.
type Tokenizer struct {
	vocab   map[string]int   // BPE token string → id
	idToTok []string         // id → token string (for Decode)
	ranks   map[string]int   // merge pair "a b" → rank (lower merges first)
	byteEnc map[byte]rune    // GPT-2 byte→unicode mapping
	byteDec map[rune]byte    // inverse, for Decode
}

// LoadTokenizer reads vocab.json + merges.txt from the model directory (the same
// directory that holds model.onnx) and builds the lookup tables.
func LoadTokenizer(dir string) (*Tokenizer, error) {
	vocabPath := filepath.Join(dir, VocabFile)
	raw, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("clip: failed to read %s: %w", vocabPath, err)
	}
	var vocab map[string]int
	if err := json.Unmarshal(raw, &vocab); err != nil {
		return nil, fmt.Errorf("clip: failed to parse %s: %w", vocabPath, err)
	}
	if len(vocab) == 0 {
		return nil, fmt.Errorf("clip: vocab %s is empty", vocabPath)
	}

	t := &Tokenizer{vocab: vocab, ranks: make(map[string]int)}

	// id → token table for Decode.
	maxID := 0
	for _, id := range vocab {
		if id > maxID {
			maxID = id
		}
	}
	t.idToTok = make([]string, maxID+1)
	for tok, id := range vocab {
		if id >= 0 {
			t.idToTok[id] = tok
		}
	}

	mergesPath := filepath.Join(dir, MergesFile)
	f, err := os.Open(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("clip: failed to open %s: %w", mergesPath, err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	rank := 0
	for sc.Scan() {
		line := sc.Text()
		// Skip the "#version:" header and blank lines; everything else is "left right".
		if line == "" || strings.HasPrefix(line, "#version:") {
			continue
		}
		if _, ok := t.ranks[line]; !ok {
			t.ranks[line] = rank
		}
		rank++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("clip: failed to read %s: %w", mergesPath, err)
	}
	if rank == 0 {
		return nil, fmt.Errorf("clip: merges %s is empty", mergesPath)
	}

	t.byteEnc, t.byteDec = bytesToUnicode()
	return t, nil
}

// Encode turns text into CLIP token ids WITHOUT padding: <|startoftext|> … <|endoftext|>.
// The result is truncated to ContextLength (77) keeping <|endoftext|> last, matching
// HuggingFace's truncation=True.
func (t *Tokenizer) Encode(text string) []int64 {
	pieces := t.tokenize(text)

	// Reserve room for BOS + EOS (HF truncates the CONTENT to 75, then wraps).
	if max := ContextLength - 2; len(pieces) > max {
		pieces = pieces[:max]
	}

	ids := make([]int64, 0, len(pieces)+2)
	ids = append(ids, TokenBOS)
	for _, p := range pieces {
		if id, ok := t.vocab[p]; ok {
			ids = append(ids, int64(id))
		} else {
			// unk_token for CLIP is <|endoftext|> (see tokenizer.json). Byte-level
			// encoding makes this practically unreachable: every single byte char is
			// in the vocab.
			ids = append(ids, TokenEOS)
		}
	}
	return append(ids, TokenEOS)
}

// EncodePadded returns exactly ContextLength (77) ids, padded with <|endoftext|> —
// the shape models/clip-text/model.onnx expects for one sequence.
func (t *Tokenizer) EncodePadded(text string) []int64 {
	ids := t.Encode(text)
	out := make([]int64, ContextLength)
	n := copy(out, ids)
	for i := n; i < ContextLength; i++ {
		out[i] = TokenEOS
	}
	return out
}

// EncodeBatch returns a flat [len(texts) * ContextLength] int64 buffer, ready to be
// wrapped as an ONNX tensor of shape [N, 77]. Row i corresponds to texts[i].
func (t *Tokenizer) EncodeBatch(texts []string) []int64 {
	out := make([]int64, 0, len(texts)*ContextLength)
	for _, s := range texts {
		out = append(out, t.EncodePadded(s)...)
	}
	return out
}

// tokenize produces the BPE token STRINGS (no BOS/EOS, no padding).
func (t *Tokenizer) tokenize(text string) []string {
	var out []string
	for _, pre := range t.pretokenize(text) {
		// Special tokens bypass BPE entirely (HuggingFace strips added tokens before
		// the pre-tokenizer; the regex above keeps them as one piece).
		if pre == bosToken || pre == eosToken {
			out = append(out, pre)
			continue
		}
		// Byte-level: encode the UTF-8 bytes into the GPT-2 printable alphabet, so any
		// byte sequence is representable with in-vocab single characters.
		var b strings.Builder
		for i := 0; i < len(pre); i++ {
			b.WriteRune(t.byteEnc[pre[i]])
		}
		out = append(out, t.bpe(b.String())...)
	}
	return out
}

// pretokenize normalizes (collapse whitespace runs → single space, then lowercase, per
// tokenizer.json's normalizer sequence) and applies CLIP's split regex.
func (t *Tokenizer) pretokenize(text string) []string {
	norm := strings.ToLower(wsRE.ReplaceAllString(text, " "))
	return splitRE.FindAllString(norm, -1)
}

// bpe applies the merge list to one byte-encoded pre-token and returns its subword
// pieces. The last character carries the "</w>" end-of-word suffix, which is what makes
// CLIP's BPE differ from GPT-2's.
func (t *Tokenizer) bpe(token string) []string {
	runes := []rune(token)
	if len(runes) == 0 {
		return nil
	}

	// word = [c0, c1, …, cN + "</w>"]
	word := make([]string, len(runes))
	for i, r := range runes {
		word[i] = string(r)
	}
	word[len(word)-1] += endOfWord

	for len(word) > 1 {
		// Find the adjacent pair with the lowest merge rank.
		bestRank := -1
		bestIdx := -1
		for i := 0; i+1 < len(word); i++ {
			r, ok := t.ranks[word[i]+" "+word[i+1]]
			if !ok {
				continue
			}
			if bestIdx == -1 || r < bestRank {
				bestRank, bestIdx = r, i
			}
		}
		if bestIdx == -1 {
			break // no mergeable pair left
		}

		// Merge EVERY occurrence of that pair, left to right (HF/OpenAI semantics).
		first, second := word[bestIdx], word[bestIdx+1]
		merged := make([]string, 0, len(word))
		for i := 0; i < len(word); {
			if i+1 < len(word) && word[i] == first && word[i+1] == second {
				merged = append(merged, first+second)
				i += 2
				continue
			}
			merged = append(merged, word[i])
			i++
		}
		word = merged
	}
	return word
}

// Decode turns token ids back into text: it strips BOS/EOS, undoes the byte-level
// mapping and turns "</w>" back into a space. Out-of-range ids are skipped.
func (t *Tokenizer) Decode(ids []int64) string {
	var b strings.Builder
	for _, id := range ids {
		if id == TokenBOS || id == TokenEOS {
			continue
		}
		if id < 0 || int(id) >= len(t.idToTok) {
			continue
		}
		b.WriteString(t.idToTok[id])
	}
	// Undo byte-level encoding.
	raw := make([]byte, 0, b.Len())
	for _, r := range b.String() {
		if bb, ok := t.byteDec[r]; ok {
			raw = append(raw, bb)
			continue
		}
		raw = append(raw, string(r)...)
	}
	return strings.TrimSpace(strings.ReplaceAll(string(raw), endOfWord, " "))
}

// bytesToUnicode is GPT-2's reversible byte↔unicode table (CLIP reuses it): it maps all
// 256 byte values onto printable, non-whitespace unicode code points so the BPE alphabet
// is finite and every byte string is encodable.
func bytesToUnicode() (map[byte]rune, map[rune]byte) {
	enc := make(map[byte]rune, 256)
	dec := make(map[rune]byte, 256)

	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}

	n := 0
	for b := 0; b < 256; b++ {
		var r rune
		if printable(b) {
			r = rune(b)
		} else {
			r = rune(256 + n)
			n++
		}
		enc[byte(b)] = r
		dec[r] = byte(b)
	}
	return enc, dec
}
