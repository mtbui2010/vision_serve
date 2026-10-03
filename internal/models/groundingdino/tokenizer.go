package groundingdino

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm" // BSD-3-Clause (the Go project)
)

// Special token ids (bert-base-uncased vocab.txt, 0-based line index == token id).
const (
	idUNK = 100
	idCLS = 101
	idSEP = 102
	// idPeriod is "." in bert-base-uncased — the separator GroundingDINO prompts use
	// between class phrases ("cup. water bottle."). postprocess splits on it.
	idPeriod = 1012
	// idComma is "," — NOT one of the four tokens the export treats as special
	// ([CLS], [SEP], ".", "?"), so commas do not add loop iterations to the baked mask
	// subgraph. That lets several classes share one pass while still being labelled
	// separately: "." bounds a PASS, "," bounds a LABEL inside it.
	idComma = 1010
)

// Tokenizer is a minimal BERT bert-base-uncased WordPiece tokenizer implemented in pure
// Go. It loads vocab.txt where line N (0-based) maps to token id N. This is enough to
// drive GroundingDINO's text branch; we deliberately avoid pulling Python at runtime
// (see CLAUDE.md).
type Tokenizer struct {
	vocab   map[string]int // token string -> id
	idToTok []string       // id -> token string
}

// LoadTokenizer reads vocab.txt and builds the lookup tables. Returns an error if the
// file is missing or empty.
func LoadTokenizer(vocabPath string) (*Tokenizer, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("groundingdino: failed to open vocab %s: %w", vocabPath, err)
	}
	defer f.Close()

	t := &Tokenizer{vocab: make(map[string]int)}
	sc := bufio.NewScanner(f)
	// Some vocab tokens are long-ish; the default buffer is fine, but raise it to be safe.
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	id := 0
	for sc.Scan() {
		tok := sc.Text()
		// vocab.txt entries are exact tokens; do NOT trim spaces (would corrupt rare
		// tokens), but BERT vocab tokens never contain leading/trailing whitespace.
		t.vocab[tok] = id
		t.idToTok = append(t.idToTok, tok)
		id++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("groundingdino: failed to read vocab %s: %w", vocabPath, err)
	}
	if len(t.idToTok) == 0 {
		return nil, fmt.Errorf("groundingdino: vocab %s is empty", vocabPath)
	}
	return t, nil
}

// Encoded is the tokenizer output: token ids wrapped with [CLS]/[SEP] plus the masks
// GroundingDINO needs.
type Encoded struct {
	InputIDs      []int64
	AttentionMask []int64 // all ones
	TokenTypeIDs  []int64 // all zeros
}

// Encode normalizes the text the way BERT's uncased BasicTokenizer does (see normalizeBERT),
// splits on whitespace, peels punctuation off as standalone tokens, WordPieces each word,
// then wraps with [CLS]/[SEP].
//
// Verified: "cat. remote." -> tokens [CLS] cat . remote . [SEP],
// ids [101 4937 1012 6556 1012 102]; and against the HF reference tokenizer on accents, CJK,
// Hangul, control/zero-width characters and punctuation (TestEncodeMatchesHFReference).
func (t *Tokenizer) Encode(text string) Encoded {
	tokens := t.tokenize(text)
	ids := make([]int64, 0, len(tokens)+2)
	ids = append(ids, idCLS)
	for _, tok := range tokens {
		if id, ok := t.vocab[tok]; ok {
			ids = append(ids, int64(id))
		} else {
			ids = append(ids, idUNK)
		}
	}
	ids = append(ids, idSEP)

	attn := make([]int64, len(ids))
	ttids := make([]int64, len(ids))
	for i := range attn {
		attn[i] = 1 // token_type_ids stay zero
	}
	return Encoded{InputIDs: ids, AttentionMask: attn, TokenTypeIDs: ttids}
}

// tokenize produces the WordPiece token strings (without [CLS]/[SEP]).
func (t *Tokenizer) tokenize(text string) []string {
	var out []string
	for _, word := range basicSplit(normalizeBERT(text)) {
		out = append(out, t.wordpiece(word)...)
	}
	return out
}

// normalizeBERT is the BertNormalizer that bert-base-uncased (and therefore GroundingDINO's
// HF processor) runs before splitting, in the same order as tokenizers' BertNormalizer:
//
//  1. clean text: drop NUL, U+FFFD and control/format characters (category C*, except
//     \t \n \r), and turn every whitespace character into a plain space;
//  2. surround each CJK ideograph with spaces, so a CJK run becomes one word per character;
//  3. strip accents: NFD, then drop non-spacing marks (Mn) — "café" -> "cafe";
//  4. lowercase.
//
// Without 3, every accented word fell out of the vocabulary as a single [UNK]; without 2, a
// CJK run was WordPieced as one word. Printable ASCII passes through steps 1-3 unchanged, so
// the ids of ordinary English prompts are exactly what they were before this function existed.
func normalizeBERT(text string) string {
	ascii := true
	for i := 0; i < len(text); i++ {
		if c := text[i]; c >= utf8.RuneSelf || (c < 0x20 && c != '\t' && c != '\n' && c != '\r') || c == 0x7f {
			ascii = false
			break
		}
	}
	if ascii { // fast path: nothing to clean, no CJK, NFD is the identity
		return strings.ToLower(text)
	}

	var b strings.Builder
	b.Grow(len(text) + 8)
	for _, r := range text {
		switch {
		case r == 0 || r == utf8.RuneError || isBERTControl(r):
			continue
		case isBERTWhitespace(r):
			b.WriteByte(' ')
		case isCJK(r):
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
		default:
			b.WriteRune(r)
		}
	}
	stripped := strings.Map(func(r rune) rune {
		if unicode.Is(unicode.Mn, r) {
			return -1
		}
		return r
	}, norm.NFD.String(b.String()))
	return strings.ToLower(stripped)
}

// isBERTWhitespace mirrors tokenizers' is_whitespace: \t \n \r or a Unicode White_Space
// character (U+00A0, U+2028, U+3000, ...). Control characters are removed before this test,
// so \v, \f and U+0085 never reach it.
func isBERTWhitespace(r rune) bool {
	return r == '\t' || r == '\n' || r == '\r' || unicode.IsSpace(r)
}

// isBERTControl mirrors tokenizers' is_control: any "Other" category character (Cc, Cf, Co,
// Cs, or unassigned) except \t \n \r, which count as whitespace.
func isBERTControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	if unicode.In(r, unicode.Cc, unicode.Cf, unicode.Co, unicode.Cs) {
		return true
	}
	// Unassigned code points (Cn) have no table of their own.
	return !unicode.In(r, unicode.L, unicode.M, unicode.N, unicode.P, unicode.S, unicode.Z)
}

// isCJK is BERT's _is_chinese_char: the CJK Unified Ideographs blocks and their extensions
// and compatibility ideographs. Hiragana, Katakana and Hangul are NOT included (BERT leaves
// them to WordPiece; Hangul syllables are decomposed into Jamo by the NFD step instead).
func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}

// basicSplit splits on whitespace and then peels off punctuation as standalone tokens (so
// "cat." -> ["cat", "."]). Its input is already normalizeBERT'd.
func basicSplit(text string) []string {
	var words []string
	for _, ws := range strings.Fields(text) {
		var cur strings.Builder
		flush := func() {
			if cur.Len() > 0 {
				words = append(words, cur.String())
				cur.Reset()
			}
		}
		for _, r := range ws {
			if isPunct(r) {
				flush()
				words = append(words, string(r))
			} else {
				cur.WriteRune(r)
			}
		}
		flush()
	}
	return words
}

// isPunct reports whether r is treated as a standalone punctuation token. This covers all
// ASCII punctuation (including '.', ',', '?', '!') plus any Unicode punctuation, matching
// BERT's whitespace/punctuation splitting.
func isPunct(r rune) bool {
	if r >= 33 && r <= 47 || r >= 58 && r <= 64 || r >= 91 && r <= 96 || r >= 123 && r <= 126 {
		return true
	}
	return unicode.IsPunct(r)
}

// maxWordChars is WordPiece's max_input_chars_per_word: a longer word is [UNK] outright.
const maxWordChars = 100

// wordpiece greedily matches the longest vocab prefix from the front; subwords after the
// first get a "##" prefix. If no prefix matches at some position, or the word is longer than
// maxWordChars characters, the whole word becomes [UNK].
func (t *Tokenizer) wordpiece(word string) []string {
	runes := []rune(word)
	if len(runes) > maxWordChars {
		return []string{"[UNK]"}
	}
	var pieces []string
	start := 0
	for start < len(runes) {
		end := len(runes)
		var cur string
		found := false
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if _, ok := t.vocab[sub]; ok {
				cur = sub
				found = true
				break
			}
			end--
		}
		if !found {
			// No subword from this position matches: the WHOLE word is [UNK].
			return []string{"[UNK]"}
		}
		pieces = append(pieces, cur)
		start = end
	}
	if len(pieces) == 0 {
		return []string{"[UNK]"}
	}
	return pieces
}

// Decode turns a slice of token ids back into a phrase, merging WordPiece "##" subwords
// and joining the rest with single spaces. Out-of-range ids are skipped.
func (t *Tokenizer) Decode(ids []int64) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.idToTok) {
			continue
		}
		tok := t.idToTok[id]
		if strings.HasPrefix(tok, "##") {
			sb.WriteString(tok[2:]) // merge subword onto the previous token
			continue
		}
		if sb.Len() > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(tok)
	}
	return strings.TrimSpace(sb.String())
}
