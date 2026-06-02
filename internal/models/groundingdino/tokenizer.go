package groundingdino

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// Special token ids (bert-base-uncased vocab.txt, 0-based line index == token id).
const (
	idUNK = 100
	idCLS = 101
	idSEP = 102
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

// Encode lowercases the text, splits on whitespace, peels ASCII punctuation off as
// standalone tokens, WordPieces each word, then wraps with [CLS]/[SEP].
//
// Verified: "cat. remote." -> tokens [CLS] cat . remote . [SEP],
// ids [101 4937 1012 6556 1012 102].
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
	for _, word := range basicSplit(strings.ToLower(text)) {
		out = append(out, t.wordpiece(word)...)
	}
	return out
}

// basicSplit splits on whitespace and then peels off ASCII punctuation as standalone
// tokens (so "cat." -> ["cat", "."]). Matches BERT BasicTokenizer behavior for our use.
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

// wordpiece greedily matches the longest vocab prefix from the front; subwords after the
// first get a "##" prefix. If no prefix matches at some position, the whole word becomes
// [UNK].
func (t *Tokenizer) wordpiece(word string) []string {
	runes := []rune(word)
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
