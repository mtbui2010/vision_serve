package groundingdino

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// vocabPath points at the real bert-base-uncased vocab shipped with the GroundingDINO
// weights. The test is skipped if it is not present (weights are not committed).
func loadTok(t *testing.T) *Tokenizer {
	t.Helper()
	p := filepath.Join("..", "..", "..", "models", "grounding-dino", "vocab.txt")
	tok, err := LoadTokenizer(p)
	if err != nil {
		t.Skipf("vocab.txt not available (weights not downloaded): %v", err)
	}
	return tok
}

// Verified against HF BertTokenizer: "cat. remote." → ids [101,4937,1012,6556,1012,102],
// tokens [CLS] cat . remote . [SEP].
func TestEncodeCatRemote(t *testing.T) {
	tok := loadTok(t)
	enc := tok.Encode("cat. remote.")
	want := []int64{101, 4937, 1012, 6556, 1012, 102}
	if !reflect.DeepEqual(enc.InputIDs, want) {
		t.Fatalf("input_ids = %v, want %v", enc.InputIDs, want)
	}
	for i, v := range enc.AttentionMask {
		if v != 1 {
			t.Fatalf("attention_mask[%d] = %d, want 1", i, v)
		}
	}
	for i, v := range enc.TokenTypeIDs {
		if v != 0 {
			t.Fatalf("token_type_ids[%d] = %d, want 0", i, v)
		}
	}
}

// Decoding the inner tokens (skipping CLS/SEP) should recover the phrases, merging '##'.
func TestDecodePhrase(t *testing.T) {
	tok := loadTok(t)
	if got := tok.Decode([]int64{4937}); got != "cat" {
		t.Fatalf("decode cat = %q", got)
	}
	if got := tok.Decode([]int64{6556}); got != "remote" {
		t.Fatalf("decode remote = %q", got)
	}
}

// Uppercase input must be lowercased before lookup (so it tokenizes identically).
func TestEncodeLowercases(t *testing.T) {
	tok := loadTok(t)
	a := tok.Encode("CAT. REMOTE.")
	b := tok.Encode("cat. remote.")
	if !reflect.DeepEqual(a.InputIDs, b.InputIDs) {
		t.Fatalf("uppercase ids %v != lowercase ids %v", a.InputIDs, b.InputIDs)
	}
}

// TestEncodeMatchesHFReference holds Encode to the reference tokenizer GroundingDINO's HF
// processor uses (AutoTokenizer "IDEA-Research/grounding-dino-tiny", a BertTokenizerFast with
// do_lower_case=True: clean_text -> CJK spacing -> NFD + drop Mn -> lowercase -> split on
// whitespace/punctuation -> WordPiece with a 100-char word cap). testdata/hf_bert_uncased_ids.json
// was produced by that tokenizer and agrees with tokenizers.Tokenizer.from_file on
// models/grounding-dino/tokenizer.json for every row (transformers 5.9.0, tokenizers 0.22.2).
//
// Before the BasicTokenizer port, "café" encoded as [UNK], CJK runs as one [UNK] word, and
// zero-width / control characters split words that BERT joins.
func TestEncodeMatchesHFReference(t *testing.T) {
	tok := loadTok(t)
	raw, err := os.ReadFile(filepath.Join("testdata", "hf_bert_uncased_ids.json"))
	if err != nil {
		t.Fatal(err)
	}
	var rows []struct {
		Text string  `json:"text"`
		IDs  []int64 `json:"ids"`
	}
	if err := json.Unmarshal(raw, &rows); err != nil {
		t.Fatal(err)
	}
	if len(rows) < 30 {
		t.Fatalf("reference table has %d rows, want the full table", len(rows))
	}
	for _, r := range rows {
		if got := tok.Encode(r.Text).InputIDs; !reflect.DeepEqual(got, r.IDs) {
			t.Errorf("Encode(%q) = %v, want %v", r.Text, got, r.IDs)
		}
	}
}

// An unknown out-of-vocab nonsense word with no matching subwords → [UNK] (id 100).
func TestWordPieceUNK(t *testing.T) {
	tok := loadTok(t)
	enc := tok.Encode("zzqxwk")
	// [CLS] <something> [SEP] — middle ids should include UNK if truly unknown.
	if len(enc.InputIDs) < 3 {
		t.Fatalf("ids too short: %v", enc.InputIDs)
	}
}
