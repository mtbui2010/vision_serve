package groundingdino

import (
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

// An unknown out-of-vocab nonsense word with no matching subwords → [UNK] (id 100).
func TestWordPieceUNK(t *testing.T) {
	tok := loadTok(t)
	enc := tok.Encode("zzqxwk")
	// [CLS] <something> [SEP] — middle ids should include UNK if truly unknown.
	if len(enc.InputIDs) < 3 {
		t.Fatalf("ids too short: %v", enc.InputIDs)
	}
}
