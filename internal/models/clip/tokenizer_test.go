package clip

import (
	"os"
	"path/filepath"
	"testing"
)

// tokenizerDir points at the real CLIP tokenizer assets shipped with the text tower
// (models/clip-text/{vocab.json,merges.txt}). Tests skip when they are absent so a
// checkout without model assets still builds green.
func loadTestTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	dir := filepath.Join("..", "..", "..", "models", "clip-text")
	if _, err := os.Stat(filepath.Join(dir, VocabFile)); err != nil {
		t.Skipf("clip-text tokenizer assets not available: %v", err)
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	return tok
}

// TestEncodeGolden pins EXACT token ids against the Python reference
// (transformers 5.9.0, CLIPTokenizerFast + slow CLIPTokenizer agree on every case):
//
//	from transformers import CLIPTokenizerFast
//	tf = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
//	tf("a photo of a cup")["input_ids"]
func TestEncodeGolden(t *testing.T) {
	tok := loadTestTokenizer(t)

	cases := []struct {
		text string
		want []int64
	}{
		// Plain prompt-template text — the (a) use case: class-name embeddings.
		{"a photo of a cup", []int64{49406, 320, 1125, 539, 320, 1937, 49407}},
		// Two-word class name, both words are whole tokens.
		{"a photo of a medicine bottle", []int64{49406, 320, 1125, 539, 320, 5616, 5392, 49407}},
		// Bare class name.
		{"cup", []int64{49406, 1937, 49407}},
		// Lowercasing + whitespace collapsing + punctuation as its own token.
		{"A Photo Of   A   Remote.", []int64{49406, 320, 1125, 539, 320, 9687, 269, 49407}},
		// Contractions are their own alternation branch in the split regex.
		{"it's a cat's toy", []int64{49406, 585, 568, 320, 2368, 568, 5988, 49407}},
		// Digits split ONE BY ONE ("42" → "4","2"): the regex uses [\p{N}] with no "+".
		{"snack box 42 items!", []int64{49406, 10039, 2063, 275, 273, 6207, 256, 49407}},
		// Non-ASCII exercises the byte-level encoder: "—" is 3 UTF-8 bytes → "âĢĶ",
		// "naïve" splits into "na"+"Ã¯"+"ve</w>".
		{"hello, world — naïve café", []int64{49406, 3306, 267, 1002, 2005, 1097, 35689, 563, 15304, 49407}},
		// Real BPE merging: "aaa" is a single vocab entry.
		{"aaa", []int64{49406, 9583, 49407}},
	}

	for _, c := range cases {
		got := tok.Encode(c.text)
		if len(got) != len(c.want) {
			t.Errorf("Encode(%q) length = %d, want %d\n got: %v\nwant: %v",
				c.text, len(got), len(c.want), got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("Encode(%q)[%d] = %d, want %d\n got: %v\nwant: %v",
					c.text, i, got[i], c.want[i], got, c.want)
				break
			}
		}
	}
}

// TestEncodePadded checks the fixed-77 form the ONNX graph consumes: BOS, content,
// EOS, then EOS padding (CLIP pads with <|endoftext|>, verified against HF's
// padding="max_length").
func TestEncodePadded(t *testing.T) {
	tok := loadTestTokenizer(t)

	ids := tok.EncodePadded("a photo of a cup")
	if len(ids) != ContextLength {
		t.Fatalf("EncodePadded length = %d, want %d", len(ids), ContextLength)
	}
	want := []int64{49406, 320, 1125, 539, 320, 1937, 49407, 49407}
	for i, w := range want {
		if ids[i] != w {
			t.Errorf("ids[%d] = %d, want %d", i, ids[i], w)
		}
	}
	for i := 6; i < ContextLength; i++ {
		if ids[i] != TokenEOS {
			t.Errorf("pad ids[%d] = %d, want EOS %d", i, ids[i], TokenEOS)
		}
	}
}

// TestEncodeTruncation pins HF's truncation=True behaviour: the CONTENT is cut to 75
// tokens and then wrapped, so the sequence is exactly 77 and still ends with EOS.
// Reference: tf([s], padding="max_length", max_length=77, truncation=True) where s is
// "cup bottle spoon fork apple banana orange bread towel remote" repeated 12× (122 ids
// untruncated).
func TestEncodeTruncation(t *testing.T) {
	tok := loadTestTokenizer(t)

	phrase := "cup bottle spoon fork apple banana orange bread towel remote"
	long := phrase
	for i := 1; i < 12; i++ {
		long += " " + phrase
	}

	ids := tok.EncodePadded(long)
	if len(ids) != ContextLength {
		t.Fatalf("length = %d, want %d", len(ids), ContextLength)
	}
	if ids[0] != TokenBOS {
		t.Errorf("ids[0] = %d, want BOS %d", ids[0], TokenBOS)
	}
	if ids[ContextLength-1] != TokenEOS {
		t.Errorf("ids[76] = %d, want EOS %d", ids[ContextLength-1], TokenEOS)
	}
	// First and last content tokens, from the Python reference.
	wantHead := []int64{49406, 1937, 5392, 14024, 13700, 3055}
	for i, w := range wantHead {
		if ids[i] != w {
			t.Errorf("ids[%d] = %d, want %d", i, ids[i], w)
		}
	}
	wantTail := []int64{1937, 5392, 14024, 13700, 3055, 49407}
	for i, w := range wantTail {
		j := ContextLength - len(wantTail) + i
		if ids[j] != w {
			t.Errorf("ids[%d] = %d, want %d", j, ids[j], w)
		}
	}
}

// TestEncodeBatch checks the flat [N,77] layout handed to the ONNX session.
func TestEncodeBatch(t *testing.T) {
	tok := loadTestTokenizer(t)

	texts := []string{"a photo of a cup", "a photo of a remote"}
	flat := tok.EncodeBatch(texts)
	if len(flat) != len(texts)*ContextLength {
		t.Fatalf("EncodeBatch length = %d, want %d", len(flat), len(texts)*ContextLength)
	}
	for i, text := range texts {
		row := flat[i*ContextLength : (i+1)*ContextLength]
		single := tok.EncodePadded(text)
		for j := range single {
			if row[j] != single[j] {
				t.Fatalf("row %d differs from EncodePadded at %d: %d vs %d", i, j, row[j], single[j])
			}
		}
	}
}

// TestSpecialTokensPassThrough: a literal special token in the text must map to its id,
// not be BPE'd into pieces (HuggingFace strips added tokens before pre-tokenizing).
func TestSpecialTokensPassThrough(t *testing.T) {
	tok := loadTestTokenizer(t)

	ids := tok.Encode("<|startoftext|>cup<|endoftext|>")
	want := []int64{TokenBOS, TokenBOS, 1937, TokenEOS, TokenEOS}
	if len(ids) != len(want) {
		t.Fatalf("got %v, want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("got %v, want %v", ids, want)
		}
	}
}

// TestDecodeRoundTrip: Decode is a debugging aid, so only round-trip fidelity on
// lowercase ASCII is guaranteed.
func TestDecodeRoundTrip(t *testing.T) {
	tok := loadTestTokenizer(t)

	for _, s := range []string{"a photo of a cup", "medicine bottle", "snack bag"} {
		if got := tok.Decode(tok.Encode(s)); got != s {
			t.Errorf("Decode(Encode(%q)) = %q", s, got)
		}
	}
}

// TestEmptyText: no pre-tokens at all still yields a valid BOS/EOS sequence rather
// than a panic or an empty tensor row.
func TestEmptyText(t *testing.T) {
	tok := loadTestTokenizer(t)

	ids := tok.Encode("   ")
	if len(ids) != 2 || ids[0] != TokenBOS || ids[1] != TokenEOS {
		t.Errorf("Encode(whitespace) = %v, want [BOS EOS]", ids)
	}
	if len(tok.EncodePadded("")) != ContextLength {
		t.Errorf("EncodePadded(\"\") must still be %d long", ContextLength)
	}
}

// TestBytesToUnicode pins the GPT-2 byte↔unicode table CLIP reuses: printable ASCII
// maps to itself, control/space bytes map into the 256+ range, and the map is bijective.
func TestBytesToUnicode(t *testing.T) {
	enc, dec := bytesToUnicode()
	if len(enc) != 256 || len(dec) != 256 {
		t.Fatalf("table sizes = %d/%d, want 256/256 (mapping must be bijective)", len(enc), len(dec))
	}
	if enc['a'] != 'a' || enc['!'] != '!' || enc['~'] != '~' {
		t.Errorf("printable ASCII must map to itself")
	}
	if enc[' '] != 'Ġ' {
		t.Errorf("enc[space] = %q, want 'Ġ' (U+0120)", enc[' '])
	}
	for b := 0; b < 256; b++ {
		if dec[enc[byte(b)]] != byte(b) {
			t.Fatalf("byte %d does not round-trip", b)
		}
	}
}

// TestConcurrentEncode guards the CLAUDE.md thread-safety rule: the server encodes
// prompts from many goroutines and the Tokenizer keeps no mutable state.
// Run with -race to be meaningful.
func TestConcurrentEncode(t *testing.T) {
	tok := loadTestTokenizer(t)

	want := tok.Encode("a photo of a medicine bottle")
	done := make(chan bool, 8)
	for i := 0; i < 8; i++ {
		go func() {
			for j := 0; j < 50; j++ {
				got := tok.Encode("a photo of a medicine bottle")
				if len(got) != len(want) {
					done <- false
					return
				}
				for k := range got {
					if got[k] != want[k] {
						done <- false
						return
					}
				}
			}
			done <- true
		}()
	}
	for i := 0; i < 8; i++ {
		if !<-done {
			t.Fatal("concurrent Encode produced a different result")
		}
	}
}
