package siglip

import (
	"encoding/json"
	"os"
	"testing"
)

// tokenizerPath is the shipped tokenizer.json. The test skips rather than fails when it is
// absent, so a checkout without model weights still builds and tests green.
const tokenizerPath = "../../../models/siglip-text/tokenizer.json"

func load(t *testing.T) *Tokenizer {
	t.Helper()
	raw, err := os.ReadFile(tokenizerPath)
	if err != nil {
		t.Skipf("tokenizer.json not present (%v) — see models/siglip-text/README.md", err)
	}
	tok, err := ParseTokenizer(raw)
	if err != nil {
		t.Fatalf("ParseTokenizer: %v", err)
	}
	return tok
}

// TestAgainstReference is the whole point of this package: token-for-token agreement with HF's
// own SigLIP tokenizer on every string the vocabulary path actually sees — all COCO and ETRI
// class names, every prompt template applied to them, and a set of shapes chosen to break a
// hand-written normalizer (empty, whitespace runs, punctuation-only, mixed case, very long).
//
// Reference ids produced by scratchpad/ovclean/dump_siglip_tokens.py.
func TestAgainstReference(t *testing.T) {
	tok := load(t)
	raw, err := os.ReadFile("testdata/tokens.json")
	if err != nil {
		t.Fatalf("read reference: %v", err)
	}
	var cases []struct {
		Text string  `json:"text"`
		IDs  []int64 `json:"ids"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("parse reference: %v", err)
	}
	if len(cases) < 100 {
		t.Fatalf("only %d reference cases; expected the full sweep", len(cases))
	}

	bad := 0
	for _, c := range cases {
		got, err := tok.Encode(c.Text)
		if err != nil {
			t.Errorf("Encode(%q): %v", c.Text, err)
			bad++
			continue
		}
		if len(got) != len(c.IDs) {
			if bad < 10 {
				t.Errorf("%q: got %d ids %v, want %d %v", c.Text, len(got), got, len(c.IDs), c.IDs)
			}
			bad++
			continue
		}
		for i := range got {
			if got[i] != c.IDs[i] {
				if bad < 10 {
					t.Errorf("%q: ids differ at %d: got %v want %v", c.Text, i, got, c.IDs)
				}
				bad++
				break
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d strings disagree with the reference tokenizer", bad, len(cases))
	}
	t.Logf("%d strings, 0 mismatches", len(cases))
}

func TestNormalize(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"A Photo Of A Cup.", "a photo of a cup"},
		{"a  photo   of a cup", "a photo of a cup"},
		{"  leading and trailing  ", "leading and trailing"},
		{"!!!", ""},
		{"rolled-up cloth", "rolledup cloth"},
		{"cup,hat,towel", "cuphattowel"},
		// the case that separates the two reference tokenizers: the slow SiglipTokenizer (which
		// produced our distillation targets) deletes "/", the fast tokenizer.json keeps it.
		{"a photo of a N/A.", "a photo of a na"},
		{"a<b>c", "abc"},
	} {
		if got := Normalize(c.in); got != c.want {
			t.Errorf("Normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestNonASCIIRejected pins the documented boundary: the Precompiled charsmap is not
// implemented, so non-ASCII must error rather than produce a plausible-but-different sequence.
func TestNonASCIIRejected(t *testing.T) {
	tok := load(t)
	for _, s := range []string{"café", "khăn", "커피", "naïve"} {
		if _, err := tok.Encode(s); err == nil {
			t.Errorf("Encode(%q) succeeded; non-ASCII must be refused", s)
		}
	}
}

// The reference fixture in testdata/tokens.json stores UNPADDED ids, so TestAgainstReference
// cannot see the filler EncodePadded appends — and that is precisely where this tokenizer was
// wrong. It padded with "<pad>" (id 0), which the vocabulary really does contain, while this
// checkpoint's tokenizer_config.json declares pad_token = "</s>" and the reference pads with
// id 1. Nothing failed: every prompt embedded to a plausible vector that was not the reference's.
// Measured through the hybrid router's SigLIP rescorer, it cost 4.3 mAP on the held-out-names
// protocol. Assert the VALUE, not just the length.
func TestEncodePaddedPadsWithEOSNotPadPiece(t *testing.T) {
	tok := load(t)
	ids, err := tok.EncodePadded("a photo of a cup.")
	if err != nil {
		t.Fatal(err)
	}
	last := ids[len(ids)-1]
	if last != int64(tok.eosID) {
		t.Errorf("padding filler is %d, want the eos id %d (pad_token = \"</s>\")", last, tok.eosID)
	}
	if pad, ok := tok.vocab["<pad>"]; ok && last == int64(pad) {
		t.Errorf("padding filler is <pad> (%d); this checkpoint pads with </s> (%d)", pad, tok.eosID)
	}
	// The whole tail must be the same filler — a single terminator followed by zeros would
	// still pass a check that only looked at the last element.
	body, err := tok.Encode("a photo of a cup.")
	if err != nil {
		t.Fatal(err)
	}
	for i := len(body); i < len(ids); i++ {
		if ids[i] != int64(tok.eosID) {
			t.Fatalf("id %d of the padded tail is %d, want %d", i, ids[i], tok.eosID)
		}
	}
}

func TestEncodePaddedShape(t *testing.T) {
	tok := load(t)
	ids, err := tok.EncodePadded("a photo of a cup")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != tok.MaxLen() {
		t.Fatalf("EncodePadded returned %d ids, want %d", len(ids), tok.MaxLen())
	}
	long, err := tok.EncodePadded(repeat("word ", 200))
	if err != nil {
		t.Fatal(err)
	}
	if len(long) != tok.MaxLen() {
		t.Fatalf("long input produced %d ids, want %d", len(long), tok.MaxLen())
	}
	batch, err := tok.EncodeBatch([]string{"cup", "hat", "towel"})
	if err != nil {
		t.Fatal(err)
	}
	if len(batch) != 3*tok.MaxLen() {
		t.Fatalf("EncodeBatch returned %d ids, want %d", len(batch), 3*tok.MaxLen())
	}
}

func repeat(s string, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		out += s
	}
	return out
}
