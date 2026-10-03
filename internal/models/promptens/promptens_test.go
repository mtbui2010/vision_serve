package promptens

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestLoadMissingFileFallsBack(t *testing.T) {
	fb := []string{"a photo of a {}."}
	got, err := Load(filepath.Join(t.TempDir(), "templates.txt"), fb)
	if err != nil || !reflect.DeepEqual(got, fb) {
		t.Fatalf("Load(missing) = %v, %v; want the fallback", got, err)
	}
}

func TestLoadParsesCommentsAndBlankLines(t *testing.T) {
	p := filepath.Join(t.TempDir(), "templates.txt")
	if err := os.WriteFile(p, []byte("# ensemble\n\n  a photo of a {}.  \nitap of a {}.\n# end\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := Load(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"a photo of a {}.", "itap of a {}."}; !reflect.DeepEqual(got, want) {
		t.Errorf("Load = %q, want %q", got, want)
	}
}

// A template without the placeholder embeds the same string for every class: every row of the
// vocabulary would be identical. It must fail at load, not at serve time.
func TestLoadRejectsBadFiles(t *testing.T) {
	dir := t.TempDir()
	for name, body := range map[string]string{
		"noplaceholder.txt": "a photo of a {}.\na photo.\n",
		"empty.txt":         "# only comments\n\n",
	} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(p, []string{"x {}"}); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := Load(dir, nil); err == nil || !strings.Contains(err.Error(), "cannot read") {
		t.Errorf("a directory: %v, want a read error (not the fallback)", err)
	}
}

// Average relies on exactly this order: class 0's templates first.
func TestApplyIsClassMajor(t *testing.T) {
	got := Apply([]string{"a photo of a {}.", "a {} {}"}, []string{"cup", "hat"})
	want := []string{"a photo of a cup.", "a cup cup", "a photo of a hat.", "a hat hat"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Apply = %q, want %q", got, want)
	}
	if got := Apply([]string{"a {}"}, nil); len(got) != 0 {
		t.Errorf("Apply with no classes = %q", got)
	}
}

// Normalise-average-normalise, with the shared normaliser's arithmetic.
func TestAverage(t *testing.T) {
	embs := [][]float32{
		{1, 0}, {0, 1}, // class 0: two orthogonal unit rows → (1,1)/√2
		{0, 1}, {0, 1}, // class 1: identical rows → (0,1)
	}
	got, err := Average(embs, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	inv := float32(1 / math.Sqrt(2))
	if got[0][0] != inv || got[0][1] != inv {
		t.Errorf("class 0 = %v, want [%v %v]", got[0], inv, inv)
	}
	if got[1][0] != 0 || got[1][1] != 1 {
		t.Errorf("class 1 = %v, want [0 1]", got[1])
	}
	if &got[0][0] == &embs[0][0] {
		t.Error("Average aliased its input")
	}
	for name, c := range map[string][3]int{"count": {3, 2, 0}, "zero templates": {2, 0, 0}} {
		if _, err := Average(embs, c[0], c[1]); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := Average([][]float32{{1, 0}, {1}}, 1, 2); err == nil {
		t.Error("rows of different widths accepted")
	}
}

func TestTemplateKey(t *testing.T) {
	a := TemplateKey([]string{"a photo of a {}.", "itap of a {}."})
	if a != TemplateKey([]string{"a photo of a {}.", "itap of a {}."}) {
		t.Error("key is not deterministic")
	}
	if a == TemplateKey([]string{"itap of a {}.", "a photo of a {}."}) {
		t.Error("key ignores template order")
	}
	// The separator keeps ["ab", "c"] and ["a", "bc"] apart.
	if TemplateKey([]string{"ab", "c"}) == TemplateKey([]string{"a", "bc"}) {
		t.Error("templates are not delimited in the key")
	}
}
