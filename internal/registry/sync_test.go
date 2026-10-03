package registry

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The license allowlist and the model-name rule exist twice: here (the registry, the last word at
// load and at `pull <folder>`) and in the Python converter (visionserve/convert/common.py, which
// refuses before a multi-GB export instead of after it). These tests fail when the two drift:
//
//   - the Python source is parsed and its allowlist / name pattern / length limit must equal
//     the Go ones literally;
//   - both sides run the shared corpus testdata/rules_sync.json through their own checks and must
//     give its answers (the same pattern text can still behave differently: Python's `$` also
//     matches before a trailing newline, Go's does not).
//
// clients/python/tests/test_go_python_sync.py is the mirror image on the Python side.

const pythonCommon = "../../clients/python/visionserve/convert/common.py"

type rulesCorpus struct {
	Licenses   map[string]*string `json:"licenses"`
	Names      map[string]bool    `json:"names"`
	NameMaxLen int                `json:"name_max_len"`
}

func loadRulesCorpus(t *testing.T) rulesCorpus {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "rules_sync.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c rulesCorpus
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("rules_sync.json: %v", err)
	}
	if len(c.Licenses) == 0 || len(c.Names) == 0 || c.NameMaxLen == 0 {
		t.Fatal("rules_sync.json is missing licenses, names or name_max_len")
	}
	return c
}

func readPythonCommon(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile(pythonCommon)
	if err != nil {
		t.Fatalf("the Python converter's rules (%s) must be readable to check they match the registry's: %v",
			pythonCommon, err)
	}
	return string(raw)
}

// pyStr matches one Python string literal ('...' or "...", optional r prefix), capturing its body.
const pyStr = `r?(?:"([^"\\\n]*)"|'([^'\\\n]*)')`

func TestPythonLicenseAllowlistMatchesGo(t *testing.T) {
	src := readPythonCommon(t)
	block := regexp.MustCompile(`(?s)\nLICENSE_ALLOWLIST\s*=\s*\{(.*?)\n\}`).FindStringSubmatch(src)
	if block == nil {
		t.Fatalf("could not find `LICENSE_ALLOWLIST = {...}` in %s (renamed? update this test)", pythonCommon)
	}
	py := map[string]string{}
	pair := regexp.MustCompile(pyStr + `\s*:\s*` + pyStr)
	for _, m := range pair.FindAllStringSubmatch(block[1], -1) {
		py[m[1]+m[2]] = m[3] + m[4]
	}
	// Every non-blank, non-comment line of the block must have been a pair: an entry written in a
	// form the pattern misses would otherwise drop out of the comparison silently.
	lines := 0
	for _, l := range strings.Split(block[1], "\n") {
		if l = strings.TrimSpace(l); l != "" && !strings.HasPrefix(l, "#") {
			lines++
		}
	}
	if lines != len(py) {
		t.Fatalf("parsed %d entries from %d lines of LICENSE_ALLOWLIST — one entry per line, quoted strings:\n%s",
			len(py), lines, block[1])
	}
	if len(py) != len(licenseAllowlist) {
		t.Errorf("Python allows %d licenses %v, Go %d %v", len(py), py, len(licenseAllowlist), licenseAllowlist)
	}
	for k, v := range licenseAllowlist {
		if py[k] != v {
			t.Errorf("license %q: Go canonicalizes to %q, Python to %q", k, v, py[k])
		}
	}
	for k := range py {
		if _, ok := licenseAllowlist[k]; !ok {
			t.Errorf("Python allows license %q, which the Go registry refuses", k)
		}
	}
}

func TestPythonNameRuleMatchesGo(t *testing.T) {
	src := readPythonCommon(t)
	m := regexp.MustCompile(`\n_NAME_RE\s*=\s*re\.compile\(\s*` + pyStr + `\s*\)`).FindStringSubmatch(src)
	if m == nil {
		t.Fatalf("could not find `_NAME_RE = re.compile(r\"...\")` in %s (renamed? update this test)", pythonCommon)
	}
	if py := m[1] + m[2]; py != validModelName.String() {
		t.Errorf("model-name pattern: Python %q, Go %q", py, validModelName.String())
	}
	n := regexp.MustCompile(`\nNAME_MAX_LEN\s*=\s*(\d+)`).FindStringSubmatch(src)
	if n == nil {
		t.Fatalf("could not find `NAME_MAX_LEN = <n>` in %s", pythonCommon)
	}
	pyMax, _ := strconv.Atoi(n[1])
	if c := loadRulesCorpus(t); pyMax != c.NameMaxLen {
		t.Errorf("Python NAME_MAX_LEN = %d, corpus name_max_len = %d", pyMax, c.NameMaxLen)
	}
}

// nameAccepted runs a name through the registry's real validation (every other field valid).
func nameAccepted(name string) (bool, error) {
	m := &Manifest{Name: name, Task: "detection", License: "MIT", ModelFile: "m.onnx"}
	m.Input.Width, m.Input.Height = 1, 1
	err := m.validate()
	return err == nil, err
}

func TestRulesCorpusGo(t *testing.T) {
	c := loadRulesCorpus(t)
	for declared, want := range c.Licenses {
		got, ok := canonicalLicense(declared)
		switch {
		case want == nil && ok:
			t.Errorf("license %q: Go accepts it as %q, the corpus says refused", declared, got)
		case want != nil && (!ok || got != *want):
			t.Errorf("license %q: Go gives (%q, %v), the corpus says %q", declared, got, ok, *want)
		}
	}
	for name, want := range c.Names {
		if got, err := nameAccepted(name); got != want {
			t.Errorf("name %q: Go accepted=%v (%v), the corpus says %v", name, got, err, want)
		}
	}
	for n, want := range map[int]bool{c.NameMaxLen: true, c.NameMaxLen + 1: false} {
		if got, err := nameAccepted(strings.Repeat("a", n)); got != want {
			t.Errorf("a %d-character name: Go accepted=%v (%v), want %v", n, got, err, want)
		}
	}
}
