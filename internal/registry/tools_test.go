package registry

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// CheckLicense must refuse exactly what a manifest declaring the licence is refused for, with
// the registry's own message, and canonicalise the same way (the shared corpus of sync_test.go).
func TestCheckLicenseMatchesManifestValidation(t *testing.T) {
	c := loadRulesCorpus(t)
	for declared, want := range c.Licenses {
		canon, err := CheckLicense(declared)
		m := &Manifest{Name: "m", Task: "classification", License: declared, ModelFile: "m.onnx"}
		m.Input.Width, m.Input.Height = 1, 1
		verr := m.validate()
		if want == nil {
			if err == nil || verr == nil || err.Error() != verr.Error() {
				t.Errorf("%q: CheckLicense %v, validate %v — both must refuse, with one message", declared, err, verr)
			}
			continue
		}
		if err != nil || canon != *want || verr != nil || m.License != *want {
			t.Errorf("%q: CheckLicense (%q, %v), validate (%q, %v); want %q", declared, canon, err, m.License, verr, *want)
		}
	}
	if got := AllowedLicenses(); strings.Join(got, ",") != "Apache-2.0,BSD-2-Clause,BSD-3-Clause,MIT" {
		t.Errorf("AllowedLicenses = %v", got)
	}
}

func TestValidTaskAndName(t *testing.T) {
	for _, task := range ValidTasks() {
		if !ValidTask(task) {
			t.Errorf("ValidTask(%q) = false", task)
		}
	}
	if ValidTask("yolo") || ValidTask("") {
		t.Error("ValidTask accepts an unknown task")
	}
	for name, ok := range loadRulesCorpus(t).Names {
		if ValidName(name) != ok {
			t.Errorf("ValidName(%q) = %v, want %v", name, !ok, ok)
		}
	}
}

// The AGPL byte markers are the Python converter's _STRONG_BYTES, literally.
func TestAGPLMarkersMatchPython(t *testing.T) {
	src := readPythonCommon(t)
	block := regexp.MustCompile(`(?s)\n_STRONG_BYTES\s*=\s*\((.*?)\)`).FindStringSubmatch(src)
	if block == nil {
		t.Fatalf("could not find `_STRONG_BYTES = (...)` in %s (renamed? update this test)", pythonCommon)
	}
	var py []string
	for _, m := range regexp.MustCompile(`b"([^"]*)"|b'([^']*)'`).FindAllStringSubmatch(block[1], -1) {
		py = append(py, m[1]+m[2])
	}
	var gol []string
	for _, m := range agplMarkers {
		gol = append(gol, string(m))
	}
	if strings.Join(py, "|") != strings.Join(gol, "|") {
		t.Errorf("Python _STRONG_BYTES %q != Go agplMarkers %q", py, gol)
	}
}

func TestScanAGPLMarkers(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	// What an Ultralytics ONNX export carries in its metadata_props (at the END of the file).
	meta := []byte(`license` + "\x12" + `AGPL-3.0 License (https://ultralytics.com/license)`)
	big := make([]byte, 3*agplScanWindow)
	cases := map[string]struct {
		data []byte
		agpl bool
	}{
		"clean small":          {[]byte("onnx bytes, apache-2.0"), false},
		"marker small":         {append([]byte("graph..."), meta...), true},
		"marker in big tail":   {append(append([]byte{}, big...), meta...), true},
		"marker in big head":   {append(append([]byte{}, meta...), big...), true},
		"marker in the middle": {append(append(append([]byte{}, big...), meta...), big...), false}, // weights: not scanned
		"mixed case":           {[]byte("x HTTPS://ULTRALYTICS.COM y"), true},
	}
	for name, c := range cases {
		err := ScanAGPLMarkers(write(strings.ReplaceAll(name, " ", "_"), c.data))
		if c.agpl != errors.Is(err, ErrAGPLModel) {
			t.Errorf("%s: err = %v, want AGPL %v", name, err, c.agpl)
		}
		if !c.agpl && err != nil {
			t.Errorf("%s: unexpected error %v", name, err)
		}
	}
	// Just over the window: a marker straddling the head/tail boundary is found.
	straddle := make([]byte, agplScanWindow+100)
	copy(straddle[agplScanWindow-4:], "agpl-3.0")
	if err := ScanAGPLMarkers(write("straddle", straddle)); !errors.Is(err, ErrAGPLModel) {
		t.Errorf("straddling marker: %v", err)
	}
}

func TestPutAndSHA256For(t *testing.T) {
	dir := t.TempDir()
	m := &Manifest{Name: "x", dir: dir}
	m.SHA256.single = "ab"
	r := New(t.TempDir())
	r.Put(m)
	if e, ok := r.Get("x"); !ok || e.Manifest != m || e.Dir != dir {
		t.Fatalf("Put/Get: %+v %v", e, ok)
	}
	if d, ok := m.SHA256.For(""); !ok || d != "ab" {
		t.Errorf("For: %q %v", d, ok)
	}
	p := filepath.Join(dir, "f")
	if err := os.WriteFile(p, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	if d, err := FileSHA256(p); err != nil || d != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
		t.Errorf("FileSHA256 = %q, %v", d, err)
	}
}
