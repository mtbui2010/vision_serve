package clireport

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func sample() *Report {
	r := &Report{
		Command: "inspect", Subject: "demo",
		Summary: []Field{
			{Key: "model", Label: "Model", Value: "demo"},
			{Key: "params", Label: "Parameters", Value: 5288548, Text: "5.29 M"},
			{Key: "fits", Label: "Input fits", Value: true},
		},
		Sections: []Section{
			{Title: "Files", Table: &Table{Header: []string{"ROLE", "PATH"}, Rows: [][]string{{"model", "m.onnx"}}}},
			{Title: "Notes", Rows: []Field{{Label: "a", Value: 1}}, Lines: []string{"line <b>1</b>"},
				Image: &Image{Caption: "input", PNG: []byte{0x89, 'P', 'N', 'G'}}},
		},
		NextSteps: []string{"visionserve run demo photo.jpg"},
		Details:   map[string]any{"files": []string{"m.onnx"}},
	}
	return r
}

func TestDecide(t *testing.T) {
	r := &Report{}
	r.Decide("all good")
	if r.Verdict != Pass || r.Reason != "all good" {
		t.Errorf("no findings: %s %q", r.Verdict, r.Reason)
	}
	r = &Report{}
	r.Add(Info, "a note")
	r.Add(Warn, "first warning")
	r.Add(Warn, "second warning")
	r.Decide("unused")
	if r.Verdict != Warn || r.Reason != "first warning" {
		t.Errorf("warn: %s %q", r.Verdict, r.Reason)
	}
	r.Add(Fail, "broken %d", 7)
	r.Reason = ""
	r.Decide("unused")
	if r.Verdict != Fail || r.Reason != "broken 7" {
		t.Errorf("fail: %s %q", r.Verdict, r.Reason)
	}
	if got := Worst(); got != Pass {
		t.Errorf("Worst() = %s", got)
	}
	if got := Worst(Info, Warn, Pass); got != Warn {
		t.Errorf("Worst = %s", got)
	}
}

func TestTextFirstLineIsTheVerdict(t *testing.T) {
	r := sample()
	r.Add(Info, "note")
	r.Add(Warn, "careful\nwith newlines")
	r.Decide("ok")
	var b bytes.Buffer
	if err := r.WriteText(&b); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(b.String(), "\n")
	if lines[0] != "WARN: careful with newlines" {
		t.Errorf("first line = %q", lines[0])
	}
	for _, want := range []string{"  Model       demo", "  Parameters  5.29 M", "Findings\n  WARN  careful",
		"  INFO  note", "Files\n  ROLE   PATH\n  model  m.onnx", "  line <b>1</b>", "[image: input",
		"Next steps\n  visionserve run demo photo.jpg"} {
		if !strings.Contains(b.String(), want) {
			t.Errorf("text lacks %q:\n%s", want, b.String())
		}
	}
	// WARN before INFO although added after it.
	if strings.Index(b.String(), "WARN  careful") > strings.Index(b.String(), "INFO  note") {
		t.Error("findings are not worst first")
	}
}

// The JSON object has exactly the four contract keys; summary keeps its order and typed values;
// details gets the findings.
func TestJSONContract(t *testing.T) {
	r := sample()
	r.Add(Warn, "w")
	r.Decide("ok")
	raw, err := r.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(top))
	for k := range top {
		keys = append(keys, k)
	}
	if strings.Join(sortStrings(keys), ",") != "details,reason,summary,verdict" {
		t.Errorf("top-level keys = %v", keys)
	}
	if !bytes.Contains(raw, []byte(`"summary": {
    "model": "demo",
    "params": 5288548,
    "fits": true
  }`)) {
		t.Errorf("summary not ordered/typed:\n%s", raw)
	}
	var d struct {
		Files    []string  `json:"files"`
		Findings []Finding `json:"findings"`
	}
	if err := json.Unmarshal(top["details"], &d); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(d.Files, []string{"m.onnx"}) || !reflect.DeepEqual(d.Findings, []Finding{{Warn, "w"}}) {
		t.Errorf("details = %+v", d)
	}

	// No details and no findings: details is {"findings": []}.
	raw, err = (&Report{Verdict: Pass, Reason: "r"}).JSON()
	if err != nil || !bytes.Contains(raw, []byte(`"details": {
    "findings": []
  }`)) {
		t.Errorf("empty report JSON = %s, %v", raw, err)
	}
	if _, err := (&Report{Details: []int{1}}).JSON(); err == nil {
		t.Error("details that are not an object must be refused")
	}
	if _, err := (&Report{Summary: []Field{{Key: "a"}, {Key: "a"}}}).JSON(); err == nil {
		t.Error("duplicate summary keys must be refused")
	}
}

func sortStrings(s []string) []string {
	out := append([]string(nil), s...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

func TestHTMLIsSelfContainedAndEscaped(t *testing.T) {
	r := sample()
	r.Add(Fail, `<script>alert(1)</script>`)
	r.Decide("ok")
	var b bytes.Buffer
	if err := r.WriteHTML(&b); err != nil {
		t.Fatal(err)
	}
	h := b.String()
	for _, want := range []string{`<section id="verdict">`, `<section id="summary">`, `<section id="details">`,
		`<section id="next-steps">`, "prefers-color-scheme: dark", `src="data:image/png;base64,iVBORw==`,
		"&lt;script&gt;alert(1)&lt;/script&gt;", "line &lt;b&gt;1&lt;/b&gt;", `class="verdict fail"`} {
		if !strings.Contains(h, want) {
			t.Errorf("HTML lacks %q", want)
		}
	}
	order := []string{`id="verdict"`, `id="summary"`, `id="details"`, `id="next-steps"`}
	for i := 1; i < len(order); i++ {
		if strings.Index(h, order[i-1]) > strings.Index(h, order[i]) {
			t.Errorf("section %s comes after %s", order[i-1], order[i])
		}
	}
	for _, bad := range []string{"<script", "http://", "https://", "<link"} {
		if strings.Contains(h, bad) {
			t.Errorf("HTML contains %q: not self-contained / not escaped", bad)
		}
	}
}

func TestEmitAndExitCodes(t *testing.T) {
	dir := t.TempDir()
	fs := flag.NewFlagSet("x", flag.ContinueOnError)
	o := AddFlags(fs)
	html := filepath.Join(dir, "r.html")
	if err := fs.Parse([]string{"--json", "--report", html}); err != nil {
		t.Fatal(err)
	}
	r := sample()
	r.Add(Fail, "nope")
	r.Decide("ok")
	var out, errOut bytes.Buffer
	err := o.Emit(&out, &errOut, r)
	if ExitCode(err) != ExitFail || !Reported(err) {
		t.Errorf("FAIL: err %v code %d", err, ExitCode(err))
	}
	var v map[string]any
	if json.Unmarshal(out.Bytes(), &v) != nil || v["verdict"] != "FAIL" {
		t.Errorf("stdout is not the JSON report alone: %s", out.String())
	}
	if st, err := os.Stat(html); err != nil || st.Mode().Perm() != 0o644 {
		t.Errorf("HTML report: %v %v", st, err)
	}
	if !strings.Contains(errOut.String(), "report: "+html) {
		t.Errorf("stderr = %q", errOut.String())
	}

	r = sample()
	r.Add(Warn, "hmm")
	r.Decide("ok")
	if err := (&Output{}).Emit(io.Discard, io.Discard, r); err != nil || ExitCode(err) != ExitOK {
		t.Errorf("WARN: %v", err)
	}
	if ExitCode(nil) != 0 || ExitCode(errors.New("x")) != 1 || ExitCode(Usagef("bad %s", "flag")) != 2 || Usage(nil) != nil {
		t.Error("exit codes")
	}
	if Reported(Usagef("x")) || Reported(errors.New("x")) {
		t.Error("only a printed FAIL report is Reported")
	}
	// An unwritable report path is a setup error (2), and the text still went to stdout.
	bad := &Output{HTMLPath: filepath.Join(dir, "no", "such", "dir.html")}
	if err := bad.Emit(io.Discard, io.Discard, sample()); ExitCode(err) != ExitUsage {
		t.Errorf("unwritable report: %v", err)
	}
}
