// Package clireport is the shared output contract of VisionServe's check-style commands
// (`visionserve inspect`, `import`, and the check / bench / sensitivity / optimize streams). A
// command builds one Report and hands it to Output.Emit; this package owns everything the
// commands must agree on, so a user (or a script) reads every command the same way.
//
// # The contract
//
// Text (the default), on stdout:
//
//	PASS: <one sentence: the verdict's reason>          <- always the FIRST line
//
//	  Model     efficientnet-b0                         <- Summary: a short label/value table
//	  Task      classification
//
//	Findings                                            <- every finding, worst first
//	  WARN  labels.txt has 1001 lines but the model outputs 1000 classes
//
//	<Section title>                                     <- Details: titled sections
//	  ...
//
//	Next steps                                          <- commands to try next
//	  visionserve run efficientnet-b0 photo.jpg
//
// The verdict is PASS, WARN or FAIL, for every command. Exit codes (ExitCode, used by main):
// 0 = PASS or WARN, 1 = FAIL, 2 = a usage or setup error (bad flags, a file that does not exist:
// the command could not judge anything; see Usage).
//
// --json prints exactly one JSON object on stdout and nothing else:
//
//	{"verdict": "PASS", "reason": "...", "summary": {...}, "details": {...}}
//
// summary holds Report.Summary as key → value (keys are the Field.Key values: stable snake_case,
// in Summary order); details is Report.Details marshalled (it must marshal to a JSON object) with
// the findings added under details.findings as [{"level","text"}]. Field names are API: add new
// ones, never rename or remove one.
//
// --report FILE.html writes ONE self-contained HTML file (inline CSS, no external asset, images
// as base64 PNG) with the sections Verdict / Summary / Details / Next steps, light and dark via
// prefers-color-scheme. The Python streams mirror this layout (the class names and section ids
// below are part of it: verdict, summary, details, next-steps).
//
// Diagnostics that are not the report (progress, warnings while working) go to stderr.
package clireport

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
)

// Verdict is a command's overall answer. Info is a finding level only (a remark that does not
// lower the verdict); a Report's Verdict is always Pass, Warn or Fail.
type Verdict string

const (
	Pass Verdict = "PASS"
	Warn Verdict = "WARN"
	Fail Verdict = "FAIL"
	Info Verdict = "INFO"
)

// Exit codes of every check-style command.
const (
	ExitOK    = 0 // PASS or WARN
	ExitFail  = 1 // FAIL
	ExitUsage = 2 // usage or setup error: nothing was judged
)

func (v Verdict) rank() int {
	switch v {
	case Fail:
		return 3
	case Warn:
		return 2
	case Pass:
		return 1
	}
	return 0 // Info, ""
}

// ExitCode is the process exit code of a report with this verdict.
func (v Verdict) ExitCode() int {
	if v == Fail {
		return ExitFail
	}
	return ExitOK
}

// Worst returns the most severe of vs (Fail > Warn > Pass); Pass when none is given. Info counts
// as Pass.
func Worst(vs ...Verdict) Verdict {
	w := Pass
	for _, v := range vs {
		if v.rank() > w.rank() {
			w = v
		}
	}
	return w
}

// Field is one summary row: Label and Text for people, Key and Value for --json.
type Field struct {
	Key   string // JSON key in "summary": stable snake_case
	Label string // shown in the text and HTML tables
	Value any    // JSON value (string, number, bool, list, ...)
	Text  string // displayed text; "" = fmt.Sprint(Value)
}

func (f Field) display() string {
	if f.Text != "" {
		return f.Text
	}
	if f.Value == nil {
		return "-"
	}
	return fmt.Sprint(f.Value)
}

// Finding is one thing a command found. The report's verdict is the worst finding's level.
type Finding struct {
	Level Verdict `json:"level"`
	Text  string  `json:"text"`
}

// Section is one titled block of the details. Any of its parts may be empty; they are shown in
// the order Rows, Table, Lines, Image.
type Section struct {
	Title string
	Rows  []Field  // label: value lines (Key and Value unused)
	Table *Table   // a table with a header row
	Lines []string // free text, shown as is (preformatted)
	Image *Image
}

// Table is a simple table: every row has len(Header) cells.
type Table struct {
	Header []string
	Rows   [][]string
}

// Image is a picture embedded in the HTML report (the text output names Caption only).
type Image struct {
	Caption string
	PNG     []byte
}

// Report is the result of one check-style command.
type Report struct {
	// Command and Subject title the HTML page ("inspect", "efficientnet-b0").
	Command, Subject string
	// Verdict and Reason form the first output line. Decide fills them from the findings.
	Verdict Verdict
	Reason  string
	// Summary is the short table under the verdict line (and the JSON "summary" object).
	Summary []Field
	// Findings are listed worst first under the summary (and in JSON details.findings).
	Findings []Finding
	// Sections are the details, in order.
	Sections []Section
	// NextSteps are commands or one-line hints to try next.
	NextSteps []string
	// Details is the command's machine-readable detail, the JSON "details" object. It must
	// marshal to a JSON object (a struct or a map), or be nil.
	Details any
}

// Add records a finding.
func (r *Report) Add(level Verdict, format string, args ...any) {
	r.Findings = append(r.Findings, Finding{Level: level, Text: fmt.Sprintf(format, args...)})
}

// Decide sets the verdict to the worst finding's level and, unless Reason is already set, the
// reason to that finding's text (the first one of that level); with no Warn or Fail finding the
// verdict is Pass and the reason passReason.
func (r *Report) Decide(passReason string) {
	levels := make([]Verdict, len(r.Findings))
	for i, f := range r.Findings {
		levels[i] = f.Level
	}
	r.Verdict = Worst(levels...)
	if r.Reason != "" {
		return
	}
	if r.Verdict == Pass {
		r.Reason = passReason
		return
	}
	for _, f := range r.Findings {
		if f.Level == r.Verdict {
			r.Reason = f.Text
			return
		}
	}
}

// sortedFindings returns the findings worst first, keeping the order within a level.
func (r *Report) sortedFindings() []Finding {
	out := append([]Finding(nil), r.Findings...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Level.rank() > out[j].Level.rank() })
	return out
}

func (r *Report) verdict() Verdict {
	if r.Verdict == "" {
		return Pass
	}
	return r.Verdict
}

// WriteText writes the text form (see the package comment).
func (r *Report) WriteText(w io.Writer) error {
	var b bytes.Buffer
	fmt.Fprintf(&b, "%s: %s\n", r.verdict(), oneLine(r.Reason))
	if len(r.Summary) > 0 {
		b.WriteString("\n")
		tw := tabwriter.NewWriter(&b, 0, 4, 2, ' ', 0)
		for _, f := range r.Summary {
			fmt.Fprintf(tw, "  %s\t%s\n", f.Label, oneLine(f.display()))
		}
		tw.Flush()
	}
	if fs := r.sortedFindings(); len(fs) > 0 {
		b.WriteString("\nFindings\n")
		for _, f := range fs {
			fmt.Fprintf(&b, "  %-4s  %s\n", f.Level, f.Text)
		}
	}
	for _, s := range r.Sections {
		fmt.Fprintf(&b, "\n%s\n", s.Title)
		if len(s.Rows) > 0 {
			tw := tabwriter.NewWriter(&b, 0, 4, 2, ' ', 0)
			for _, f := range s.Rows {
				fmt.Fprintf(tw, "  %s\t%s\n", f.Label, oneLine(f.display()))
			}
			tw.Flush()
		}
		if s.Table != nil {
			tw := tabwriter.NewWriter(&b, 0, 4, 2, ' ', 0)
			fmt.Fprintf(tw, "  %s\n", strings.Join(s.Table.Header, "\t"))
			for _, row := range s.Table.Rows {
				cells := make([]string, len(row))
				for i, c := range row {
					cells[i] = oneLine(c)
				}
				fmt.Fprintf(tw, "  %s\n", strings.Join(cells, "\t"))
			}
			tw.Flush()
		}
		for _, l := range s.Lines {
			fmt.Fprintf(&b, "  %s\n", l)
		}
		if s.Image != nil {
			fmt.Fprintf(&b, "  [image: %s — in the --report HTML]\n", s.Image.Caption)
		}
	}
	if len(r.NextSteps) > 0 {
		b.WriteString("\nNext steps\n")
		for _, s := range r.NextSteps {
			fmt.Fprintf(&b, "  %s\n", s)
		}
	}
	_, err := w.Write(b.Bytes())
	return err
}

// oneLine keeps a table cell or the verdict line on one line.
func oneLine(s string) string {
	return strings.Join(strings.Fields(strings.ReplaceAll(s, "\n", " ")), " ")
}

// JSON returns the --json object (see the package comment), indented, with a final newline.
func (r *Report) JSON() ([]byte, error) {
	var sum bytes.Buffer
	sum.WriteByte('{')
	seen := map[string]bool{}
	for i, f := range r.Summary {
		if f.Key == "" || seen[f.Key] {
			return nil, fmt.Errorf("clireport: summary field %d (%q) needs a unique key", i, f.Label)
		}
		seen[f.Key] = true
		k, _ := json.Marshal(f.Key)
		v, err := json.Marshal(f.Value)
		if err != nil {
			return nil, fmt.Errorf("clireport: summary %s: %w", f.Key, err)
		}
		if i > 0 {
			sum.WriteByte(',')
		}
		sum.Write(k)
		sum.WriteByte(':')
		sum.Write(v)
	}
	sum.WriteByte('}')

	details := map[string]json.RawMessage{}
	if r.Details != nil {
		raw, err := json.Marshal(r.Details)
		if err != nil {
			return nil, fmt.Errorf("clireport: details: %w", err)
		}
		if err := json.Unmarshal(raw, &details); err != nil {
			return nil, fmt.Errorf("clireport: details must marshal to a JSON object: %w", err)
		}
		if details == nil { // the Details value marshalled to null
			details = map[string]json.RawMessage{}
		}
	}
	findings := r.sortedFindings()
	if findings == nil {
		findings = []Finding{}
	}
	fr, _ := json.Marshal(findings)
	details["findings"] = fr
	dr, err := json.Marshal(details) // map keys sorted: stable
	if err != nil {
		return nil, err
	}

	out := struct {
		Verdict Verdict         `json:"verdict"`
		Reason  string          `json:"reason"`
		Summary json.RawMessage `json:"summary"`
		Details json.RawMessage `json:"details"`
	}{r.verdict(), r.Reason, sum.Bytes(), dr}
	raw, err := json.Marshal(out)
	if err != nil {
		return nil, err
	}
	var ind bytes.Buffer
	if err := json.Indent(&ind, raw, "", "  "); err != nil {
		return nil, err
	}
	ind.WriteByte('\n')
	return ind.Bytes(), nil
}

// WriteHTMLFile writes the HTML report to path atomically (a temporary file in the same
// directory, renamed into place), so a reader never sees half a report.
func (r *Report) WriteHTMLFile(path string) error {
	var b bytes.Buffer
	if err := r.WriteHTML(&b); err != nil {
		return err
	}
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".*")
	if err != nil {
		return fmt.Errorf("write report %s: %w", path, err)
	}
	defer os.Remove(tmp.Name()) // no-op after the rename
	if _, err := tmp.Write(b.Bytes()); err != nil {
		tmp.Close()
		return fmt.Errorf("write report %s: %w", path, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("write report %s: %w", path, err)
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil { // CreateTemp makes 0600
		return fmt.Errorf("write report %s: %w", path, err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return fmt.Errorf("write report %s: %w", path, err)
	}
	return nil
}

// Output holds the shared output flags (--json, --report) of a command.
type Output struct {
	JSON     bool
	HTMLPath string
}

// AddFlags defines --json and --report on fs and returns where their values land.
func AddFlags(fs *flag.FlagSet) *Output {
	o := &Output{}
	fs.BoolVar(&o.JSON, "json", false, "print one JSON object {verdict, reason, summary, details} on stdout instead of text")
	fs.StringVar(&o.HTMLPath, "report", "", "also write a self-contained HTML report to this file (e.g. report.html)")
	return o
}

// Emit prints the report on stdout (text, or JSON with --json), writes the HTML report when
// --report was given, and returns the command's result: nil for PASS/WARN, an *ExitError with
// code 1 for FAIL (marked Reported, so main prints no second error line). The HTML path is
// announced on stderr (stdout stays the report alone).
func (o *Output) Emit(stdout, stderr io.Writer, r *Report) error {
	if o.JSON {
		raw, err := r.JSON()
		if err != nil {
			return err
		}
		if _, err := stdout.Write(raw); err != nil {
			return err
		}
	} else if err := r.WriteText(stdout); err != nil {
		return err
	}
	if o.HTMLPath != "" {
		if err := r.WriteHTMLFile(o.HTMLPath); err != nil {
			return &ExitError{Code: ExitUsage, Err: err}
		}
		fmt.Fprintf(stderr, "report: %s\n", o.HTMLPath)
	}
	if r.verdict() == Fail {
		return &ExitError{Code: ExitFail, Err: fmt.Errorf("%s: %s", r.Verdict, r.Reason), Reported: true}
	}
	return nil
}

// ExitError carries a command's exit code. Reported means the command already told the user
// (a FAIL report was printed): main prints nothing more.
type ExitError struct {
	Code     int
	Err      error
	Reported bool
}

func (e *ExitError) Error() string {
	if e.Err == nil {
		return fmt.Sprintf("exit status %d", e.Code)
	}
	return e.Err.Error()
}

func (e *ExitError) Unwrap() error { return e.Err }

// Usage wraps err as a usage or setup error (exit code 2): wrong flags, a missing file — the
// command could not judge anything.
func Usage(err error) error {
	if err == nil {
		return nil
	}
	return &ExitError{Code: ExitUsage, Err: err}
}

// Usagef is Usage(fmt.Errorf(format, args...)).
func Usagef(format string, args ...any) error { return Usage(fmt.Errorf(format, args...)) }

// ExitCode is the process exit code for a command's error: 0 for nil, the ExitError's code, and
// 1 for any other error (as every command has always exited).
func ExitCode(err error) int {
	if err == nil {
		return ExitOK
	}
	var e *ExitError
	if errors.As(err, &e) {
		return e.Code
	}
	return ExitFail
}

// Reported reports whether err was already shown to the user (a printed FAIL report).
func Reported(err error) bool {
	var e *ExitError
	return errors.As(err, &e) && e.Reported
}
