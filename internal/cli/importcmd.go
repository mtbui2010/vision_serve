package cli

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"visionserve/internal/catalog"
	"visionserve/internal/cli/clireport"
	"visionserve/internal/engine"
	"visionserve/internal/lifecycle"
	"visionserve/internal/registry"
)

const importUsage = `visionserve import <model.onnx> --name NAME --task TASK --license ID [flags]

Makes an existing ONNX file servable: reads its header, writes a manifest.yaml for it, checks
the result the way a load does, and installs the folder into the model registry. Everything
read from the file and everything ASSUMED (not in the file) is printed, so you can confirm it.
No Python, no Docker; for a PyTorch/TensorFlow checkpoint use 'visionserve convert'.

Required:
  --name NAME       registry name (letters, digits, '.', '_', '-')
  --task TASK       classification | detection | depth   (the tasks import can decode from the
                    header alone; detection = DETR-style outputs, e.g. RF-DETR / RT-DETR)
  --license ID      licence of the ORIGINAL model: Apache-2.0 | MIT | BSD-3-Clause | BSD-2-Clause.
                    AGPL (Ultralytics YOLO, FastSAM, YOLO-World) is always refused.

Optional (default: read from the graph, or the usual default, printed as "assumed"):
  --labels FILE     class names, one per line, in output order
  --input WxH       input resolution, e.g. 224x224 (required when the graph's size is dynamic)
  --resize MODE     squash (default) | letterbox | center_crop | keep_aspect (what the architecture allows)
  --mean a,b,c      normalisation after /255 (default ImageNet 0.485,0.456,0.406)
  --std a,b,c       (default ImageNet 0.229,0.224,0.225)
  --layout L        NCHW | NHWC (default: where the graph has its 3 channels)
  --force           replace an installed model of the same name
  --dry-run         check everything and print the manifest; install nothing
  --models DIR      registry (default $VISIONSERVE_MODELS, else ~/.visionserve/models)
  --json            print one JSON object {verdict, reason, summary, details}
  --report FILE     also write a self-contained HTML report

Exit status: 0 PASS or WARN (imported), 1 FAIL (refused, nothing installed), 2 usage error.
`

// importRequest is one `visionserve import` invocation.
type importRequest struct {
	Source                string
	Name, Task, License   string
	Labels                string
	Input, Resize, Layout string
	Mean, Std             string
	Force, DryRun         bool
	ModelsDir             string
}

func runImport(args []string) error {
	fs := flag.NewFlagSet("import", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprint(fs.Output(), importUsage) }
	var q importRequest
	fs.StringVar(&q.Name, "name", "", "registry name")
	fs.StringVar(&q.Task, "task", "", "classification | detection | depth")
	fs.StringVar(&q.License, "license", "", "licence of the original model")
	fs.StringVar(&q.Labels, "labels", "", "class names file")
	fs.StringVar(&q.Input, "input", "", "input resolution WxH")
	fs.StringVar(&q.Resize, "resize", "", "resize mode")
	fs.StringVar(&q.Mean, "mean", "", "normalisation mean a,b,c")
	fs.StringVar(&q.Std, "std", "", "normalisation std a,b,c")
	fs.StringVar(&q.Layout, "layout", "", "NCHW | NHWC")
	fs.BoolVar(&q.Force, "force", false, "replace an installed model of the same name")
	fs.BoolVar(&q.DryRun, "dry-run", false, "check and print the manifest; install nothing")
	modelsFlag := fs.String("models", "", "model registry directory")
	out := clireport.AddFlags(fs)
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return clireport.Usage(err)
	}
	if len(pos) != 1 {
		return clireport.Usagef("usage: visionserve import <model.onnx> --name NAME --task TASK --license ID (see visionserve import --help)")
	}
	q.Source = pos[0]
	q.ModelsDir = modelsDir(*modelsFlag)
	r, err := importModel(q, os.Stderr)
	if err != nil {
		return err
	}
	return out.Emit(os.Stdout, os.Stderr, r)
}

// importDetails is the JSON "details" of import (field names are API).
type importDetails struct {
	Source      string          `json:"source"`
	Manifest    string          `json:"manifest_yaml"`
	Read        []string        `json:"read"`
	Assumed     []string        `json:"assumed"`
	DryRun      bool            `json:"dry_run"`
	InstalledTo string          `json:"installed_to"` // "" = nothing installed
	Card        *inspectDetails `json:"card"`         // the inspect card of the result (null when refused early)
}

// importModel runs one import. The error is a usage error (exit 2); a refusal is a FAIL report.
func importModel(q importRequest, stderr io.Writer) (*clireport.Report, error) {
	// Usage: the flags themselves.
	if q.Name == "" || q.Task == "" {
		return nil, clireport.Usagef("--name and --task are required (see visionserve import --help)")
	}
	st, err := os.Stat(q.Source)
	if err != nil || !st.Mode().IsRegular() {
		return nil, clireport.Usagef("%s: not a file (import takes an .onnx file)", q.Source)
	}
	var opts importOptions
	if q.Input != "" {
		w, h, err := parseWxH(q.Input)
		if err != nil {
			return nil, clireport.Usage(err)
		}
		opts.Width, opts.Height = w, h
	}
	if q.Mean != "" || q.Std != "" {
		if q.Mean == "" || q.Std == "" {
			return nil, clireport.Usagef("--mean and --std go together")
		}
		if opts.Mean, err = parseTriple("--mean", q.Mean); err != nil {
			return nil, clireport.Usage(err)
		}
		if opts.Std, err = parseTriple("--std", q.Std); err != nil {
			return nil, clireport.Usage(err)
		}
	}
	if q.Labels != "" && !fileExists(q.Labels) {
		return nil, clireport.Usagef("--labels %s: no such file", q.Labels)
	}
	opts.Layout, opts.Resize = q.Layout, q.Resize

	r := &clireport.Report{Command: "import", Subject: q.Name}
	d := importDetails{Source: q.Source, Read: []string{}, Assumed: []string{}, DryRun: q.DryRun}
	r.Details = &d
	refused := func(format string, args ...any) *clireport.Report {
		r.Add(clireport.Fail, format, args...)
		r.Summary = importSummary(q, nil, "nothing installed")
		r.NextSteps = []string{"fix the problem above and run the same command again — nothing was installed"}
		r.Decide("")
		return r
	}

	// The licence gate, as the converter and the registry apply it.
	if strings.TrimSpace(q.License) == "" {
		return refused("a --license is required (Apache-2.0, MIT, BSD-3-Clause or BSD-2-Clause). Check the ORIGINAL model's " +
			"license — 'it is on HuggingFace' says nothing about it."), nil
	}
	license, err := registry.CheckLicense(q.License)
	if err != nil {
		return refused("%v", err), nil
	}
	q.License = license
	if !registry.ValidName(q.Name) {
		return refused("name %q is invalid: use letters, digits, '.', '_' or '-', starting with a letter or digit (max 128)", q.Name), nil
	}
	if !registry.ValidTask(q.Task) {
		return refused("task %q is invalid (%s)", q.Task, strings.Join(registry.ValidTasks(), "/")), nil
	}
	if dst := filepath.Join(q.ModelsDir, q.Name); !q.Force && fileExists(filepath.Join(dst, "manifest.yaml")) {
		if !q.DryRun {
			return refused("model %q already exists at %s (use --force to replace it)", q.Name, dst), nil
		}
		r.Add(clireport.Warn, "model %q already exists at %s: installing it needs --force", q.Name, dst)
	}

	// The file: a readable ONNX header, no AGPL marker, external data next to it.
	if err := registry.ScanAGPLMarkers(q.Source); err != nil {
		if errors.Is(err, registry.ErrAGPLModel) {
			return refused("%v", err), nil
		}
		return nil, clireport.Usage(err)
	}
	facts, err := engine.ReadFacts(q.Source)
	if err != nil {
		return refused("%s is not a readable ONNX model: %v", q.Source, err), nil
	}
	for _, e := range facts.External {
		switch {
		case e.Unsafe:
			return refused("%s stores weights at %q, outside its folder: ONNX Runtime refuses that path", q.Source, e.Location), nil
		case e.OnDisk < 0:
			return refused("%s keeps %s of weights in %s, which is missing — keep it next to the .onnx file", q.Source, humanSize(e.Bytes), e.Location), nil
		}
	}

	// What the header says, against the task's decoder.
	plan, err := planImport(facts, q.Task, opts)
	d.Read, d.Assumed = nonNil(plan.Read), nonNil(plan.Assumed)
	if err != nil {
		var rf *refusal
		if errors.As(err, &rf) {
			return refused("%v", err), nil
		}
		return nil, err
	}

	// Labels against the class count.
	if q.Labels != "" {
		n, err := countLabels(q.Labels)
		if err != nil {
			return nil, clireport.Usage(err)
		}
		switch {
		case plan.Classes > 0 && n != plan.Classes && q.Task == "classification":
			return refused("%s has %d names but the model outputs %d class scores: names are given by position, so they would be wrong",
				q.Labels, n, plan.Classes), nil
		case plan.Classes > 0 && n != plan.Classes:
			r.Add(clireport.Warn, "%s has %d names but the model outputs %d class scores: indices past the file are reported as "+
				"class_<i>, and an offset (e.g. COCO's 91-slot layout with an 'N/A' at 0) shifts every name — check it", q.Labels, n, plan.Classes)
		}
		d.Read = append(d.Read, fmt.Sprintf("labels: %d names from %s", n, q.Labels))
	} else if plan.Classes > 0 {
		d.Assumed = append(d.Assumed, fmt.Sprintf("no --labels: classes are reported as class_0 … class_%d", plan.Classes-1))
	}

	// Stage the would-be model folder (symlinks to the source files) and judge it like a load.
	stage, err := os.MkdirTemp("", "visionserve-import-")
	if err != nil {
		return nil, clireport.Usage(err)
	}
	defer os.RemoveAll(stage) // removes the links, never their targets
	onnxName := filepath.Base(q.Source)
	labelsName := ""
	if q.Labels != "" {
		labelsName = "labels.txt"
	}
	links := map[string]string{onnxName: q.Source}
	for _, e := range facts.External {
		links[filepath.FromSlash(e.Location)] = filepath.Join(filepath.Dir(q.Source), e.Location)
	}
	if labelsName != "" {
		if _, clash := links[labelsName]; clash {
			return refused("the model already has a file named %s; rename the labels file", labelsName), nil
		}
		links[labelsName] = q.Labels
	}
	for rel, src := range links {
		abs, err := filepath.Abs(src)
		if err != nil {
			return nil, clireport.Usage(err)
		}
		dst := filepath.Join(stage, rel)
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return nil, clireport.Usage(err)
		}
		// A symlink costs nothing (the install copies the content it points at). Where one cannot
		// be made (Windows without the privilege), the file is copied instead.
		if err := os.Symlink(abs, dst); err != nil {
			if err := copyRegular(abs, dst); err != nil {
				return nil, clireport.Usage(fmt.Errorf("stage %s: %w", rel, err))
			}
		}
	}
	// Pin the bytes the licence was declared for, as pull and convert do: the ONNX file and its
	// external data.
	pins := manifestPins{SideFiles: map[string]string{}}
	if pins.Model, err = registry.FileSHA256(q.Source); err != nil {
		return nil, clireport.Usage(err)
	}
	for _, e := range facts.External {
		sum, err := registry.FileSHA256(filepath.Join(filepath.Dir(q.Source), e.Location))
		if err != nil {
			return nil, clireport.Usage(err)
		}
		pins.SideFiles[filepath.ToSlash(e.Location)] = sum
	}
	yaml := manifestYAML(q, plan, onnxName, labelsName, pins)
	d.Manifest = yaml
	if err := os.WriteFile(filepath.Join(stage, "manifest.yaml"), []byte(yaml), 0o644); err != nil {
		return nil, clireport.Usage(err)
	}
	man, err := registry.LoadManifest(filepath.Join(stage, "manifest.yaml"))
	if err != nil {
		return refused("the generated manifest is refused by the registry: %s", strings.TrimPrefix(err.Error(), "registry: ")), nil
	}
	reg := registry.New(stage)
	reg.Put(man)
	mgr := lifecycle.NewManager(reg)
	check := mgr.Describe(q.Name)
	mgr.Close()
	if check.BuildErr != nil {
		var shapeErr *lifecycle.InputShapeError
		if errors.As(check.BuildErr, &shapeErr) {
			return refused("%s", shapeMismatchText(shapeErr)), nil
		}
		return refused("the model would not load: %s", strings.TrimPrefix(check.BuildErr.Error(), "lifecycle: ")), nil
	}

	assumptions := clireport.Section{Title: "What import read and assumed"}
	for _, s := range d.Read {
		assumptions.Lines = append(assumptions.Lines, "read:    "+s)
	}
	for _, s := range d.Assumed {
		assumptions.Lines = append(assumptions.Lines, "assumed: "+s)
	}
	manifestSec := clireport.Section{Title: "manifest.yaml", Lines: strings.Split(strings.TrimRight(yaml, "\n"), "\n")}

	if q.DryRun {
		card := modelCard(reg, q.Name, inspectOptions{})
		cd := card.Details.(inspectDetails)
		d.Card = &cd
		r.Findings = append(r.Findings, card.Findings...)
		r.Summary = importSummary(q, &plan, "dry run: nothing installed")
		r.Sections = append([]clireport.Section{assumptions, manifestSec}, card.Sections...)
		r.NextSteps = []string{"run the same command without --dry-run to install it"}
		decideImport(r, &d, fmt.Sprintf("%s would import cleanly (dry run, nothing installed)", q.Name))
		return r, nil
	}

	// Install: the same validated, staged, atomic copy as `visionserve pull <folder>`.
	if err := catalog.InstallLocal(stage, catalog.PullOptions{ModelsDir: q.ModelsDir, Force: q.Force, Out: stderr}); err != nil {
		return refused("install failed: %v", err), nil
	}
	installed := filepath.Join(q.ModelsDir, q.Name)
	d.InstalledTo = installed

	// The card of what is now in the registry.
	reg2 := registry.New(q.ModelsDir)
	if _, err := reg2.Scan(); err != nil {
		return nil, clireport.Usage(err)
	}
	card := modelCard(reg2, q.Name, inspectOptions{})
	cd := card.Details.(inspectDetails)
	d.Card = &cd
	r.Findings = append(r.Findings, card.Findings...)
	r.Summary = importSummary(q, &plan, installed)
	r.Sections = append([]clireport.Section{assumptions, manifestSec}, card.Sections...)
	r.NextSteps = card.NextSteps
	decideImport(r, &d, fmt.Sprintf("imported %s into %s", q.Name, installed))
	return r, nil
}

// decideImport sets the verdict: a model imported with assumed settings is a WARN (it may run
// and still be wrong), naming how many there are.
func decideImport(r *clireport.Report, d *importDetails, pass string) {
	if n := len(d.Assumed); n > 0 {
		r.Add(clireport.Warn, "%s, but %d setting%s were assumed, not read from the file — check them under \"What import read and assumed\"",
			pass, n, plural(n))
	}
	r.Decide(pass)
}

func importSummary(q importRequest, p *importPlan, dest string) []clireport.Field {
	fs := []clireport.Field{
		{Key: "model", Label: "Model", Value: q.Name},
		{Key: "task", Label: "Task", Value: q.Task},
		{Key: "license", Label: "Licence", Value: q.License},
		{Key: "source", Label: "From", Value: q.Source},
	}
	arch, input := "", ""
	if p != nil {
		arch = p.Arch
		input = fmt.Sprintf("%d×%d %s, %s, %s", p.Width, p.Height, p.Layout, p.Resize, normName(p.Mean, p.Std))
	}
	fs = append(fs,
		clireport.Field{Key: "architecture", Label: "Architecture", Value: arch, Text: orDefault(arch, "-")},
		clireport.Field{Key: "input", Label: "Input", Value: input, Text: orDefault(input, "-")},
		clireport.Field{Key: "installed_to", Label: "Installed", Value: dest},
	)
	return fs
}

// manifestYAML writes the manifest of an import. Every value is either validated (name, licence,
// task) or formatted here (numbers), and file names are JSON-quoted (valid YAML), so nothing a
// user typed can inject keys.
// manifestPins are the content pins an import writes: sha256 of the ONNX file, and of each
// external-data file (sha256_files, keyed by path relative to the model folder).
type manifestPins struct {
	Model     string
	SideFiles map[string]string
}

func manifestYAML(q importRequest, p importPlan, onnxName, labelsName string, pins manifestPins) string {
	var b strings.Builder
	assumed := map[string]bool{}
	for _, a := range p.Assumed {
		switch {
		case strings.HasPrefix(a, "resize:"):
			assumed["resize"] = true
		case strings.HasPrefix(a, "normalisation:"):
			assumed["norm"] = true
		case strings.HasPrefix(a, "layout"):
			assumed["layout"] = true
		}
	}
	mark := func(k string) string {
		if assumed[k] {
			return "   # assumed: check it"
		}
		return ""
	}
	fmt.Fprintf(&b, "# Written by `visionserve import` from %s.\n", filepath.Base(q.Source))
	b.WriteString("# Lines marked 'assumed' are not in the ONNX file: check them (format: docs/manifest-spec.md).\n")
	fmt.Fprintf(&b, "name: %s\ntask: %s\nlicense: %s\narchitecture: %s\nmodel_file: %s\n", q.Name, q.Task, q.License, p.Arch, jsonQuote(onnxName))
	if pins.Model != "" {
		fmt.Fprintf(&b, "sha256: %s   # binds the licence above to these exact bytes\n", pins.Model)
	}
	if len(pins.SideFiles) > 0 {
		b.WriteString("sha256_files:\n")
		for _, k := range sortedKeys(pins.SideFiles) {
			fmt.Fprintf(&b, "  %s: %s\n", jsonQuote(k), pins.SideFiles[k])
		}
	}
	if labelsName != "" {
		fmt.Fprintf(&b, "labels: %s\n", jsonQuote(labelsName))
	}
	b.WriteString("preprocess:\n")
	fmt.Fprintf(&b, "  resize: %s%s\n", p.Resize, mark("resize"))
	fmt.Fprintf(&b, "  width: %d\n  height: %d\n", p.Width, p.Height)
	fmt.Fprintf(&b, "  mean: %s%s\n", yamlFloats(p.Mean), mark("norm"))
	fmt.Fprintf(&b, "  std: %s%s\n", yamlFloats(p.Std), mark("norm"))
	fmt.Fprintf(&b, "  layout: %s%s\n", p.Layout, mark("layout"))
	b.WriteString("postprocess:\n")
	fmt.Fprintf(&b, "  type: %s\n", p.PostType)
	switch q.Task {
	case "classification":
		b.WriteString("  max_detections: 5   # top-K classes returned\n")
	case "detection":
		b.WriteString("  box_format: cxcywh\n  conf_threshold: 0.5   # assumed: check it\n  max_detections: 100\n")
	}
	b.WriteString("runtime:\n  prefer: [cuda, cpu]\n  idle_unload_seconds: 300\n")
	return b.String()
}

func jsonQuote(s string) string { q, _ := json.Marshal(s); return string(q) }

func yamlFloats(v []float32) string {
	parts := make([]string, len(v))
	for i, f := range v {
		parts[i] = strconv.FormatFloat(float64(f), 'g', -1, 32)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// parseWxH parses "640x480" (or "640X480", "640×480"); a single number means a square.
func parseWxH(s string) (w, h int, err error) {
	s = strings.NewReplacer("X", "x", "×", "x").Replace(strings.TrimSpace(s))
	ws, hs, found := strings.Cut(s, "x")
	if !found {
		hs = ws
	}
	w, err1 := strconv.Atoi(strings.TrimSpace(ws))
	h, err2 := strconv.Atoi(strings.TrimSpace(hs))
	if err1 != nil || err2 != nil || w <= 0 || h <= 0 || w > 16384 || h > 16384 {
		return 0, 0, fmt.Errorf("--input %q: want WxH with 1..16384 pixels a side, e.g. 224x224", s)
	}
	return w, h, nil
}

// parseTriple parses three comma-separated finite numbers (RGB order).
func parseTriple(flagName, s string) ([]float32, error) {
	parts := strings.Split(strings.Trim(strings.TrimSpace(s), "[]"), ",")
	if len(parts) != 3 {
		return nil, fmt.Errorf("%s %q: want three numbers (R,G,B), e.g. 0.5,0.5,0.5", flagName, s)
	}
	out := make([]float32, 3)
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("%s %q: %q is not a finite number", flagName, s, p)
		}
		out[i] = float32(v)
	}
	if flagName == "--std" {
		for _, v := range out {
			if v == 0 {
				return nil, fmt.Errorf("--std %q: a std of 0 divides by zero", s)
			}
		}
	}
	return out, nil
}

// maxLabelsBytes bounds the labels file read (a 100k-class file is a few MB).
const maxLabelsBytes = 64 << 20

// countLabels counts the non-empty lines of a labels file, as registry.LoadLabels reads them.
func countLabels(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, maxLabelsBytes+1))
	if err != nil {
		return 0, err
	}
	if len(raw) > maxLabelsBytes {
		return 0, fmt.Errorf("--labels %s is larger than %d MB", path, maxLabelsBytes>>20)
	}
	n := 0
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) != "" {
			n++
		}
	}
	return n, nil
}

// copyRegular copies the file src to the new file dst.
func copyRegular(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return append([]string(nil), s...)
}
