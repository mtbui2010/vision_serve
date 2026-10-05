package cli

import (
	"errors"
	"flag"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"visionserve/internal/catalog"
	"visionserve/internal/cli/clireport"
	"visionserve/internal/engine"
	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/internal/vision/preprocess"
)

const inspectUsage = `visionserve inspect <model-name | model.onnx | model-dir> [flags]

Prints a model card: whether the model is ready to serve (PASS / WARN / FAIL and why), its
files and licence, what each ONNX file declares (inputs, outputs, opset, parameters), the
preprocessing the manifest asks for and whether its tensor fits the graph, and how it would
run (execution providers, sessions, threads). Nothing is loaded into ONNX Runtime.

A bare .onnx file (no manifest) gets the file facts and the 'visionserve import' command line
that would make it servable.

Flags:
  --image PHOTO     run the model's real preprocessing on PHOTO: tensor shape, value range,
                    scale/pad, and the tensor turned back into a picture (<name>-input.png)
  --image-out FILE  where --image writes that picture (default <name>-input.png)
  --prompt TEXT     text prompt for --image on open-vocabulary models
  --models DIR      model registry (default $VISIONSERVE_MODELS, else ~/.visionserve/models)
  --tensorrt        show the execution-provider chain with TensorRT opted in
  --json            print one JSON object {verdict, reason, summary, details} instead of text
  --report FILE     also write a self-contained HTML report

Exit status: 0 PASS or WARN, 1 FAIL, 2 usage error.
`

// inspectOptions are the inputs of one inspect run.
type inspectOptions struct {
	ModelsDir string
	Image     string // --image: photo to preprocess
	ImageOut  string // --image-out ("" = <name>-input.png)
	Prompt    string
	Stderr    io.Writer
}

// runInspect: visionserve inspect <model | file.onnx | dir>.
func runInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprint(fs.Output(), inspectUsage) }
	var o inspectOptions
	modelsFlag := fs.String("models", "", "model registry directory")
	fs.StringVar(&o.Image, "image", "", "photo to run the model's preprocessing on")
	fs.StringVar(&o.ImageOut, "image-out", "", "where --image writes the tensor as a picture")
	fs.StringVar(&o.Prompt, "prompt", "", "text prompt for --image on open-vocabulary models")
	trt := addTensorRTFlag(fs)
	out := clireport.AddFlags(fs)
	pos, err := parseInterleaved(fs, args)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return clireport.Usage(err)
	}
	if len(pos) != 1 {
		return clireport.Usagef("usage: visionserve inspect <model-name | model.onnx | model-dir> [--image photo.jpg] [--json] [--report r.html]")
	}
	applyTensorRTFlag(trt)
	o.ModelsDir = modelsDir(*modelsFlag)
	o.Stderr = os.Stderr
	r, err := inspect(pos[0], o)
	if err != nil {
		return err
	}
	return out.Emit(os.Stdout, os.Stderr, r)
}

// parseInterleaved parses flags that may come before, between or after the positionals
// (`inspect rf-detr --json`), and returns the positionals.
func parseInterleaved(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	rem := args
	for {
		if err := fs.Parse(rem); err != nil {
			return nil, err
		}
		rem = fs.Args()
		if len(rem) == 0 {
			return pos, nil
		}
		pos = append(pos, rem[0])
		rem = rem[1:]
	}
}

// inspect builds the card of target: a registry model name, a model directory or an .onnx
// file. The error is a usage error (nothing could be judged); every finding about the model is
// in the report.
func inspect(target string, o inspectOptions) (*clireport.Report, error) {
	if o.Stderr == nil {
		o.Stderr = io.Discard
	}
	st, statErr := os.Stat(target)
	switch {
	case statErr == nil && st.IsDir():
		mpath := filepath.Join(target, "manifest.yaml")
		if _, err := os.Stat(mpath); err == nil {
			return inspectManifestFile(mpath, o), nil
		}
		onnx, _ := filepath.Glob(filepath.Join(target, "*.onnx"))
		if len(onnx) == 1 {
			return inspectBareONNX(onnx[0], o), nil
		}
		return nil, clireport.Usagef("%s has no manifest.yaml and %d .onnx files: name the .onnx file to inspect", target, len(onnx))
	case statErr == nil:
		if strings.EqualFold(filepath.Ext(target), ".yaml") || strings.EqualFold(filepath.Ext(target), ".yml") {
			return inspectManifestFile(target, o), nil
		}
		return inspectBareONNX(target, o), nil
	case strings.ContainsRune(target, filepath.Separator) || strings.HasSuffix(strings.ToLower(target), ".onnx"):
		return nil, clireport.Usagef("%s: no such file or directory", target)
	}

	// A registry name.
	reg := registry.New(o.ModelsDir)
	if _, err := reg.Scan(); err != nil { // the warnings about this model are reported below
		return nil, clireport.Usage(err)
	}
	if _, ok := reg.Get(target); !ok {
		// A model directory whose manifest the scan refused: show why, as a FAIL card.
		mpath := filepath.Join(reg.Root(), target, "manifest.yaml")
		if _, err := os.Stat(mpath); err == nil {
			return inspectManifestFile(mpath, o), nil
		}
		// A folder in the registry without a manifest (weights copied in by hand).
		if dir := filepath.Join(reg.Root(), target); validRegistryDir(target) && isDir(dir) {
			return inspect(dir, o)
		}
		return nil, clireport.Usagef("model %q is not in the registry %s (see `visionserve list`; a file or folder path also works)",
			target, reg.Root())
	}
	entry, _ := reg.Get(target)
	r := modelCard(reg, target, o)
	addUnknownKeys(r, entry.Manifest)
	decideModel(r, target)
	return r, nil
}

// addUnknownKeys warns about manifest keys no setting reads (a typo keeps the default silently).
func addUnknownKeys(r *clireport.Report, man *registry.Manifest) {
	for _, k := range man.UnknownKeys() {
		r.Add(clireport.Warn, "manifest key %s is not a VisionServe setting and is ignored (a typo?)", k)
	}
}

// validRegistryDir reports whether a name is a plain directory name (no separator, not "." or
// ".."), so joining it to the registry root stays inside it.
func validRegistryDir(name string) bool {
	return registry.ValidName(name) && name != "." && name != ".."
}

// inspectManifestFile is the card of the model whose manifest is at mpath, outside or inside a
// registry. A manifest the registry refuses is a FAIL card with the registry's reason.
func inspectManifestFile(mpath string, o inspectOptions) *clireport.Report {
	man, err := registry.LoadManifest(mpath)
	if err != nil {
		r := &clireport.Report{Command: "inspect", Subject: filepath.Base(filepath.Dir(mpath))}
		r.Add(clireport.Fail, "%s", strings.TrimPrefix(err.Error(), "registry: "))
		r.Summary = []clireport.Field{{Key: "manifest", Label: "Manifest", Value: mpath}}
		r.NextSteps = []string{"fix " + mpath + " (format: docs/manifest-spec.md), then run `visionserve inspect` again"}
		r.Details = inspectDetails{Files: []fileDetails{}, ONNX: []onnxDetails{}, LoadError: err.Error()}
		r.Decide("")
		return r
	}
	reg := registry.New(filepath.Dir(man.Dir()))
	reg.Put(man)
	r := modelCard(reg, man.Name, o)
	addUnknownKeys(r, man)
	decideModel(r, man.Name)
	return r
}

// decideModel sets the verdict of a model card.
func decideModel(r *clireport.Report, name string) {
	d, _ := r.Details.(inspectDetails)
	pass := name + " is ready to serve: the manifest is valid, the weights are present and nothing a load checks is wrong"
	if p := d.Preprocess; p != nil && p.Fits != nil && *p.Fits {
		pass = fmt.Sprintf("%s is ready to serve: its preprocessing makes a %s tensor and %s accepts it",
			name, dimsText(p.Produces, nil), p.GraphFile)
	}
	r.Reason = ""
	r.Decide(pass)
}

// --- JSON details (field names are API: add, never rename) ---

type inspectDetails struct {
	Model         *modelDetails      `json:"model"` // null for a bare .onnx file
	Files         []fileDetails      `json:"files"`
	ONNX          []onnxDetails      `json:"onnx"`
	Preprocess    *preprocessDetails `json:"preprocess"`
	Runtime       *runtimeDetails    `json:"runtime"`
	Image         *imageDetails      `json:"image"`                    // with --image
	LoadError     string             `json:"load_error,omitempty"`     // what a load would fail with
	ImportCommand string             `json:"import_command,omitempty"` // bare .onnx: how to make it servable
}

type modelDetails struct {
	Name           string `json:"name"`
	Task           string `json:"task"`
	Architecture   string `json:"architecture"`
	License        string `json:"license"`
	LicenseAllowed bool   `json:"license_allowed"`
	Dir            string `json:"dir"`
	Manifest       string `json:"manifest"`
	SourceURL      string `json:"source_url,omitempty"`
	Labels         string `json:"labels,omitempty"` // labels file, relative to dir
	LabelCount     int    `json:"label_count"`      // lines in the labels file (0 = none)
	OutputClasses  int    `json:"output_classes"`   // class scores the decoder reads (-1 = unknown / not a classifier)
}

type fileDetails struct {
	Role   string `json:"role"` // a files: role, "model", "labels", "external_data" or "side_file"
	Path   string `json:"path"` // relative to the model directory
	Bytes  int64  `json:"bytes"`
	Exists bool   `json:"exists"`
	// Pin is "ok", "mismatch", "not_pinned" or "missing"; SHA256 is computed only for a pinned file.
	Pin          string `json:"pin"`
	PinnedSHA256 string `json:"pinned_sha256,omitempty"`
	SHA256       string `json:"sha256,omitempty"`
}

type onnxDetails struct {
	Path            string            `json:"path"`
	Roles           []string          `json:"roles"`
	Bytes           int64             `json:"bytes"`
	Error           string            `json:"error,omitempty"`
	IRVersion       int64             `json:"ir_version"`
	Opsets          map[string]int64  `json:"opsets"` // domain ("ai.onnx" for the default) → version
	Producer        string            `json:"producer"`
	ProducerVersion string            `json:"producer_version"`
	Nodes           int               `json:"nodes"`
	Initializers    int               `json:"initializers"`
	Params          int64             `json:"params"`
	ParamsByType    map[string]int64  `json:"params_by_type"`
	WeightBytes     int64             `json:"weight_bytes"`
	External        []externalDetails `json:"external_data"`
	Inputs          []tensorDetails   `json:"inputs"`
	Outputs         []tensorDetails   `json:"outputs"`
	Metadata        map[string]string `json:"metadata"`
}

type externalDetails struct {
	Location string `json:"location"`
	Tensors  int    `json:"tensors"`
	Bytes    int64  `json:"bytes"`   // declared
	OnDisk   int64  `json:"on_disk"` // -1 = missing
}

type tensorDetails struct {
	Name  string   `json:"name"`
	Dtype string   `json:"dtype"`
	Shape []int64  `json:"shape"` // -1 = dynamic
	Dims  []string `json:"dims"`  // display form: numbers, or the dynamic dim's name ("?" unnamed)
}

type preprocessDetails struct {
	Resize     string    `json:"resize"`
	Width      int       `json:"width"`
	Height     int       `json:"height"`
	MultipleOf int       `json:"multiple_of,omitempty"`
	CropPct    float32   `json:"crop_pct,omitempty"` // center_crop: kept fraction of the resized short side
	Resample   string    `json:"resample"`
	Mean       []float32 `json:"mean"`
	Std        []float32 `json:"std"`
	Rescale    bool      `json:"rescale"` // pixels divided by 255 first
	Layout     string    `json:"layout"`
	Pad        float32   `json:"pad"`
	Produces   []int64   `json:"produces"` // -1 = varies with the photo; null = not probed
	GraphFile  string    `json:"graph_file,omitempty"`
	GraphInput string    `json:"graph_input,omitempty"`
	GraphShape []int64   `json:"graph_shape"`
	Fits       *bool     `json:"fits"`                  // null = not judged (NotChecked says why)
	NotChecked string    `json:"not_checked,omitempty"` // why the fit was not judged
	// SetByModel is true for a pipeline model, which builds its session inputs itself: the values
	// above are what the manifest declares, and --image shows what the model really makes.
	SetByModel bool `json:"set_by_model"`
	// Resolved is true when the values are the spec the model's code really applies (defaults
	// and the architecture's reading of legacy fields included), false when they are the
	// manifest's declaration.
	Resolved bool `json:"resolved"`
	// Example is the model's own preprocessing run on a 640×360 gray photo (plain models).
	Example *exampleDetails `json:"example"`
}

type exampleDetails struct {
	PhotoWidth  int     `json:"photo_width"`
	PhotoHeight int     `json:"photo_height"`
	Tensor      []int64 `json:"tensor"`
	ScaleX      float64 `json:"scale_x"`
	ScaleY      float64 `json:"scale_y"`
	PadX        int     `json:"pad_x"`
	PadY        int     `json:"pad_y"`
}

type runtimeDetails struct {
	Providers         []string         `json:"providers"`
	TensorRT          string           `json:"tensorrt"` // "off", "on", "requested, library missing", "n/a"
	IdleUnloadSeconds int              `json:"idle_unload_seconds"`
	Sessions          []sessionDetails `json:"sessions"`
	// The client-resize hint GET /api/models publishes (lifecycle.UsefulSide); null = SDKs send
	// full-resolution images.
	MaxUsefulSide      *int `json:"max_useful_side"`
	MaxUsefulShortSide *int `json:"max_useful_short_side"`
}

type sessionDetails struct {
	Role        string `json:"role"`
	Path        string `json:"path"`
	Pool        int    `json:"pool"`
	Threads     int    `json:"threads"` // 0 = ONNX Runtime default
	ThreadsFrom string `json:"threads_from"`
}

type imageDetails struct {
	Path     string    `json:"path"`
	Width    int       `json:"width"`
	Height   int       `json:"height"`
	Input    string    `json:"input"` // the ONNX input shown
	Role     string    `json:"role"`
	Shape    []int64   `json:"shape"`
	Dtype    string    `json:"dtype"`
	Min      float64   `json:"min"`
	Max      float64   `json:"max"`
	Mean     []float64 `json:"channel_mean"`
	Meta     *metaJSON `json:"meta"`
	Picture  string    `json:"picture"` // the PNG written ("" = none)
	Inverted string    `json:"picture_from"`
	Inputs   []string  `json:"all_inputs"` // every tensor the first session gets: name shape dtype
}

type metaJSON struct {
	OrigWidth  int     `json:"orig_width"`
	OrigHeight int     `json:"orig_height"`
	ScaleX     float64 `json:"scale_x"`
	ScaleY     float64 `json:"scale_y"`
	PadX       int     `json:"pad_x"`
	PadY       int     `json:"pad_y"`
}

// --- the card of a manifest model ---

// modelCard builds the card of the registered model name (its verdict is set by decideModel,
// after the caller adds its own findings).
func modelCard(reg *registry.Registry, name string, o inspectOptions) *clireport.Report {
	r := &clireport.Report{Command: "inspect", Subject: name}
	mgr := lifecycle.NewManager(reg)
	defer mgr.Close()
	plan := mgr.Describe(name)
	man := plan.Manifest
	if man == nil { // not reachable: callers resolve the name first
		r.Add(clireport.Fail, "model %q is not in the registry", name)
		return r
	}
	dir := man.Dir()
	rel := func(p string) string {
		if rp, err := filepath.Rel(dir, p); err == nil {
			return filepath.ToSlash(rp)
		}
		return p
	}
	d := inspectDetails{Files: []fileDetails{}, ONNX: []onnxDetails{}}
	if plan.BuildErr != nil {
		d.LoadError = plan.BuildErr.Error()
	}

	// Model.
	_, licErr := registry.CheckLicense(man.License)
	md := &modelDetails{Name: man.Name, Task: man.Task, Architecture: man.ArchOrName(), License: man.License,
		LicenseAllowed: licErr == nil, Dir: dir, Manifest: filepath.Join(dir, "manifest.yaml"),
		SourceURL: man.SourceURL, OutputClasses: -1}
	if man.Labels != "" {
		md.Labels = rel(man.LabelsPath())
		if labels, err := man.LoadLabels(); err == nil {
			md.LabelCount = len(labels)
		}
	}
	d.Model = md

	// Files: the session files by role, then external data, labels and pinned side files.
	roles := map[string]string{}
	if files := man.FilesAbs(); len(files) > 0 {
		roles = files
	} else {
		roles["model"] = man.ModelFilePath()
	}
	roleNames := make([]string, 0, len(roles))
	for role := range roles {
		roleNames = append(roleNames, role)
	}
	sort.Strings(roleNames)
	var total int64
	counted := map[string]bool{}
	addFile := func(f fileDetails, abs string) {
		if f.Exists && !counted[abs] {
			counted[abs] = true
			total += f.Bytes
		}
		d.Files = append(d.Files, f)
	}
	byPath := map[string][]string{} // ONNX path → roles
	var onnxPaths []string
	for _, role := range roleNames {
		p := roles[role]
		pinRole := role
		if len(man.Files) == 0 {
			pinRole = ""
		}
		want, pinned := man.SHA256.For(pinRole)
		addFile(fileFacts(role, rel(p), p, want, pinned), p)
		if _, seen := byPath[p]; !seen {
			onnxPaths = append(onnxPaths, p)
		}
		byPath[p] = append(byPath[p], role)
	}

	// ONNX files.
	var params int64
	var factsByPath = map[string]*engine.Facts{}
	for _, p := range onnxPaths {
		od := onnxDetails{Path: rel(p), Roles: byPath[p], Opsets: map[string]int64{}, ParamsByType: map[string]int64{},
			External: []externalDetails{}, Inputs: []tensorDetails{}, Outputs: []tensorDetails{}, Metadata: map[string]string{}}
		f, err := engine.ReadFacts(p)
		switch {
		case errors.Is(err, os.ErrNotExist):
			od.Error = "missing"
			r.Add(clireport.Fail, "weights file %s is missing — download it (see the README in %s) or fix the manifest", rel(p), dir)
		case err != nil:
			od.Error = err.Error()
			r.Add(clireport.Fail, "%s is not a readable ONNX model: %v", rel(p), err)
		default:
			factsByPath[p] = f
			fillONNX(&od, f)
			params += f.Params
			for _, e := range f.External {
				abs := filepath.Join(filepath.Dir(p), e.Location)
				fd := fileDetails{Role: "external_data", Path: rel(abs), Bytes: e.OnDisk, Exists: e.OnDisk >= 0, Pin: "not_pinned"}
				if want, ok := man.SHA256Files[filepath.ToSlash(fd.Path)]; ok {
					fd = fileFacts("external_data", fd.Path, abs, want, true)
				}
				switch {
				case e.Unsafe:
					r.Add(clireport.Fail, "%s stores weights at %q, outside its folder: ONNX Runtime refuses that path", rel(p), e.Location)
				case e.OnDisk < 0:
					fd.Pin = "missing"
					r.Add(clireport.Fail, "%s keeps %s of weights in %s, which is missing — copy it next to the .onnx file",
						rel(p), humanSize(e.Bytes), e.Location)
				case e.OnDisk < e.Bytes:
					r.Add(clireport.Fail, "%s is %s but %s expects %s in it (truncated download?)",
						e.Location, humanSize(e.OnDisk), rel(p), humanSize(e.Bytes))
				}
				addFile(fd, abs)
			}
			if err := registry.ScanAGPLMarkers(p); errors.Is(err, registry.ErrAGPLModel) {
				r.Add(clireport.Fail, "%v", err)
			}
		}
		d.ONNX = append(d.ONNX, od)
	}
	if man.Labels != "" {
		p := man.LabelsPath()
		want, pinned := man.SHA256Files[filepath.ToSlash(rel(p))]
		addFile(fileFacts("labels", rel(p), p, want, pinned), p)
	}
	for _, side := range sortedKeys(man.SHA256Files) {
		abs := filepath.Join(dir, side)
		if counted[abs] || abs == man.LabelsPath() {
			continue
		}
		addFile(fileFacts("side_file", filepath.ToSlash(side), abs, man.SHA256Files[side], true), abs)
	}
	for _, f := range d.Files {
		switch {
		case f.Pin == "mismatch":
			r.Add(clireport.Fail, "%s does not match its pinned sha256 (%s…): the file changed or the download is corrupt", f.Path, short(f.PinnedSHA256))
		case !f.Exists && f.Role == "labels":
			r.Add(clireport.Fail, "labels file %s is missing", f.Path)
		case !f.Exists && f.Role == "side_file":
			r.Add(clireport.Fail, "pinned file %s is missing", f.Path)
		}
	}

	// Licence.
	if licErr != nil {
		r.Add(clireport.Fail, "%v", licErr)
	}
	if man.SHA256.IsEmpty() {
		r.Add(clireport.Info, "no sha256 pin: the declared licence is not bound to these exact weight bytes (pull and convert pin them)")
	}

	// The load decision (everything Load checks before ONNX Runtime).
	if plan.BuildErr != nil && !alreadyReported(r, plan.BuildErr) {
		var shapeErr *lifecycle.InputShapeError
		if errors.As(plan.BuildErr, &shapeErr) {
			r.Add(clireport.Fail, "%s", shapeMismatchText(shapeErr))
		} else {
			r.Add(clireport.Fail, "a load would fail: %s", strings.TrimPrefix(plan.BuildErr.Error(), "lifecycle: "))
		}
	}

	// Output classes vs labels.
	if f := factsByPath[man.ModelFilePath()]; f != nil && len(man.Files) == 0 {
		md.OutputClasses = outputClasses(man.ArchOrName(), f.Outputs)
		if md.OutputClasses > 0 && md.LabelCount > 0 && md.OutputClasses != md.LabelCount {
			r.Add(clireport.Warn, "%s has %d lines but the model outputs %d class scores: classes are named by position, so every name past the first mismatch is wrong",
				md.Labels, md.LabelCount, md.OutputClasses)
		}
	}

	// Preprocessing: what the model really applies when it can say (models.PreprocessReporter),
	// else what the manifest declares.
	spec, err := man.PreprocessSpec()
	resolved := false
	if rep, ok := plan.Model.(models.PreprocessReporter); ok {
		if s, rerr := rep.ResolvedPreprocess(); rerr == nil {
			spec, err, resolved = s, nil, true
		}
	}
	if err == nil {
		d.Preprocess = preprocessFacts(spec, plan.Input, rel)
		d.Preprocess.Resolved = resolved
		switch mdl := plan.Model.(type) {
		case models.PipelineModel:
			d.Preprocess.SetByModel = true
		case models.Model:
			d.Preprocess.Example = exampleMapping(mdl)
		}
		if plan.Input.Skipped != "" && plan.Model != nil {
			r.Add(clireport.Info, "input shape not checked: %s", skippedText(plan.Input.Skipped))
		}
	}

	// Runtime (the EP chain is the manifest's even when the model cannot be built).
	if plan.Providers == nil {
		plan.Providers, _ = man.Providers()
		plan.IdleUnloadSeconds = man.Runtime.IdleUnloadSeconds
	}
	if plan.Providers != nil {
		d.Runtime = runtimeFacts(plan, rel)
		if u := lifecycle.UsefulSide(man); u.Long > 0 {
			d.Runtime.MaxUsefulSide = &u.Long
		} else if u.Short > 0 {
			d.Runtime.MaxUsefulShortSide = &u.Short
		}
	}

	// --image.
	var imageSec *clireport.Section
	if o.Image != "" {
		if plan.Model == nil {
			r.Add(clireport.Warn, "--image skipped: the model cannot be built (see above)")
		} else {
			img, sec, err := previewInput(mgr, man, plan.Model, o)
			if err != nil {
				r.Add(clireport.Warn, "--image: %v", err)
			} else {
				d.Image, imageSec = img, &sec
			}
		}
	}

	r.Details = d
	r.Summary = modelSummary(man, d, total, params)
	r.Sections = modelSections(man, d)
	if imageSec != nil {
		r.Sections = append(r.Sections, *imageSec)
	}
	r.NextSteps = nextSteps(man, o.Image == "")
	if errors.Is(plan.BuildErr, lifecycle.ErrModelNotFound) {
		if _, ok := catalog.Lookup(man.Name); ok {
			r.NextSteps = append([]string{"visionserve pull " + man.Name + "   # download the weights"}, r.NextSteps...)
		}
	}
	return r
}

// alreadyReported reports whether a finding already says what err says (a missing weights file
// is both an ONNX finding and Load's ErrModelNotFound).
func alreadyReported(r *clireport.Report, err error) bool {
	if errors.Is(err, lifecycle.ErrModelNotFound) {
		for _, f := range r.Findings {
			if f.Level == clireport.Fail && strings.Contains(f.Text, "missing") {
				return true
			}
		}
	}
	if strings.Contains(err.Error(), "sha256") || strings.Contains(err.Error(), "digest") {
		for _, f := range r.Findings {
			if strings.Contains(f.Text, "pinned sha256") {
				return true
			}
		}
	}
	return false
}

func shapeMismatchText(e *lifecycle.InputShapeError) string {
	return fmt.Sprintf("the manifest prepares a %s tensor (input %d×%d, layout %s) but %s input %q takes %s — make the manifest's input size and layout match the export, or re-export the model",
		dimsText(e.Produced, nil), e.Width, e.Height, orDefault(e.Layout, "NCHW"), filepath.Base(e.File), e.Input, dimsText(e.Graph, nil))
}

func skippedText(s string) string {
	switch {
	case strings.HasPrefix(s, "pipeline model"):
		return "the model feeds its sessions itself (prompts, several inputs), so which tensor goes where is not guessed; use --image to see the first one"
	case strings.HasPrefix(s, "no unambiguous input"):
		return "the graph has several inputs and the model does not name the image one (" + s + ")"
	}
	return s
}

func fileFacts(role, relPath, abs, pinned string, isPinned bool) fileDetails {
	f := fileDetails{Role: role, Path: relPath, Bytes: -1, Pin: "not_pinned"}
	st, err := os.Stat(abs)
	if err != nil || !st.Mode().IsRegular() {
		f.Pin = "missing"
		return f
	}
	f.Exists, f.Bytes = true, st.Size()
	if isPinned && pinned != "" {
		f.PinnedSHA256 = pinned
		sum, err := registry.FileSHA256(abs)
		switch {
		case err != nil:
			f.Pin = "missing"
		case strings.EqualFold(sum, pinned):
			f.SHA256, f.Pin = sum, "ok"
		default:
			f.SHA256, f.Pin = sum, "mismatch"
		}
	}
	return f
}

func fillONNX(od *onnxDetails, f *engine.Facts) {
	od.Bytes = f.FileBytes
	od.IRVersion = f.IRVersion
	for _, op := range f.Opsets {
		dom := op.Domain
		if dom == "" {
			dom = "ai.onnx"
		}
		od.Opsets[dom] = op.Version
	}
	od.Producer, od.ProducerVersion = f.ProducerName, f.ProducerVersion
	od.Nodes, od.Initializers = f.Nodes, f.Initializers
	od.Params, od.WeightBytes = f.Params, f.WeightBytes
	for k, v := range f.ParamsByType {
		od.ParamsByType[k] = v
	}
	for k, v := range f.Metadata {
		od.Metadata[k] = v
	}
	for _, e := range f.External {
		od.External = append(od.External, externalDetails{Location: e.Location, Tensors: e.Tensors, Bytes: e.Bytes, OnDisk: e.OnDisk})
	}
	conv := func(src []engine.IOFacts) []tensorDetails {
		out := make([]tensorDetails, 0, len(src))
		for _, v := range src {
			td := tensorDetails{Name: v.Name, Dtype: orUnknown(engine.ElemTypeName(v.ElemType)), Shape: v.Shape, Dims: []string{}}
			if td.Shape == nil {
				td.Shape = []int64{}
			}
			txt := dimsText(v.Shape, v.DimNames)
			if len(v.Shape) > 0 {
				td.Dims = strings.Split(strings.Trim(txt, "[]"), ",")
			}
			out = append(out, td)
		}
		return out
	}
	od.Inputs, od.Outputs = conv(f.Inputs), conv(f.Outputs)
}

// outputClasses is the number of class scores the architecture's decoder reads from these
// outputs (-1 = unknown, dynamic, or not a classifier/detector).
func outputClasses(arch string, outs []engine.IOFacts) int {
	switch arch {
	case "efficientnet", "mobilenet-v3":
		if len(outs) > 0 && len(outs[0].Shape) > 0 {
			if c := outs[0].Shape[len(outs[0].Shape)-1]; c > 0 {
				return int(c)
			}
		}
	case "rf-detr", "rt-detr":
		f := &engine.Facts{Inputs: []engine.IOFacts{{IOInfo: engine.IOInfo{Name: "x", ElemType: 1, Shape: []int64{1, 3, 1, 1}}}}, Outputs: outs}
		if p, err := planImport(f, "detection", importOptions{Width: 1, Height: 1}); err == nil {
			return p.Classes
		}
	}
	return -1
}

func preprocessFacts(s preprocess.Spec, fit lifecycle.InputFit, rel func(string) string) *preprocessDetails {
	p := &preprocessDetails{Resize: string(s.Resize), Width: s.Width, Height: s.Height, MultipleOf: s.MultipleOf, CropPct: s.CropPct,
		Mean: s.Mean, Std: s.Std, Rescale: !s.NoRescale, Layout: string(s.Layout), Pad: s.PadValue,
		Produces: fit.Produced, GraphInput: fit.Input, GraphShape: fit.Graph, NotChecked: fit.Skipped}
	if p.Resize == "" {
		p.Resize = "architecture default"
	}
	if p.Layout == "" {
		p.Layout = "NCHW"
	}
	if s.Resize != "" {
		p.Resample = filterName(s)
	} else if s.Resample != "" {
		p.Resample = string(s.Resample)
	} else {
		p.Resample = "mode default"
	}
	if p.Mean == nil {
		p.Mean, p.Std = []float32{}, []float32{}
	}
	if fit.File != "" {
		p.GraphFile = rel(fit.File)
	}
	if fit.Skipped == "" && fit.Produced != nil {
		ok := fit.Err == nil
		p.Fits = &ok
	}
	return p
}

// exampleMapping runs a plain model's own Preprocess on a 640×360 gray photo and reports where
// the photo lands in the tensor — the real geometry, whatever the manifest's words say (a legacy
// SCRFD `letterbox` pads at the top-left, for instance). nil when the model refuses the probe.
func exampleMapping(mdl models.Model) *exampleDetails {
	img := image.NewNRGBA(image.Rect(0, 0, 640, 360))
	for i := range img.Pix {
		img.Pix[i] = 128
	}
	t, meta, err := mdl.Preprocess(img)
	if err != nil {
		return nil
	}
	return &exampleDetails{PhotoWidth: 640, PhotoHeight: 360, Tensor: t.Shape,
		ScaleX: meta.ScaleX, ScaleY: meta.ScaleY, PadX: meta.PadX, PadY: meta.PadY}
}

func filterName(s preprocess.Spec) string {
	if s.Resample != "" {
		return string(s.Resample)
	}
	if s.Filter().Support > 1.5 { // CatmullRom (support 2) vs Linear (support 1)
		return string(preprocess.Bicubic)
	}
	return string(preprocess.Bilinear)
}

func runtimeFacts(plan lifecycle.LoadPlan, rel func(string) string) *runtimeDetails {
	rt := &runtimeDetails{IdleUnloadSeconds: plan.IdleUnloadSeconds, Sessions: []sessionDetails{}, TensorRT: "n/a"}
	hasCUDA := false
	for _, p := range plan.Providers {
		rt.Providers = append(rt.Providers, string(p))
		hasCUDA = hasCUDA || p == engine.ProviderCUDA || p == engine.ProviderTensorRT
	}
	switch {
	case engine.EPOverride() != "":
		rt.TensorRT = "VISIONSERVE_EP replaces the chain"
	case !hasCUDA:
	case engine.TensorRTRequested() && engine.TRTAvailable():
		rt.TensorRT = "on"
	case engine.TensorRTRequested():
		rt.TensorRT = "requested, library missing (falls back to CUDA)"
	default:
		rt.TensorRT = "off"
	}
	for _, s := range plan.Sessions {
		rt.Sessions = append(rt.Sessions, sessionDetails{Role: s.Role, Path: rel(s.Path), Pool: s.Pool, Threads: s.Threads, ThreadsFrom: s.ThreadsFrom})
	}
	return rt
}
