package cli

import (
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"visionserve/internal/cli/clireport"
	"visionserve/internal/engine"
	"visionserve/internal/registry"
	"visionserve/pkg/api"
)

// --- rendering ---

func modelSummary(man *registry.Manifest, d inspectDetails, total, params int64) []clireport.Field {
	md := d.Model
	lic := md.License + " (allowed)"
	if !md.LicenseAllowed {
		lic = md.License + " (NOT allowed)"
	}
	nOnnx := len(d.ONNX)
	weights := fmt.Sprintf("%d ONNX file%s, %s in all (with labels and side files)", nOnnx, plural(nOnnx), humanSize(total))
	fs := []clireport.Field{
		{Key: "model", Label: "Model", Value: md.Name},
		{Key: "task", Label: "Task", Value: md.Task},
		{Key: "architecture", Label: "Architecture", Value: md.Architecture},
		{Key: "license", Label: "Licence", Value: md.License, Text: lic},
		{Key: "total_bytes", Label: "Files", Value: total, Text: weights},
		{Key: "params", Label: "Parameters", Value: params, Text: humanCount(params)},
	}
	in := clireport.Field{Key: "input_shape", Label: "Input", Value: nil, Text: "-"}
	fits := clireport.Field{Key: "input_fits", Label: "Fits the graph", Value: nil, Text: "not checked"}
	if p := d.Preprocess; p != nil {
		how := fmt.Sprintf("photo → %s %d×%d", p.Resize, p.Width, p.Height)
		if len(p.Mean) == 3 {
			how += ", " + normName(p.Mean, p.Std)
		}
		if p.Produces != nil {
			in.Value = p.Produces
			how += " → tensor " + dimsText(p.Produces, nil)
		}
		in.Text = how
		if p.SetByModel {
			in.Text = "prepared by the model itself (prompts, several inputs) — `--image photo.jpg` shows it"
		}
		if p.Fits != nil {
			fits.Value = *p.Fits
			fits.Text = "yes"
			if !*p.Fits {
				fits.Text = "NO"
			}
			fits.Text += fmt.Sprintf(" (%s input %q is %s)", p.GraphFile, p.GraphInput, dimsText(p.GraphShape, graphDimNames(d, p)))
		} else if p.NotChecked != "" {
			fits.Text = "not checked: " + skippedText(p.NotChecked)
		}
	}
	fs = append(fs, in, fits)
	var outs []string
	if len(d.ONNX) == 1 && len(man.Files) == 0 {
		for _, o := range d.ONNX[0].Outputs {
			outs = append(outs, o.Name+" "+"["+strings.Join(o.Dims, ",")+"]")
		}
	}
	if outs != nil {
		fs = append(fs, clireport.Field{Key: "outputs", Label: "Outputs", Value: outs, Text: strings.Join(outs, ", ")})
	} else {
		fs = append(fs, clireport.Field{Key: "outputs", Label: "Outputs", Value: []string{}, Text: fmt.Sprintf("see the %d ONNX files below", len(d.ONNX))})
	}
	if rt := d.Runtime; rt != nil {
		fs = append(fs, clireport.Field{Key: "ep_chain", Label: "Runs on", Value: rt.Providers,
			Text: strings.Join(rt.Providers, " → ") + " (first one available on this machine)"})
	} else {
		fs = append(fs, clireport.Field{Key: "ep_chain", Label: "Runs on", Value: []string{}, Text: "-"})
	}
	return fs
}

// graphDimNames finds the dim names of the judged graph input, for display.
func graphDimNames(d inspectDetails, p *preprocessDetails) []string {
	for _, o := range d.ONNX {
		if o.Path != p.GraphFile {
			continue
		}
		for _, in := range o.Inputs {
			if in.Name == p.GraphInput {
				return in.Dims
			}
		}
	}
	return nil
}

func normName(mean, std []float32) string {
	switch {
	case sameFloats(mean, imagenetMean) && sameFloats(std, imagenetStd):
		return "ImageNet mean/std"
	case sameFloats(mean, []float32{0.5, 0.5, 0.5}) && sameFloats(std, []float32{0.5, 0.5, 0.5}):
		return "to [-1, 1]"
	}
	return fmt.Sprintf("mean %s std %s", floatsText(mean), floatsText(std))
}

func modelSections(man *registry.Manifest, d inspectDetails) []clireport.Section {
	md := d.Model
	var secs []clireport.Section

	mrows := []clireport.Field{
		{Label: "Licence", Value: md.License, Text: licenceText(md)},
		{Label: "Directory", Value: md.Dir},
	}
	if md.SourceURL != "" {
		mrows = append(mrows, clireport.Field{Label: "Source", Value: md.SourceURL})
	}
	if md.Labels != "" {
		txt := fmt.Sprintf("%s (%d classes)", md.Labels, md.LabelCount)
		if md.OutputClasses > 0 {
			txt += fmt.Sprintf("; the model outputs %d class scores", md.OutputClasses)
		}
		mrows = append(mrows, clireport.Field{Label: "Labels", Value: txt})
	}
	secs = append(secs, clireport.Section{Title: "Model", Rows: mrows})

	ft := &clireport.Table{Header: []string{"ROLE", "FILE", "SIZE", "SHA256"}}
	for _, f := range d.Files {
		size := "MISSING"
		if f.Exists {
			size = humanSize(f.Bytes)
		}
		ft.Rows = append(ft.Rows, []string{f.Role, f.Path, size, pinText(f)})
	}
	secs = append(secs, clireport.Section{Title: "Files", Table: ft})

	for _, o := range d.ONNX {
		secs = append(secs, onnxSection(o))
	}

	if p := d.Preprocess; p != nil {
		rows := []clireport.Field{
			{Label: "Resize", Value: p.Resize, Text: resizeText(p)},
			{Label: "Normalise", Value: normText(p)},
			{Label: "Layout", Value: p.Layout},
		}
		if p.Produces != nil {
			rows = append(rows, clireport.Field{Label: "Produces", Value: dimsText(p.Produces, nil) + " float32" + variesNote(p.Produces)})
		}
		if e := p.Example; e != nil {
			rows = append(rows, clireport.Field{Label: "Example", Value: fmt.Sprintf(
				"a %d×%d photo becomes %s: x scaled ×%.4g, y ×%.4g, then shifted by +%d,+%d (the model's own code, run now)",
				e.PhotoWidth, e.PhotoHeight, dimsText(e.Tensor, nil), e.ScaleX, e.ScaleY, e.PadX, e.PadY)})
		}
		if p.GraphInput != "" {
			rows = append(rows, clireport.Field{Label: "Graph takes", Value: fmt.Sprintf("%s input %q %s", p.GraphFile, p.GraphInput, dimsText(p.GraphShape, graphDimNames(d, p)))})
		}
		fit := "not checked: " + skippedText(p.NotChecked)
		if p.Fits != nil {
			fit = "yes: every fixed dimension matches"
			if !*p.Fits {
				fit = "NO: a load refuses this model"
			}
		}
		rows = append(rows, clireport.Field{Label: "Fits", Value: fit})
		title := "Preprocessing (what the model's code applies)"
		switch {
		case p.SetByModel:
			title = "Preprocessing (as the manifest declares it; this model builds its inputs itself — see --image)"
		case !p.Resolved:
			title = "Preprocessing (as the manifest declares it)"
		}
		secs = append(secs, clireport.Section{Title: title, Rows: rows})
	}

	if rt := d.Runtime; rt != nil {
		rows := []clireport.Field{{Label: "Providers", Value: strings.Join(rt.Providers, " → ") + " (each session takes the first one this machine has)"}}
		if rt.TensorRT != "n/a" {
			txt := rt.TensorRT
			if txt == "off" {
				txt = "off (opt in with --tensorrt or VISIONSERVE_TENSORRT=1)"
			}
			rows = append(rows, clireport.Field{Label: "TensorRT", Value: txt})
		}
		for _, s := range rt.Sessions {
			th := "ONNX Runtime default threads"
			if s.Threads > 0 {
				th = fmt.Sprintf("%d intra-op thread%s (%s)", s.Threads, plural(s.Threads), s.ThreadsFrom)
			}
			pool := "1 session"
			if s.Pool > 1 {
				pool = fmt.Sprintf("pool of %d sessions", s.Pool)
			}
			rows = append(rows, clireport.Field{Label: "Session " + s.Role, Value: fmt.Sprintf("%s, %s, %s", s.Path, pool, th)})
		}
		idle := "never"
		if rt.IdleUnloadSeconds > 0 {
			idle = fmt.Sprintf("after %d s without requests", rt.IdleUnloadSeconds)
		}
		rows = append(rows, clireport.Field{Label: "Unloads", Value: idle})
		secs = append(secs, clireport.Section{Title: "Runtime", Rows: rows})
	}
	return secs
}

func onnxSection(o onnxDetails) clireport.Section {
	title := "ONNX file " + o.Path
	if o.Error != "" {
		return clireport.Section{Title: title, Rows: []clireport.Field{{Label: "Error", Value: o.Error}}}
	}
	ops := make([]string, 0, len(o.Opsets))
	for _, dom := range sortedKeys64(o.Opsets) {
		ops = append(ops, fmt.Sprintf("%s %d", dom, o.Opsets[dom]))
	}
	producer := strings.TrimSpace(o.Producer + " " + o.ProducerVersion)
	if producer == "" {
		producer = "-"
	}
	weights := humanSize(o.WeightBytes) + " of weights inside the file"
	if len(o.External) > 0 {
		var ext []string
		for _, e := range o.External {
			ext = append(ext, fmt.Sprintf("%s (%d tensors, %s)", e.Location, e.Tensors, humanSize(e.Bytes)))
		}
		weights = humanSize(o.WeightBytes) + " of weights, external data in " + strings.Join(ext, ", ")
	}
	var rows []clireport.Field
	if len(o.Roles) > 0 {
		rows = append(rows, clireport.Field{Label: "Roles", Value: strings.Join(o.Roles, ", ")})
	}
	rows = append(rows, []clireport.Field{
		{Label: "Format", Value: fmt.Sprintf("ONNX IR %d, opset %s", o.IRVersion, orDefault(strings.Join(ops, ", "), "-"))},
		{Label: "Producer", Value: producer},
		{Label: "Graph", Value: fmt.Sprintf("%d nodes, %d weight tensors", o.Nodes, o.Initializers)},
		{Label: "Parameters", Value: paramsText(o)},
		{Label: "Weights", Value: weights},
	}...)
	t := &clireport.Table{Header: []string{"", "NAME", "TYPE", "SHAPE"}}
	dynamic := map[string]bool{}
	add := func(kind string, ts []tensorDetails) {
		for _, v := range ts {
			t.Rows = append(t.Rows, []string{kind, v.Name, v.Dtype, "[" + strings.Join(v.Dims, ",") + "]"})
			for i, d := range v.Shape {
				if d < 0 {
					dynamic[v.Dims[i]] = true
				}
			}
		}
	}
	add("in", o.Inputs)
	add("out", o.Outputs)
	var lines []string
	if len(dynamic) > 0 {
		names := make([]string, 0, len(dynamic))
		for n := range dynamic {
			names = append(names, n)
		}
		sort.Strings(names)
		lines = append(lines, "dynamic dims (any size at run time): "+strings.Join(names, ", "))
	}
	return clireport.Section{Title: title, Rows: rows, Table: t, Lines: lines}
}

func paramsText(o onnxDetails) string {
	if o.Params == 0 {
		return "0"
	}
	s := humanCount(o.Params)
	if len(o.ParamsByType) > 1 || (len(o.ParamsByType) == 1 && o.ParamsByType["float32"] == 0) {
		var parts []string
		for _, k := range sortedKeys64(o.ParamsByType) {
			parts = append(parts, fmt.Sprintf("%s %s", humanCount(o.ParamsByType[k]), k))
		}
		s += " (" + strings.Join(parts, ", ") + ")"
	}
	return s
}

func licenceText(md *modelDetails) string {
	if md.LicenseAllowed {
		return md.License + " — on the permissive allowlist (" + strings.Join(registry.AllowedLicenses(), ", ") + ")"
	}
	return md.License + " — NOT on the permissive allowlist"
}

func pinText(f fileDetails) string {
	switch f.Pin {
	case "ok":
		return "pinned, matches (" + short(f.SHA256) + "…)"
	case "mismatch":
		return "PINNED, DOES NOT MATCH (want " + short(f.PinnedSHA256) + "…, file " + short(f.SHA256) + "…)"
	case "missing":
		return "-"
	}
	return "not pinned"
}

func resizeText(p *preprocessDetails) string {
	s := fmt.Sprintf("%s to %d×%d (%s)", p.Resize, p.Width, p.Height, p.Resample)
	if p.MultipleOf > 0 {
		s += fmt.Sprintf(", sides a multiple of %d", p.MultipleOf)
	}
	switch p.Resize {
	case "squash":
		s += ": the whole photo, aspect ratio not kept"
	case "letterbox":
		s += fmt.Sprintf(": aspect ratio kept, padded with gray %g", p.Pad)
	case "center_crop":
		s += ": short side resized, centre cut out"
	case "keep_aspect":
		s += ": aspect ratio kept, size varies per photo"
	case "top_left_pad":
		s += fmt.Sprintf(": aspect ratio kept, placed at the top-left, the rest filled with gray %g", p.Pad)
	case "long_side", "long_side_pad":
		s += ": the long side scaled to fit, aspect ratio kept"
	case "none":
		s = "none: the photo is fed at its own size"
	}
	return s
}

func normText(p *preprocessDetails) string {
	scale := "pixels/255"
	if !p.Rescale {
		scale = "pixels as 0..255"
	}
	if len(p.Mean) == 0 {
		return scale + ", no mean/std"
	}
	s := fmt.Sprintf("%s, then (x - mean) / std with mean %s, std %s", scale, floatsText(p.Mean), floatsText(p.Std))
	if n := normName(p.Mean, p.Std); !strings.HasPrefix(n, "mean ") {
		s += " (" + n + ")"
	}
	return s
}

func variesNote(d []int64) string {
	for _, v := range d {
		if v < 0 {
			return " (? = depends on the photo's size)"
		}
	}
	return ""
}

func nextSteps(man *registry.Manifest, suggestImage bool) []string {
	run := "visionserve run " + man.Name + " photo.jpg"
	switch api.Task(man.Task) {
	case api.TaskOpenVocab:
		run += ` --prompt "cat. dog."`
	case api.TaskSegmentation:
		run += " --box 10,10,200,200"
	case api.TaskInstanceDetection:
		run += " --template example.jpg"
	}
	steps := []string{run + "   # predict, JSON on stdout (--save draws the result)"}
	if suggestImage {
		steps = append(steps, "visionserve inspect "+man.Name+" --image photo.jpg   # see the exact tensor the model gets")
	}
	return append(steps, "edit "+filepath.Join(man.Dir(), "manifest.yaml")+"   # format: docs/manifest-spec.md")
}

// --- bare .onnx ---

func inspectBareONNX(path string, o inspectOptions) *clireport.Report {
	r := &clireport.Report{Command: "inspect", Subject: filepath.Base(path)}
	d := inspectDetails{Files: []fileDetails{}, ONNX: []onnxDetails{}}
	od := onnxDetails{Path: filepath.Base(path), Roles: []string{}, Opsets: map[string]int64{}, ParamsByType: map[string]int64{},
		External: []externalDetails{}, Inputs: []tensorDetails{}, Outputs: []tensorDetails{}, Metadata: map[string]string{}}
	f, err := engine.ReadFacts(path)
	if err != nil {
		od.Error = err.Error()
		r.Add(clireport.Fail, "%s is not a readable ONNX model: %v", filepath.Base(path), err)
		d.ONNX = append(d.ONNX, od)
		r.Details = d
		r.Summary = []clireport.Field{{Key: "file", Label: "File", Value: path}}
		r.Sections = []clireport.Section{onnxSection(od)}
		r.Decide("")
		return r
	}
	fillONNX(&od, f)
	d.ONNX = append(d.ONNX, od)
	total := f.FileBytes
	d.Files = append(d.Files, fileDetails{Role: "model", Path: filepath.Base(path), Bytes: f.FileBytes, Exists: true, Pin: "not_pinned"})
	for _, e := range f.External {
		d.Files = append(d.Files, fileDetails{Role: "external_data", Path: e.Location, Bytes: e.OnDisk, Exists: e.OnDisk >= 0, Pin: "not_pinned"})
		switch {
		case e.Unsafe:
			r.Add(clireport.Fail, "the model stores weights at %q, outside its folder: ONNX Runtime refuses that path", e.Location)
		case e.OnDisk < 0:
			r.Add(clireport.Fail, "the model keeps %s of weights in %s, which is missing — keep it next to the .onnx file", humanSize(e.Bytes), e.Location)
		default:
			total += e.OnDisk
		}
	}
	if err := registry.ScanAGPLMarkers(path); errors.Is(err, registry.ErrAGPLModel) {
		r.Add(clireport.Fail, "%v", err)
	}
	if dir := filepath.Dir(path); fileExists(filepath.Join(dir, "manifest.yaml")) {
		r.Add(clireport.Info, "%s has a manifest.yaml: `visionserve inspect %s` shows the full model card", dir, dir)
	}
	if o.Image != "" {
		r.Add(clireport.Info, "--image needs a model with a manifest (the manifest says how photos are prepared); import it first")
	}

	task := guessTask(f)
	cmd := importCommand(path, f, task)
	d.ImportCommand = cmd
	r.Add(clireport.Warn, "%s has no manifest: VisionServe cannot serve it until it is imported (command below)", filepath.Base(path))
	r.Details = d

	var ins, outs []string
	for _, v := range od.Inputs {
		ins = append(ins, fmt.Sprintf("%s [%s] %s", v.Name, strings.Join(v.Dims, ","), v.Dtype))
	}
	for _, v := range od.Outputs {
		outs = append(outs, fmt.Sprintf("%s [%s] %s", v.Name, strings.Join(v.Dims, ","), v.Dtype))
	}
	guess := task
	if guess == "" {
		guess = "none of " + strings.Join(importableTasks, "/") + " (import cannot pick a decoder)"
	}
	r.Summary = []clireport.Field{
		{Key: "file", Label: "File", Value: path},
		{Key: "total_bytes", Label: "Size", Value: total, Text: humanSize(total)},
		{Key: "params", Label: "Parameters", Value: f.Params, Text: humanCount(f.Params)},
		{Key: "inputs", Label: "Inputs", Value: ins, Text: strings.Join(ins, "; ")},
		{Key: "outputs", Label: "Outputs", Value: outs, Text: strings.Join(outs, "; ")},
		{Key: "task_guess", Label: "Looks like", Value: task, Text: guess},
	}
	r.Sections = []clireport.Section{onnxSection(od)}
	r.NextSteps = []string{
		cmd,
		"#   --license: the licence of the ORIGINAL model (Apache-2.0, MIT, BSD-3-Clause or BSD-2-Clause) — check it, \"it is on HuggingFace\" says nothing",
		"#   import prints every value it assumed (input size, resize, mean/std) so you can correct it",
	}
	r.Decide("")
	return r
}

var nameSanitizer = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// importCommand is the `visionserve import` command line for a bare .onnx file.
func importCommand(path string, f *engine.Facts, task string) string {
	name := strings.Trim(nameSanitizer.ReplaceAllString(strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)), "-"), "-._")
	if name == "" {
		name = "my-model"
	}
	if task == "" {
		task = "TASK"
	}
	cmd := []string{"visionserve", "import", shellQuote(path), "--name", shellQuote(name), "--task", task, "--license", "LICENSE-ID"}
	if task != "TASK" {
		if _, err := planImport(f, task, importOptions{}); err != nil && strings.Contains(err.Error(), "--input WxH") {
			cmd = append(cmd, "--input", "WxH")
		}
	}
	return strings.Join(cmd, " ")
}
