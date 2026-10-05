package lifecycle

import (
	"fmt"
	"image"
	"path/filepath"
	"slices"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/registry"
)

// InputShapeError is a load error: the manifest's preprocessing produces a tensor that the ONNX
// graph's fixed input dimensions cannot accept. Without this check the model loaded, /api/preprocess
// showed the tensor, and the first /api/predict failed inside ONNX Runtime with
// "index: 2 Got: 640 Expected: 560", naming no manifest field. It is a server-side
// misconfiguration, so it maps to 500 (statusOf's default), not to a caller error.
type InputShapeError struct {
	Model string
	// File is the ONNX file whose input is meant, Input that input's name.
	File, Input string
	// Graph is the input's declared shape (-1 = dynamic); Produced is the shape the preprocessing
	// produces (-1 where it varies with the image size).
	Graph, Produced []int64
	// Width, Height are the manifest's preprocess size (input.width/height or preprocess.size).
	Width, Height int
	// Layout is the manifest's resolved layout ("" = NCHW).
	Layout string
}

func (e *InputShapeError) Error() string {
	layout := e.Layout
	if layout == "" {
		layout = "NCHW"
	}
	var what string
	switch {
	case sameDimsReordered(e.Produced, e.Graph):
		what = "manifest preprocess layout " + layout
	case e.spatialOnly():
		what = fmt.Sprintf("manifest preprocess width×height %d×%d", e.Width, e.Height)
	default:
		what = fmt.Sprintf("manifest preprocessing (width×height %d×%d, layout %s)", e.Width, e.Height, layout)
	}
	note := ""
	if slices.Contains(e.Graph, -1) || slices.Contains(e.Produced, -1) {
		note = "; -1 = dynamic in the graph, or varies with the image"
	}
	return fmt.Sprintf("lifecycle: %q: %s does not match %s input %q %s (the preprocessing produces %s%s) — "+
		"make the manifest's input size and layout match the export, or re-export the model",
		e.Model, what, filepath.Base(e.File), e.Input, dimsString(e.Graph), dimsString(e.Produced), note)
}

// spatialOnly reports whether every disagreeing axis is one the manifest size sets (its produced
// value is the manifest width or height) — the "wrong input size" case.
func (e *InputShapeError) spatialOnly() bool {
	if len(e.Produced) != len(e.Graph) {
		return false
	}
	differs := false
	for i, p := range e.Produced {
		g := e.Graph[i]
		if p < 0 || g < 0 || p == g {
			continue
		}
		if p != int64(e.Width) && p != int64(e.Height) {
			return false
		}
		differs = true
	}
	return differs
}

// sameDimsReordered reports whether a and b hold the same fixed dims in a different order (an
// NCHW tensor for an NHWC graph). Both must be fully fixed and of equal rank.
func sameDimsReordered(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	count := map[int64]int{}
	for i := range a {
		if a[i] < 0 || b[i] < 0 {
			return false
		}
		count[a[i]]++
		count[b[i]]--
	}
	for _, c := range count {
		if c != 0 {
			return false
		}
	}
	for i := range a {
		if a[i] != b[i] {
			return true
		}
	}
	return false
}

func dimsString(d []int64) string {
	s := make([]string, len(d))
	for i, v := range d {
		s[i] = fmt.Sprint(v)
	}
	return "[" + strings.Join(s, ",") + "]"
}

// buildChecked is buildModel plus checkInputShape: what a load builds. The check is not in
// buildModel because /api/preprocess builds the model too, and should keep showing the tensor the
// manifest produces: that is how one sees what the graph is being fed.
func (m *Manager) buildChecked(name string) (models.Base, *registry.Manifest, error) {
	base, man, err := m.buildModel(name)
	if err != nil {
		return nil, nil, err
	}
	if err := checkInputShape(man, base); err != nil {
		return nil, nil, err
	}
	return base, man, nil
}

// shapeProbes are the images the check preprocesses: landscape, portrait and a small square, so
// every dimension a mode derives from the image size (keep_aspect, long_side without pad,
// multiple_of padding, none) takes different values across them and is left unchecked.
var shapeProbes = [][2]int{{640, 360}, {360, 640}, {97, 97}}

// checkInputShape refuses a model whose preprocessing cannot feed its graph (InputShapeError).
// It compares what the model's OWN preprocessing produces — the call serving makes, not a
// re-derivation of the spec — with the input dims the ONNX header declares: a dim that is dynamic
// in the graph, or that varies with the image size, is not judged; every other one must be equal.
// It reads only the file header and preprocesses three small images, and creates no session.
//
// The tensor must feed an input unambiguously, otherwise nothing is checked:
//   - a plain Model: the input it names (InputName), else the graph's only input — the engine
//     binds every graph input and Predict passes one tensor, so a several-input graph without
//     InputName is not a configuration this check can judge;
//   - a PipelineModel: only the explain role, where ExplainPreprocess's tensor is the sole input
//     of the role's session (explain binds all its inputs and passes one tensor). Every other
//     role is fed by the model's Infer in ways only it knows (prompts, tokens, chained outputs),
//     so it is not guessed here; the SAM family and PaddleOCR have export-fixed preprocessing anyway.
//
// A file the header reader cannot parse, or a probe the model refuses to preprocess, is not
// judged: creating the session (or the first request) reports those on its own.
func checkInputShape(man *registry.Manifest, base models.Base) error {
	_, err := judgeInputShape(man, base)
	return err
}

// judgeInputShape is checkInputShape; skipped says why nothing was judged ("" = it was), for the
// test that runs it over every shipped manifest.
func judgeInputShape(man *registry.Manifest, base models.Base) (skipped string, err error) {
	fit := JudgeInputShape(man, base)
	return fit.Skipped, fit.Err
}

// InputFit is the input-shape check's verdict together with what it compared, for tools that show
// it (`visionserve inspect`, `visionserve import`). Load refuses exactly when Err is set.
type InputFit struct {
	// File and Input are the ONNX file and graph input judged ("" when skipped before one was
	// chosen). Graph is that input's declared dims (-1 = dynamic); Produced is the shape the
	// model's preprocessing produces (-1 = varies with the image size; nil when not probed).
	File, Input     string
	Graph, Produced []int64
	// Skipped says why nothing was judged ("" = judged); Err is the *InputShapeError of a
	// preprocessing that cannot feed the graph.
	Skipped string
	Err     error
}

// JudgeInputShape runs the load-time input-shape check (checkInputShape) on a built model and
// reports what it compared. It reads the file header and preprocesses three small images; it
// creates no session.
func JudgeInputShape(man *registry.Manifest, base models.Base) InputFit {
	var (
		file, input string
		pre         func(image.Image) (engine.Tensor, error)
	)
	switch mdl := base.(type) {
	case models.Model:
		file, input = man.ModelFilePath(), mdl.InputName()
		pre = func(img image.Image) (engine.Tensor, error) { t, _, err := mdl.Preprocess(img); return t, err }
	case models.PipelineModel:
		ep, ok := mdl.(models.ExplainPreprocessor)
		if !ok || man.Explain == nil || man.Explain.Role == "" {
			return InputFit{Skipped: "pipeline model without an explain role"}
		}
		path, ok := man.FilesAbs()[man.Explain.Role]
		if !ok {
			return InputFit{Skipped: "explain role not in files"} // reported by the explain session itself
		}
		file = path
		pre = func(img image.Image) (engine.Tensor, error) { t, _, err := ep.ExplainPreprocess(img); return t, err }
	default:
		return InputFit{Skipped: "neither Model nor PipelineModel"}
	}

	ins, _, err := engine.InspectHeader(file)
	if err != nil || len(ins) == 0 {
		return InputFit{File: file, Skipped: "header unreadable or no inputs"}
	}
	var graph *engine.IOInfo
	switch {
	case input != "":
		for i := range ins {
			if ins[i].Name == input {
				graph = &ins[i]
			}
		}
	case len(ins) == 1:
		graph = &ins[0]
	}
	if graph == nil || graph.Shape == nil {
		// An ambiguous binding, an input name the graph lacks (the session reports it), or unknown rank.
		return InputFit{File: file, Skipped: fmt.Sprintf("no unambiguous input (InputName %q, %d graph inputs)", input, len(ins))}
	}

	fit := InputFit{File: file, Input: graph.Name, Graph: graph.Shape}
	produced, ok := probeShape(pre)
	if !ok {
		fit.Skipped = "the probes could not be preprocessed"
		return fit
	}
	fit.Produced = produced
	if !shapeFits(produced, graph.Shape) {
		spec, _ := man.PreprocessSpec()
		fit.Err = &InputShapeError{
			Model: man.Name, File: file, Input: graph.Name,
			Graph: graph.Shape, Produced: produced,
			Width: spec.Width, Height: spec.Height, Layout: string(spec.Layout),
		}
	}
	return fit
}

// probeShape preprocesses every shapeProbes image and returns the tensor shape, with -1 on each
// axis whose size differs between probes. ok is false when a probe fails or the rank changes.
func probeShape(pre func(image.Image) (engine.Tensor, error)) (shape []int64, ok bool) {
	for i, wh := range shapeProbes {
		img := image.NewNRGBA(image.Rect(0, 0, wh[0], wh[1])) // flat mid-gray
		for p := 0; p < len(img.Pix); p += 4 {
			img.Pix[p], img.Pix[p+1], img.Pix[p+2], img.Pix[p+3] = 128, 128, 128, 255
		}
		t, err := pre(img)
		if err != nil || len(t.Shape) == 0 {
			return nil, false
		}
		if i == 0 {
			shape = append([]int64(nil), t.Shape...)
			continue
		}
		if len(t.Shape) != len(shape) {
			return nil, false
		}
		for a, d := range t.Shape {
			if shape[a] != d {
				shape[a] = -1
			}
		}
	}
	return shape, true
}

// shapeFits reports whether a tensor of shape produced (-1 = varies) can feed an input declared
// as graph (-1 = dynamic): same rank, and every axis fixed on both sides equal.
func shapeFits(produced, graph []int64) bool {
	if len(produced) != len(graph) {
		return false
	}
	for i := range produced {
		if produced[i] >= 0 && graph[i] >= 0 && produced[i] != graph[i] {
			return false
		}
	}
	return true
}
