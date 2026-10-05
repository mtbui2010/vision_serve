package cli

import (
	"fmt"
	"strings"

	"visionserve/internal/engine"
	"visionserve/internal/models/detr"
)

// importOptions are the choices `visionserve import` takes from its flags; the zero value means
// "read it from the graph, or use the default and say so".
type importOptions struct {
	Width, Height int    // --input WxH (0 = from the graph)
	Layout        string // --layout NCHW|NHWC ("" = from the graph)
	Resize        string // --resize ("" = squash)
	Mean, Std     []float32
}

// importPlan is what import decided from an ONNX header for one task: the architecture that
// decodes the outputs, the input geometry, and every fact read and assumption made (each is
// shown to the user).
type importPlan struct {
	Task, Arch, PostType string
	Width, Height        int
	Layout               string
	Resize               string
	Mean, Std            []float32
	InputName            string
	// Classes is the number of class scores the decoder reads (-1 = not a class output, or
	// dynamic): what a labels file must match.
	Classes int
	Read    []string // facts read from the graph
	Assumed []string // defaults taken without evidence: the user must confirm them
}

// importableTasks are the tasks import can build a manifest for from the header alone: each has a
// generic decoder whose output contract the header can check. Every other task needs code that
// knows the model (prompts, several sessions, tokenizers).
var importableTasks = []string{"classification", "detection", "depth"}

var (
	imagenetMean = []float32{0.485, 0.456, 0.406}
	imagenetStd  = []float32{0.229, 0.224, 0.225}
)

// refusal is a reason import cannot build a working manifest. It is a FAIL verdict, not a usage
// error: the flags were well-formed, the model does not fit.
type refusal struct{ msg string }

func (r *refusal) Error() string { return r.msg }

func refuse(format string, args ...any) error { return &refusal{fmt.Sprintf(format, args...)} }

// planImport reads the input geometry and checks the outputs against the decoder of task. A
// model this cannot serve correctly is refused with the reason, never guessed.
func planImport(f *engine.Facts, task string, o importOptions) (importPlan, error) {
	p := importPlan{Task: task, Classes: -1}
	known := false
	for _, t := range importableTasks {
		known = known || t == task
	}
	if !known {
		return p, refuse("import cannot build a %q model from the ONNX header alone: it handles %s (one image in, "+
			"a decoder VisionServe has). Segmentation, open-vocabulary, embedding and grasp models need code that "+
			"knows the model — use `visionserve convert`, or write the manifest by hand (docs/manifest-spec.md)",
			task, strings.Join(importableTasks, ", "))
	}

	// The input: one float32 image tensor, NCHW or NHWC.
	if len(f.Inputs) != 1 {
		names := make([]string, len(f.Inputs))
		for i, in := range f.Inputs {
			names[i] = in.Name
		}
		return p, refuse("the graph has %d inputs (%s); import handles models with one image input — a model "+
			"that also takes prompts, sizes or masks needs its own architecture (`visionserve convert`)",
			len(f.Inputs), strings.Join(names, ", "))
	}
	in := f.Inputs[0]
	p.InputName = in.Name
	if in.ElemType != 1 {
		return p, refuse("input %q is %s; VisionServe feeds images as float32 — re-export the model with a float32 input",
			in.Name, orUnknown(engine.ElemTypeName(in.ElemType)))
	}
	if len(in.Shape) != 4 {
		return p, refuse("input %q has shape %s; an image model takes a 4-D batch: [1,3,H,W] (NCHW) or [1,H,W,3] (NHWC)",
			in.Name, dimsText(in.Shape, in.DimNames))
	}
	if b := in.Shape[0]; b > 1 {
		return p, refuse("input %q has batch size %d; VisionServe sends one image at a time — re-export with batch 1 or a dynamic batch",
			in.Name, b)
	}
	c1, c3 := in.Shape[1], in.Shape[3]
	switch strings.ToUpper(o.Layout) {
	case "NCHW", "NHWC":
		p.Layout = strings.ToUpper(o.Layout)
		axis, c := 1, c1
		if p.Layout == "NHWC" {
			axis, c = 3, c3
		}
		if c >= 0 && c != 3 {
			return p, refuse("--layout %s puts the colour channels at axis %d, but input %q %s has %d there",
				p.Layout, axis, in.Name, dimsText(in.Shape, in.DimNames), c)
		}
	case "":
		switch {
		case c1 == 3 && c3 != 3:
			p.Layout = "NCHW"
			p.Read = append(p.Read, fmt.Sprintf("layout NCHW: input %q %s has its 3 colour channels at axis 1", in.Name, dimsText(in.Shape, in.DimNames)))
		case c3 == 3 && c1 != 3:
			p.Layout = "NHWC"
			p.Read = append(p.Read, fmt.Sprintf("layout NHWC: input %q %s has its 3 colour channels last", in.Name, dimsText(in.Shape, in.DimNames)))
		case c1 < 0 && c3 < 0, c1 == 3 && c3 == 3:
			p.Layout = "NCHW"
			p.Assumed = append(p.Assumed, fmt.Sprintf("layout NCHW: the graph does not say where the channels are (%s) — pass --layout NHWC if they are last",
				dimsText(in.Shape, in.DimNames)))
		default:
			return p, refuse("input %q %s has no axis of 3 colour channels; VisionServe feeds RGB images (3 channels)",
				in.Name, dimsText(in.Shape, in.DimNames))
		}
	default:
		return p, refuse("--layout %q is invalid (NCHW or NHWC)", o.Layout)
	}
	h, w := in.Shape[2], in.Shape[3]
	if p.Layout == "NHWC" {
		h, w = in.Shape[1], in.Shape[2]
	}
	switch {
	case o.Width > 0:
		p.Width, p.Height = o.Width, o.Height
	case h > 0 && w > 0:
		p.Width, p.Height = int(w), int(h)
		p.Read = append(p.Read, fmt.Sprintf("input size %d×%d (width×height) from the graph's fixed dims", w, h))
	default:
		return p, refuse("the graph accepts any input size (input %q is %s); pass --input WxH: the resolution the model was trained at",
			in.Name, dimsText(in.Shape, in.DimNames))
	}

	p.Resize = o.Resize
	if p.Resize == "" {
		p.Resize = "squash"
		p.Assumed = append(p.Assumed, fmt.Sprintf("resize: squash the photo to %d×%d without keeping its aspect ratio (the usual training resize) — "+
			"pass --resize letterbox if the model was trained on padded images", p.Width, p.Height))
	}
	p.Mean, p.Std = o.Mean, o.Std
	if p.Mean == nil {
		p.Mean, p.Std = imagenetMean, imagenetStd
		p.Assumed = append(p.Assumed, "normalisation: pixels/255, then ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] "+
			"(torchvision / timm default) — pass --mean and --std if the model was trained otherwise (e.g. 0.5,0.5,0.5)")
	}

	// The outputs, against the task's decoder.
	outs := f.Outputs
	switch task {
	case "classification":
		p.Arch, p.PostType = "efficientnet", "classification"
		o0 := outs[0]
		if !(len(o0.Shape) == 2 && o0.Shape[0] <= 1) && len(o0.Shape) != 1 {
			return p, refuse("the first output %q is %s; a classification model must output one row of class scores, [1, classes]",
				o0.Name, dimsText(o0.Shape, o0.DimNames))
		}
		if c := o0.Shape[len(o0.Shape)-1]; c > 0 {
			p.Classes = int(c)
		}
		p.Read = append(p.Read, fmt.Sprintf("output %q %s: %s class scores", o0.Name, dimsText(o0.Shape, o0.DimNames), countText(p.Classes)))
		if len(outs) > 1 {
			p.Assumed = append(p.Assumed, fmt.Sprintf("only the first output (%q) is read; the other %d are ignored", o0.Name, len(outs)-1))
		}
		p.Assumed = append(p.Assumed, "scores: the output is raw logits and softmax turns them into probabilities; the top 5 classes are returned")
	case "detection":
		p.Arch, p.PostType = "rf-detr", "detr"
		ts := make([]engine.Tensor, len(outs))
		for i, o := range outs {
			ts[i] = engine.Tensor{Shape: o.Shape}
		}
		split, err := detr.SplitOutputs(ts, 0, 0)
		if err != nil || len(outs) < 2 || split.Logits.Shape == nil {
			return p, refuse("the outputs (%s) are not a DETR-style set of queries ([1,Q,4] boxes + [1,Q,classes] scores), the only "+
				"detection decoder import can pick from the header. YOLO-style outputs ([1,84,8400]) need NMS and a different "+
				"decoder — use `visionserve convert`", outputsText(outs))
		}
		b, l := split.Boxes.Shape, split.Logits.Shape
		if len(b) != 3 || len(l) != 3 || !sameDim(b[1], l[1]) || b[0] > 1 || l[0] > 1 {
			return p, refuse("the boxes %s and class scores %s are not [1,Q,4] and [1,Q,classes] over the same Q queries; "+
				"import cannot decode them — use `visionserve convert`", dimsText(b, nil), dimsText(l, nil))
		}
		if c := l[2]; c > 0 {
			p.Classes = int(c)
		}
		p.Read = append(p.Read, fmt.Sprintf("outputs: boxes %s and class scores %s (%s queries, %s classes)",
			dimsText(b, nil), dimsText(l, nil), countText(int(maxInt64(b[1], -1))), countText(p.Classes)))
		p.Assumed = append(p.Assumed, "decoder: DETR-style, NMS-free (RF-DETR / RT-DETR): a sigmoid score per class and query, boxes as "+
			"cx,cy,w,h normalised to 0..1 of the input, score threshold 0.5. A softmax DETR (facebook/detr, with a 'no object' class) "+
			"or a YOLO export decodes WRONGLY with it — convert those with `visionserve convert`")
	case "depth":
		p.Arch, p.PostType = "midas", "depth"
		o0 := outs[0]
		ok := len(o0.Shape) == 2 || (len(o0.Shape) == 3 && o0.Shape[0] <= 1)
		if !ok {
			return p, refuse("the first output %q is %s; a depth model must output one map, [1,H,W] or [H,W]",
				o0.Name, dimsText(o0.Shape, o0.DimNames))
		}
		p.Read = append(p.Read, fmt.Sprintf("output %q %s: one depth map", o0.Name, dimsText(o0.Shape, o0.DimNames)))
		p.Assumed = append(p.Assumed, "depth: the map is relative (inverse) depth, min-max normalised to 0..1 per image (MiDaS / Depth Anything style)")
	}
	return p, nil
}

func sameDim(a, b int64) bool { return a < 0 || b < 0 || a == b }

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func countText(n int) string {
	if n < 0 {
		return "a dynamic number of"
	}
	return fmt.Sprint(n)
}

func orUnknown(s string) string {
	if s == "" {
		return "not a tensor"
	}
	return s
}

func outputsText(outs []engine.IOFacts) string {
	parts := make([]string, len(outs))
	for i, o := range outs {
		parts[i] = o.Name + " " + dimsText(o.Shape, o.DimNames)
	}
	return strings.Join(parts, ", ")
}

// dimsText renders a shape with its dynamic dims named: [batch,3,224,224]; an unnamed dynamic dim
// is "?". A nil shape (scalar or unknown rank) is "[]".
func dimsText(shape []int64, names []string) string {
	parts := make([]string, len(shape))
	for i, d := range shape {
		switch {
		case d >= 0:
			parts[i] = fmt.Sprint(d)
		case i < len(names) && names[i] != "":
			parts[i] = names[i]
		default:
			parts[i] = "?"
		}
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// guessTask suggests a task from the outputs alone, for the import command line inspect prints
// for a bare .onnx: "" when no importable decoder fits.
func guessTask(f *engine.Facts) string {
	for _, task := range []string{"detection", "classification", "depth"} {
		if _, err := planImport(f, task, importOptions{Width: 1, Height: 1}); err == nil {
			return task
		}
	}
	return ""
}
