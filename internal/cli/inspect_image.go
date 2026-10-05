package cli

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"sort"
	"strings"

	"visionserve/internal/cli/clireport"
	"visionserve/internal/engine"
	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/internal/server"
	"visionserve/internal/vision/preprocess"
	"visionserve/pkg/api"
)

// --- --image: the real preprocessing, offline ---

// previewInput runs the model's preprocessing on the --image photo through the same code path as
// POST /api/preprocess (lifecycle.Manager.Preprocess) and returns the tensor facts and a section
// with the tensor turned back into a picture (also written as a PNG).
func previewInput(mgr *lifecycle.Manager, man *registry.Manifest, base models.Base, o inspectOptions) (*imageDetails, clireport.Section, error) {
	var sec clireport.Section
	img, err := loadImage(o.Image)
	if err != nil {
		return nil, sec, err
	}
	b := img.Bounds()
	text := o.Prompt
	if text == "" && api.Task(man.Task) == api.TaskOpenVocab {
		text = "object."
	}
	req := server.Request{PredictJSONRequest: api.PredictJSONRequest{Model: man.Name, Prompt: text}}
	prompt, err := req.ToPrompt(b.Dx(), b.Dy())
	if err != nil {
		return nil, sec, err
	}
	res, err := mgr.Preprocess(context.Background(), man.Name, img, prompt)
	if err != nil {
		return nil, sec, err
	}
	id := &imageDetails{Path: o.Image, Width: b.Dx(), Height: b.Dy()}
	for _, nt := range res.Inputs {
		dt := "float32"
		if nt.Tensor.Dtype == "i64" {
			dt = "int64"
		}
		id.Inputs = append(id.Inputs, fmt.Sprintf("%s/%s %s %s", nt.Role, nt.Name, dimsText(nt.Tensor.Shape, nil), dt))
	}
	pick := pickImageTensor(res.Inputs)
	if pick < 0 {
		return nil, sec, fmt.Errorf("the model's first session gets no image tensor (%s)", strings.Join(id.Inputs, "; "))
	}
	nt := res.Inputs[pick]
	t := nt.Tensor
	id.Input, id.Role, id.Shape, id.Dtype = nt.Name, nt.Role, t.Shape, "float32"
	id.Min, id.Max, id.Mean = tensorStats(t)
	if m := res.Meta; m != nil {
		id.Meta = &metaJSON{OrigWidth: m.OrigWidth, OrigHeight: m.OrigHeight, ScaleX: m.ScaleX, ScaleY: m.ScaleY, PadX: m.PadX, PadY: m.PadY}
	}

	// Undo the normalisation exactly when the model says which spec made the tensor
	// (models.PreprocessReporter); any other tensor is min-max scaled for viewing.
	var spec *preprocess.Spec
	if rep, ok := base.(models.PreprocessReporter); ok {
		if s, err := rep.ResolvedPreprocess(); err == nil {
			spec = &s
		}
	}
	pic, how := tensorPicture(t, spec)
	id.Inverted = how
	outPath := o.ImageOut
	if outPath == "" {
		outPath = man.Name + "-input.png"
	}
	var pngBuf bytes.Buffer
	if pic != nil {
		if err := png.Encode(&pngBuf, pic); err == nil {
			if err := os.WriteFile(outPath, pngBuf.Bytes(), 0o644); err != nil {
				return nil, sec, fmt.Errorf("write %s: %w", outPath, err)
			}
			id.Picture = outPath
		}
	}

	rows := []clireport.Field{
		{Label: "Photo", Value: fmt.Sprintf("%s, %d×%d", o.Image, b.Dx(), b.Dy())},
		{Label: "Tensor", Value: fmt.Sprintf("%s/%s %s float32", nt.Role, nt.Name, dimsText(t.Shape, nil))},
		{Label: "Values", Value: fmt.Sprintf("min %.4g, max %.4g, mean per channel %s", id.Min, id.Max, floats64Text(id.Mean))},
	}
	if id.Meta != nil {
		rows = append(rows, clireport.Field{Label: "Mapping", Value: fmt.Sprintf(
			"tensor_x = photo_x × %.4g + %d, tensor_y = photo_y × %.4g + %d (how boxes map back)",
			id.Meta.ScaleX, id.Meta.PadX, id.Meta.ScaleY, id.Meta.PadY)})
	}
	if len(res.Inputs) > 1 {
		rows = append(rows, clireport.Field{Label: "All inputs", Value: strings.Join(id.Inputs, "; ")})
	}
	if id.Picture != "" {
		rows = append(rows, clireport.Field{Label: "Picture", Value: id.Picture + " (" + how + ")"})
	}
	sec = clireport.Section{Title: "Input tensor for " + o.Image, Rows: rows}
	if pngBuf.Len() > 0 {
		sec.Image = &clireport.Image{Caption: "what the model sees: the tensor turned back into a picture (" + how + ")", PNG: pngBuf.Bytes()}
	}
	return id, sec, nil
}

// pickImageTensor returns the index of the first float32 tensor shaped like an image (rank 3 or
// 4 with a 3-channel axis), or -1.
func pickImageTensor(ins []lifecycle.NamedTensor) int {
	for i, nt := range ins {
		if nt.Tensor.Dtype == "i64" {
			continue
		}
		if _, _, _, ok := imageAxes(nt.Tensor.Shape); ok {
			return i
		}
	}
	return -1
}

// imageAxes finds the channel, height and width axes of an image tensor: NCHW, NHWC, CHW or HWC.
func imageAxes(s []int64) (c, h, w int, ok bool) {
	switch {
	case len(s) == 4 && s[0] == 1 && s[1] == 3:
		return 1, 2, 3, true
	case len(s) == 4 && s[0] == 1 && s[3] == 3:
		return 3, 1, 2, true
	case len(s) == 3 && s[0] == 3:
		return 0, 1, 2, true
	case len(s) == 3 && s[2] == 3:
		return 2, 0, 1, true
	}
	return 0, 0, 0, false
}

// tensorStats returns the min, max and per-channel mean of an image tensor.
func tensorStats(t engine.Tensor) (lo, hi float64, mean []float64) {
	c, _, _, ok := imageAxes(t.Shape)
	if !ok || len(t.Data) == 0 {
		return 0, 0, nil
	}
	lo, hi = math.Inf(1), math.Inf(-1)
	sums := make([]float64, 3)
	counts := make([]int, 3)
	stride := 1
	for i := c + 1; i < len(t.Shape); i++ {
		stride *= int(t.Shape[i])
	}
	for i, v := range t.Data {
		f := float64(v)
		lo, hi = math.Min(lo, f), math.Max(hi, f)
		ch := (i / stride) % 3
		sums[ch] += f
		counts[ch]++
	}
	mean = make([]float64, 3)
	for i := range mean {
		if counts[i] > 0 {
			mean[i] = sums[i] / float64(counts[i])
		}
	}
	return lo, hi, mean
}

// tensorPicture turns an image tensor back into a picture: by inverting spec's normalisation when
// given, else by min-max scaling all values to 0..255. how says which.
func tensorPicture(t engine.Tensor, spec *preprocess.Spec) (image.Image, string) {
	c, h, w, ok := imageAxes(t.Shape)
	if !ok {
		return nil, ""
	}
	H, W := int(t.Shape[h]), int(t.Shape[w])
	if H <= 0 || W <= 0 || len(t.Data) != 3*H*W {
		return nil, ""
	}
	stride := make([]int, len(t.Shape))
	st := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		stride[i] = st
		st *= int(t.Shape[i])
	}
	at := func(ch, y, x int) float64 { return float64(t.Data[ch*stride[c]+y*stride[h]+x*stride[w]]) }

	pixel := func(ch int, v float64) float64 { return v }
	how := "min-max scaled to 0..255"
	if spec != nil {
		mean, std := spec.Mean, spec.Std
		rescale := !spec.NoRescale
		pixel = func(ch int, v float64) float64 {
			m, s := 0.0, 1.0
			if ch < len(mean) {
				m = float64(mean[ch])
			}
			if ch < len(std) && std[ch] != 0 {
				s = float64(std[ch])
			}
			if rescale {
				return (v*s + m) * 255
			}
			return v*s + m
		}
		how = "normalisation undone"
	} else {
		lo, hi, _ := tensorStats(t)
		if hi > lo {
			pixel = func(_ int, v float64) float64 { return (v - lo) / (hi - lo) * 255 }
		}
	}
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			var px [3]uint8
			for ch := 0; ch < 3; ch++ {
				v := math.Round(pixel(ch, at(ch, y, x)))
				px[ch] = uint8(math.Max(0, math.Min(255, v)))
			}
			img.SetNRGBA(x, y, color.NRGBA{px[0], px[1], px[2], 255})
		}
	}
	return img, how
}

// --- small formatting helpers ---

func humanSize(n int64) string {
	if n < 0 {
		return "-"
	}
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

func humanCount(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.2f B", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.2f M", float64(n)/1e6)
	case n >= 1e3:
		return fmt.Sprintf("%.1f k", float64(n)/1e3)
	}
	return fmt.Sprint(n)
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

func short(d string) string {
	if len(d) > 12 {
		return d[:12]
	}
	return d
}

func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

func floatsText(v []float32) string {
	parts := make([]string, len(v))
	for i, f := range v {
		parts[i] = fmt.Sprintf("%g", f)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func floats64Text(v []float64) string {
	parts := make([]string, len(v))
	for i, f := range v {
		parts[i] = fmt.Sprintf("%.4g", f)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func sameFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-6 {
			return false
		}
	}
	return true
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedKeys64(m map[string]int64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func fileExists(p string) bool { st, err := os.Stat(p); return err == nil && st.Mode().IsRegular() }
