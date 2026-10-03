package pipeline

import (
	"fmt"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/internal/models/detr"
	"visionserve/internal/vision/util"
)

// Closed is the closed-set Detector stage: a plain Model (RF-DETR, RT-DETR, …) driven through
// the session under Role — the model's own Preprocess builds the input tensor and the mapping
// back to ORIGINAL image coordinates, the Runner executes the session, the model's own
// Postprocess decodes the detections (NMS-free for DETRs).
type Closed struct {
	Role  string
	Model models.Model
	// DETR marks an RF-DETR export: its outputs are identified with detr.SplitOutputs (Labels =
	// the manifest's label count), Postprocess is handed exactly [boxes, logits], and the forward
	// pass keeps the boxes and the query features for a head that scores the same queries.
	// Otherwise Postprocess gets every output, as the model's own lifecycle path does.
	DETR   bool
	Labels int
	// Prefix starts every error message ("hybrid", "grasp").
	Prefix string
}

// ClosedPass is one forward pass of a Closed detector: its decoded detections, and for a DETR
// export the raw boxes and query features (zero Tensor when the export has none) with the
// preprocess mapping, so a head can score the SAME queries without a second image pass.
type ClosedPass struct {
	Dets  []models.Detection
	Boxes engine.Tensor
	Feats engine.Tensor
	Meta  models.PreprocessMeta
}

// Forward runs the detector once.
func (d *Closed) Forward(c Call) (*ClosedPass, error) {
	in, meta, err := d.Model.Preprocess(c.Img)
	if err != nil {
		return nil, fmt.Errorf("%s: detector preprocess: %w", d.Prefix, err)
	}
	inName := util.FirstName(c.Runner.InputNames(d.Role), d.Model.InputName())
	if inName == "" {
		return nil, fmt.Errorf("%s: detector session %q has no input name", d.Prefix, d.Role)
	}
	outs, err := c.Runner.Run(d.Role, map[string]engine.Tensor{inName: in})
	if err != nil {
		return nil, fmt.Errorf("%s: detector inference: %w", d.Prefix, err)
	}
	p := &ClosedPass{Meta: meta}
	if d.DETR {
		o, err := detr.SplitOutputs(outs, d.Labels, 0)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", d.Prefix, err)
		}
		if o.Logits.Data == nil {
			return nil, fmt.Errorf("%s: detector session %q emitted no class logits (outputs %v)", d.Prefix, d.Role, util.ShapesOf(outs))
		}
		p.Boxes, p.Feats = o.Boxes, o.Feats
		outs = []engine.Tensor{o.Boxes, o.Logits}
	}
	res, err := d.Model.Postprocess(outs, meta)
	if err != nil {
		return nil, fmt.Errorf("%s: detector postprocess: %w", d.Prefix, err)
	}
	p.Dets = res.Detections
	return p, nil
}

// Detect implements Detector: every detection when words is empty (a closed-set detector's own
// vocabulary IS the prompt then), otherwise only those whose class is one of words.
func (d *Closed) Detect(c Call, words []string) ([]models.Detection, error) {
	p, err := d.Forward(c)
	if err != nil {
		return nil, err
	}
	if len(words) == 0 {
		return p.Dets, nil
	}
	return FilterByClass(p.Dets, words), nil
}
