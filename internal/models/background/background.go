// Package background implements "background" — class-agnostic support-surface
// (table / floor) segmentation. It returns a SINGLE background mask (the support
// surface) using one of several interchangeable methods, selected per request via
// the `method` field:
//
//   - "auto"     (default): "depth" when it finds a clear plane, otherwise "cv".
//   - "depth":   MiDaS depth → fit the dominant plane (affine disparity) →
//     the near-plane region IS the support surface. Fastest (~tens of ms)
//     and the most accurate for tabletop scenes (objects rise above the plane).
//   - "sam":     MobileSAM prompted at six lower-frame seed points (a 3×2 grid, one
//     point per prompt, encoder run once), validated by area + border touch.
//   - "cv":      classical CV (no inference) — the large, low-texture region grown
//     from the image border/bottom. Fastest; least robust.
//   - "automask": MobileSAM Automatic Mask Generator, then KEEP the large / border-
//     touching masks (the support surfaces) and union them. Slow (N² decoder
//     calls) but method-of-record / fallback.
//
// Foreground (objects) is just the complement of the returned background mask; callers
// invert if they need it.
//
// Like Grounded-SAM / grasp, this model only ORCHESTRATES sessions owned by
// lifecycle.Manager (VRAM-safe): MobileSAM encoder/decoder (roles encoder/decoder) and
// MiDaS (role depth). The "cv" method uses no session.
package background

import (
	"fmt"
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/models"
	"visionserve/internal/models/mobilesam"
	"visionserve/internal/vision/mask"
	"visionserve/internal/vision/util"
)

func init() {
	models.Register("background", New)
}

const (
	roleEncoder = "encoder"
	roleDecoder = "decoder"
	roleDepth   = "depth"
)

// Method names (prompt.Method). Empty => defaultMethod.
const (
	methodAuto = "auto" // depth → cv fallback (robust default: depth's accuracy when a
	//                         clear plane exists, cv's reliability on monocular scenes)
	methodDepth    = "depth"
	methodSAM      = "sam"
	methodCV       = "cv"
	methodAutomask = "automask"
	defaultMethod  = methodAuto
)

// Background-mask area/border heuristics (percent of image area), shared by the methods
// that classify masks (automask, sam validation). A mask counts as a support surface
// when its area ≥ bgMaxAreaPct, or it touches the image border AND covers ≥ borderMinAreaPct.
const (
	defaultBgMaxAreaPct = 50.0
	borderMinAreaPct    = 5.0
	foregroundGridSize  = 8 // automask grid default (N×N); per-request override via grid_size
)

// ImageNet normalization MiDaS was trained with (see models/midas/manifest.yaml).
var (
	imagenetMean = []float32{0.485, 0.456, 0.406}
	imagenetStd  = []float32{0.229, 0.224, 0.225}
)

// maskSegmenter is the MobileSAM seam used by the sam / automask methods: automask or
// box/point-prompted segmentation returning raw bitmaps at original-image resolution.
type maskSegmenter interface {
	InferMasks(img image.Image, prompt models.Prompt, r models.Runner) ([]models.Mask, []mobilesam.MaskBitmap, error)
}

type backgroundModel struct {
	cfg      models.Config
	sam      models.PipelineModel // MobileSAM (sam / automask methods)
	seg      maskSegmenter
	hasSAM   bool
	hasDepth bool
}

// New builds the background model. MobileSAM (encoder/decoder) is required for the
// sam/automask methods; the MiDaS session (role depth) is required for the depth method.
// The cv method needs neither. A manifest may omit sessions it does not need.
func New(cfg models.Config) (models.Base, error) {
	m := &backgroundModel{cfg: cfg}

	if cfg.Files[roleEncoder] != "" && cfg.Files[roleDecoder] != "" {
		samBase, err := mobilesam.New(cfg)
		if err != nil {
			return nil, fmt.Errorf("background: segmenter: %w", err)
		}
		sam, ok := samBase.(models.PipelineModel)
		if !ok {
			return nil, fmt.Errorf("background: segmenter is not a PipelineModel")
		}
		seg, ok := samBase.(maskSegmenter)
		if !ok {
			return nil, fmt.Errorf("background: segmenter does not expose InferMasks")
		}
		if gs, ok := samBase.(interface{ SetAutoGridSize(int) }); ok {
			gs.SetAutoGridSize(foregroundGridSize)
		}
		m.sam, m.seg, m.hasSAM = sam, seg, true
	}
	m.hasDepth = cfg.Files[roleDepth] != ""

	if !m.hasSAM && !m.hasDepth {
		return nil, fmt.Errorf("background: manifest must declare MobileSAM (files.encoder/decoder) and/or MiDaS (files.depth)")
	}
	return m, nil
}

func (m *backgroundModel) Name() string      { return m.cfg.Name }
func (m *backgroundModel) Task() models.Task { return models.TaskSegmentation }

// Roles lists every session this model may use; lifecycle loads them all up front.
func (m *backgroundModel) Roles() []string {
	var roles []string
	if m.hasSAM {
		roles = append(roles, m.sam.Roles()...)
	}
	if m.hasDepth {
		roles = append(roles, roleDepth)
	}
	return roles
}

func (m *backgroundModel) PoolSizes() map[string]int {
	if m.hasSAM {
		if ps, ok := m.sam.(models.PoolSizer); ok {
			return ps.PoolSizes()
		}
	}
	return nil
}

// resolveMethod picks the method from the request (defaulting), validating availability.
func (m *backgroundModel) resolveMethod(prompt models.Prompt) (string, error) {
	method := prompt.Method
	if method == "" {
		method = defaultMethod
	}
	switch method {
	case methodAuto:
		// always available: tries depth (if a MiDaS session exists) then falls back to cv.
	case methodDepth:
		if !m.hasDepth {
			return "", fmt.Errorf("background: method=depth needs a MiDaS session (files.depth) — not declared in the manifest")
		}
	case methodSAM, methodAutomask:
		if !m.hasSAM {
			return "", fmt.Errorf("background: method=%s needs MobileSAM (files.encoder/decoder)", method)
		}
	case methodCV:
		// no session needed
	default:
		return "", fmt.Errorf("background: unknown method %q (use auto, depth, sam, cv, or automask)", method)
	}
	return method, nil
}

// Infer dispatches to the selected method, which returns the support-surface (background)
// mask as a row-major bool bitmap at original-image resolution. The result is a single
// segmentation mask (0 masks when no support surface is found).
func (m *backgroundModel) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
	method, err := m.resolveMethod(prompt)
	if err != nil {
		return models.Result{}, err
	}
	res := models.Result{Task: models.TaskSegmentation}

	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	if w <= 0 || h <= 0 {
		return res, nil
	}

	var data []bool
	switch method {
	case methodAuto:
		data, err = m.backgroundAuto(img, prompt, r)
	case methodDepth:
		data, err = m.backgroundDepth(img, prompt, r)
	case methodSAM:
		data, err = m.backgroundSAM(img, prompt, r)
	case methodCV:
		data, err = m.backgroundCV(img, prompt, r)
	case methodAutomask:
		data, err = m.backgroundAutomask(img, prompt, r)
	}
	if err != nil {
		return models.Result{}, err
	}
	if data == nil || !anySet(data) {
		return res, nil // no support surface found
	}
	res.Masks = []models.Mask{unionBitmap(data, w, h).ToMask()}
	return res, nil
}

// backgroundAuto (method=auto, the default) tries the depth method first — when a clear
// support plane exists (RGB-D or a clean tabletop) it is the fastest and most accurate —
// and falls back to the classical-CV method when depth bows out (returns nothing), which is
// typical of monocular relative-depth on cluttered far-background scenes. cv always yields a
// result, so auto is robust by default.
func (m *backgroundModel) backgroundAuto(img image.Image, prompt models.Prompt, r models.Runner) ([]bool, error) {
	if m.hasDepth {
		data, err := m.backgroundDepth(img, prompt, r)
		if err != nil {
			return nil, err
		}
		if anySet(data) {
			return data, nil
		}
		// depth found no distinct plane → fall back to cv.
	}
	return m.backgroundCV(img, prompt, r)
}

// runDepth returns a depth/disparity map for the plane fit, with its width and height.
// When the request supplies an EXTERNAL depth map (RGB-D sensor, prompt.Depth), it is used
// directly (resized to the 256×256 working resolution, NaN-invalid preserved); a depth map whose
// data does not match its declared dims is an ERROR, not a reason to run MiDaS instead — the
// caller asked for its own geometry, and backgroundDepth treats any prompt.Depth as real RGB-D
// (it skips the monocular guard), so a silent MiDaS fallback also lost that guard. Otherwise it
// runs the MiDaS session (role depth) on the image with MiDaS's own preprocessing and returns the
// map at the session's OWN output resolution, read from the output shape. Used by method=depth.
func (m *backgroundModel) runDepth(img image.Image, prompt models.Prompt, r models.Runner) (depth []float32, dw, dh int, err error) {
	const sz = 256
	if prompt.Depth != nil {
		if prompt.DepthW <= 0 || prompt.DepthH <= 0 || len(prompt.Depth) != prompt.DepthW*prompt.DepthH {
			return nil, 0, 0, fmt.Errorf("background: external depth map has %d values but declares %dx%d",
				len(prompt.Depth), prompt.DepthW, prompt.DepthH)
		}
		return resizeDepthNearest(prompt.Depth, prompt.DepthW, prompt.DepthH, sz, sz), sz, sz, nil
	}
	resized := imageproc.Resize(img, sz, sz)
	in := imageproc.ImageToCHWFloat(resized, imagenetMean, imagenetStd)
	inName := util.FirstName(r.InputNames(roleDepth), "")
	outs, err := r.Run(roleDepth, map[string]engine.Tensor{inName: in})
	if err != nil {
		return nil, 0, 0, fmt.Errorf("background: depth inference: %w", err)
	}
	if len(outs) == 0 {
		return nil, 0, 0, fmt.Errorf("background: depth returned no output")
	}
	dw, dh, err = depthMapDims(outs[0])
	if err != nil {
		return nil, 0, 0, err
	}
	return outs[0].Data[:dw*dh], dw, dh, nil
}

// depthMapDims accepts exactly ONE depth map — [H,W], [1,H,W] or [1,1,H,W] (MiDaS small emits
// [1,256,256]) — and returns its width and height. Anything else (a batch, several channels, a
// flat vector) is an error rather than a guess about which values form the map.
func depthMapDims(t engine.Tensor) (w, h int, err error) {
	if len(t.Shape) < 2 {
		return 0, 0, fmt.Errorf("background: depth output shape %v is not a [.., H, W] map", t.Shape)
	}
	for _, d := range t.Shape[:len(t.Shape)-2] {
		if d != 1 {
			return 0, 0, fmt.Errorf("background: depth output shape %v holds more than one map", t.Shape)
		}
	}
	h, w = int(t.Shape[len(t.Shape)-2]), int(t.Shape[len(t.Shape)-1])
	if w <= 0 || h <= 0 || len(t.Data) < w*h {
		return 0, 0, fmt.Errorf("background: depth output shape %v with %d values", t.Shape, len(t.Data))
	}
	return w, h, nil
}

// resizeDepthNearest nearest-neighbor resizes a float depth map sw×sh → dw×dh (row-major).
// Nearest sampling preserves NaN (invalid) markers without blending them into valid depth.
func resizeDepthNearest(src []float32, sw, sh, dw, dh int) []float32 {
	return mask.UpsampleNearest(src, sh, sw, dh, dw)
}

// bgThresholds resolves the area thresholds (percent of image) for the mask-classifying
// methods (sam and automask), honoring per-request bg_max_area / fg_min_area. Both methods
// classify through isBackgroundMask, so the two knobs mean the same thing on either. The depth
// and cv methods do not classify masks and ignore them.
func (m *backgroundModel) bgThresholds(prompt models.Prompt) (bgMaxPct, minPct float64) {
	bgMaxPct = defaultBgMaxAreaPct
	if prompt.BgMaxArea > 0 {
		bgMaxPct = prompt.BgMaxArea
	}
	if prompt.FgMinArea > 0 {
		minPct = prompt.FgMinArea
	}
	return bgMaxPct, minPct
}

// isBackgroundMask reports whether a mask (area in pixels, plus border touch) qualifies as
// a support surface under the area/border heuristics. imgArea is W*H. A mask below minPct
// (fg_min_area) is noise whatever else holds; one at or above bgMaxPct (bg_max_area) is a
// surface; otherwise it must touch the border and cover borderMinAreaPct.
func isBackgroundMask(areaPx, imgArea float64, border bool, bgMaxPct, minPct float64) bool {
	areaPct := areaPx / imgArea * 100.0
	if areaPct < minPct {
		return false
	}
	if areaPct >= bgMaxPct {
		return true
	}
	return border && areaPct >= borderMinAreaPct
}

// --- shared bitmap helpers ---

func anySet(data []bool) bool {
	for _, v := range data {
		if v {
			return true
		}
	}
	return false
}

func bitmapArea(data []bool) float64 {
	n := 0
	for _, v := range data {
		if v {
			n++
		}
	}
	return float64(n)
}

// touchesBorder reports whether any set pixel lies on the image's outer edge.
func touchesBorder(data []bool, w, h int) bool {
	for x := 0; x < w; x++ {
		if data[x] || data[(h-1)*w+x] {
			return true
		}
	}
	for y := 0; y < h; y++ {
		if data[y*w] || data[y*w+w-1] {
			return true
		}
	}
	return false
}

// unionBitmap wraps a row-major mask into a MaskBitmap with its tight bbox (Conf=1) so it
// can be encoded to the public models.Mask (column-major RLE) via ToMask.
func unionBitmap(data []bool, w, h int) mobilesam.MaskBitmap {
	return mobilesam.MaskBitmap{Data: data, W: w, H: h, BBox: mask.Bitmap{Data: data, W: w, H: h}.BBox(), Conf: 1.0}
}

// upsampleMask nearest-neighbor resizes a sw×sh bool mask to dw×dh (row-major); the same
// slice comes back when the size is unchanged.
func upsampleMask(src []bool, sw, sh, dw, dh int) []bool {
	return mask.ResizeNearest(mask.Bitmap{Data: src, W: sw, H: sh}, dh, dw).Data
}
