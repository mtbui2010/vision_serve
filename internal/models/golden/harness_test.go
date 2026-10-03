// Package golden pins the CURRENT end-to-end pre/postprocess outputs of the non-open-vocab
// model packages (and the shared mask/RLE/NMS/explain helpers they use) so a refactor can
// prove it changed nothing.
//
// Every case drives a model only through its public surface (models.New + the Model /
// PipelineModel interfaces, mobilesam.Segment, pkg/api, morph, explain), feeding fixed
// synthetic images and synthetic ONNX outputs with the REAL tensor shapes of the shipped
// exports (inspected with onnxruntime; see the shape comments next to each fake). A fake
// models.Runner records a digest of every tensor the model hands to "ONNX", so the
// preprocessing is pinned too, not only the decoded Result.
//
// The expected values live in testdata/*.json. Regenerate them ONLY when a behaviour change
// is intended:
//
//	go test ./internal/models/golden -update
package golden

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	"visionserve/pkg/api"
)

var update = flag.Bool("update", false, "rewrite the golden files in testdata/")

// repoRoot is the module root relative to this package directory.
const repoRoot = "../../.."

// ---------------------------------------------------------------------------------------
// Golden file plumbing
// ---------------------------------------------------------------------------------------

// golden collects the records of one test (one testdata/<name>.json file).
type golden struct {
	t     *testing.T
	name  string
	cases map[string]any
}

func newGolden(t *testing.T, name string) *golden {
	return &golden{t: t, name: name, cases: map[string]any{}}
}

// add records the value pinned for one case. Case names must be unique.
func (g *golden) add(caseName string, v any) {
	g.t.Helper()
	if _, dup := g.cases[caseName]; dup {
		g.t.Fatalf("golden %s: duplicate case %q", g.name, caseName)
	}
	g.cases[caseName] = v
}

// check compares every recorded case with testdata/<name>.json (or rewrites it with -update).
func (g *golden) check() {
	g.t.Helper()
	path := filepath.Join("testdata", g.name+".json")
	got := map[string]json.RawMessage{}
	for k, v := range g.cases {
		b, err := json.Marshal(v)
		if err != nil {
			g.t.Fatalf("golden %s: marshal case %q: %v", g.name, k, err)
		}
		got[k] = b
	}
	if *update {
		b, err := json.MarshalIndent(got, "", " ")
		if err != nil {
			g.t.Fatalf("golden %s: %v", g.name, err)
		}
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			g.t.Fatal(err)
		}
		if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
			g.t.Fatal(err)
		}
		g.t.Logf("golden %s: wrote %d cases to %s", g.name, len(got), path)
		return
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		g.t.Fatalf("golden %s: %v (run with -update to create it)", g.name, err)
	}
	want := map[string]json.RawMessage{}
	if err := json.Unmarshal(raw, &want); err != nil {
		g.t.Fatalf("golden %s: parse %s: %v", g.name, path, err)
	}
	names := make([]string, 0, len(got))
	for k := range got {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		w, ok := want[k]
		if !ok {
			g.t.Errorf("golden %s: case %q is not in %s (run with -update if it is new)", g.name, k, path)
			continue
		}
		var a, b bytes.Buffer
		if err := json.Compact(&a, got[k]); err != nil {
			g.t.Fatal(err)
		}
		if err := json.Compact(&b, w); err != nil {
			g.t.Fatal(err)
		}
		if !bytes.Equal(a.Bytes(), b.Bytes()) {
			g.t.Errorf("golden %s: case %q differs:\n%s", g.name, k, firstDiff(a.String(), b.String()))
			if dir := os.Getenv("GOLDEN_DUMP"); dir != "" {
				_ = os.MkdirAll(dir, 0o755)
				_ = os.WriteFile(filepath.Join(dir, g.name+"."+sanitize(k)+".got.json"), a.Bytes(), 0o644)
				_ = os.WriteFile(filepath.Join(dir, g.name+"."+sanitize(k)+".want.json"), b.Bytes(), 0o644)
			}
		}
	}
	for k := range want {
		if _, ok := got[k]; !ok {
			g.t.Errorf("golden %s: case %q is in %s but was not produced", g.name, k, path)
		}
	}
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '/' || r == ' ' || r == ':' {
			return '_'
		}
		return r
	}, s)
}

// firstDiff shows a window around the first differing byte of two JSON strings.
func firstDiff(got, want string) string {
	i := 0
	for i < len(got) && i < len(want) && got[i] == want[i] {
		i++
	}
	lo := i - 120
	if lo < 0 {
		lo = 0
	}
	clip := func(s string) string {
		hi := i + 200
		if hi > len(s) {
			hi = len(s)
		}
		if lo > len(s) {
			return ""
		}
		return s[lo:hi]
	}
	return fmt.Sprintf("  at byte %d\n  got:  …%s…\n  want: …%s…", i, clip(got), clip(want))
}

// ---------------------------------------------------------------------------------------
// Tensor / bitmap digests
// ---------------------------------------------------------------------------------------

// tensorRec pins a tensor: its shape, an exact digest of its bits and a few readable stats.
type tensorRec struct {
	Shape []int64   `json:"shape"`
	Dtype string    `json:"dtype,omitempty"`
	N     int       `json:"n"`
	SHA   string    `json:"sha"`
	Sum   float64   `json:"sum"`
	Head  []float32 `json:"head,omitempty"`
	HeadI []int64   `json:"head_i,omitempty"`
}

func shaF32(d []float32) string {
	h := sha256.New()
	var buf [4096]byte
	n := 0
	for _, v := range d {
		binary.LittleEndian.PutUint32(buf[n:], math.Float32bits(v))
		n += 4
		if n == len(buf) {
			h.Write(buf[:])
			n = 0
		}
	}
	h.Write(buf[:n])
	return hex.EncodeToString(h.Sum(nil))[:24]
}

func shaI64(d []int64) string {
	h := sha256.New()
	var b [8]byte
	for _, v := range d {
		binary.LittleEndian.PutUint64(b[:], uint64(v))
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))[:24]
}

func shaBool(d []bool) string {
	h := sha256.New()
	buf := make([]byte, len(d))
	for i, v := range d {
		if v {
			buf[i] = 1
		}
	}
	h.Write(buf)
	return hex.EncodeToString(h.Sum(nil))[:24]
}

// shaCache memoises digests of LARGE read-only buffers (the shared image embedding and the
// shared zero mask_input are handed to every decoder call) by their backing array. Each
// entry keeps its slice alive, so the address cannot be recycled for another buffer while
// the cache (one fake runner = one case) exists.
type shaCache struct{ m sync.Map }

type sliceKey struct {
	p uintptr
	n int
}

type cachedSHA struct {
	keep []float32
	sha  string
}

func (c *shaCache) f32(d []float32) string {
	if len(d) < 1<<16 {
		return shaF32(d)
	}
	k := sliceKey{uintptr(unsafe.Pointer(unsafe.SliceData(d))), len(d)}
	if v, ok := c.m.Load(k); ok {
		return v.(cachedSHA).sha
	}
	s := shaF32(d)
	c.m.Store(k, cachedSHA{keep: d, sha: s})
	return s
}

func recTensorC(c *shaCache, t engine.Tensor) tensorRec {
	r := tensorRec{Shape: t.Shape, Dtype: t.Dtype}
	if t.Dtype == "i64" {
		r.N = len(t.DataI64)
		r.SHA = shaI64(t.DataI64)
		for i, v := range t.DataI64 {
			r.Sum += float64(v)
			if i < 6 {
				r.HeadI = append(r.HeadI, v)
			}
		}
		return r
	}
	r.N = len(t.Data)
	if c != nil {
		r.SHA = c.f32(t.Data)
	} else {
		r.SHA = shaF32(t.Data)
	}
	if len(t.Data) <= 1<<16 || c == nil {
		for _, v := range t.Data {
			r.Sum += float64(v)
		}
	}
	for i := 0; i < len(t.Data) && i < 6; i++ {
		r.Head = append(r.Head, t.Data[i])
	}
	return r
}

func recTensor(t engine.Tensor) tensorRec { return recTensorC(nil, t) }

// ---------------------------------------------------------------------------------------
// Fake Runner
// ---------------------------------------------------------------------------------------

type genFunc func(role string, in map[string]engine.Tensor) ([]engine.Tensor, error)

// fakeRunner is a models.Runner that answers with synthetic outputs and records a digest of
// every input it receives. Safe for concurrent use (the MobileSAM AMG calls it in parallel).
type fakeRunner struct {
	inNames  map[string][]string
	outNames map[string][]string
	gen      genFunc

	cache shaCache
	mu    sync.Mutex
	calls []callRec
}

type callRec struct {
	role   string
	digest string
	inputs map[string]tensorRec
}

func newFakeRunner(in, out map[string][]string, gen genFunc) *fakeRunner {
	return &fakeRunner{inNames: in, outNames: out, gen: gen}
}

func (f *fakeRunner) InputNames(role string) []string  { return f.inNames[role] }
func (f *fakeRunner) OutputNames(role string) []string { return f.outNames[role] }

func (f *fakeRunner) Run(role string, inputs map[string]engine.Tensor) ([]engine.Tensor, error) {
	recs := make(map[string]tensorRec, len(inputs))
	names := make([]string, 0, len(inputs))
	for k, t := range inputs {
		recs[k] = recTensorC(&f.cache, t)
		names = append(names, k)
	}
	sort.Strings(names)
	h := sha256.New()
	for _, k := range names {
		fmt.Fprintf(h, "%s:%v:%s;", k, recs[k].Shape, recs[k].SHA)
	}
	d := hex.EncodeToString(h.Sum(nil))[:16]
	f.mu.Lock()
	f.calls = append(f.calls, callRec{role: role, digest: d, inputs: recs})
	f.mu.Unlock()
	return f.gen(role, inputs)
}

// callsRecord returns the calls as a canonical (order-independent) record: the sorted
// "role:digest" list, plus the full input description of the smallest-digest call per role.
func (f *fakeRunner) callsRecord() ([]string, map[string]map[string]tensorRec) {
	f.mu.Lock()
	defer f.mu.Unlock()
	cs := append([]callRec(nil), f.calls...)
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].role != cs[j].role {
			return cs[i].role < cs[j].role
		}
		return cs[i].digest < cs[j].digest
	})
	list := make([]string, len(cs))
	first := map[string]map[string]tensorRec{}
	for i, c := range cs {
		list[i] = c.role + ":" + c.digest
		if _, ok := first[c.role]; !ok {
			first[c.role] = c.inputs
		}
	}
	return list, first
}

// ---------------------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------------------

// pipelineRec is what a PipelineModel case pins.
type pipelineRec struct {
	Calls      []string                        `json:"calls"`
	CallInputs map[string]map[string]tensorRec `json:"call_inputs,omitempty"`
	Result     *api.Result                     `json:"result,omitempty"`
	Bitmaps    []bitmapRec                     `json:"bitmaps,omitempty"`
	Err        string                          `json:"err,omitempty"`
}

type bitmapRec struct {
	W, H int
	SHA  string
	Area int
	BBox [4]float64
	Conf float64
}

func recBitmap(data []bool, w, h int, bbox [4]float64, conf float64) bitmapRec {
	area := 0
	for _, v := range data {
		if v {
			area++
		}
	}
	return bitmapRec{W: w, H: h, SHA: shaBool(data), Area: area, BBox: bbox, Conf: conf}
}

func runPipeline(t *testing.T, pm models.PipelineModel, img image.Image, p models.Prompt, r *fakeRunner) pipelineRec {
	t.Helper()
	res, err := pm.Infer(img, p, r)
	rec := pipelineRec{}
	rec.Calls, rec.CallInputs = r.callsRecord()
	if err != nil {
		rec.Err = err.Error()
		return rec
	}
	rec.Result = &res
	return rec
}

// preRec pins Model.Preprocess.
type preRec struct {
	Tensor tensorRec             `json:"tensor"`
	Meta   models.PreprocessMeta `json:"meta"`
	Err    string                `json:"err,omitempty"`
}

// postRec pins Model.Postprocess. A dense depth map is pinned by digest (DepthMapRec) and
// removed from Result to keep the golden files small.
type postRec struct {
	Result      *api.Result `json:"result,omitempty"`
	DepthMapRec *tensorRec  `json:"depth_map,omitempty"`
	Err         string      `json:"err,omitempty"`
}

func doPre(m models.Model, img image.Image) (preRec, models.PreprocessMeta) {
	ten, meta, err := m.Preprocess(img)
	if err != nil {
		return preRec{Err: err.Error()}, meta
	}
	return preRec{Tensor: recTensor(ten), Meta: meta}, meta
}

func doPost(m models.Model, outs []engine.Tensor, meta models.PreprocessMeta) postRec {
	res, err := m.Postprocess(outs, meta)
	if err != nil {
		return postRec{Err: err.Error()}
	}
	rec := postRec{Result: &res}
	if len(res.DepthMap) > 0 {
		d := recTensor(engine.F32(res.DepthMap, int64(res.DepthHeight), int64(res.DepthWidth)))
		var mn, mx float32 = res.DepthMap[0], res.DepthMap[0]
		for _, v := range res.DepthMap {
			mn, mx = min(mn, v), max(mx, v)
		}
		d.Head = append(d.Head, mn, mx)
		rec.DepthMapRec = &d
		res.DepthMap = nil
	}
	return rec
}

// ---------------------------------------------------------------------------------------
// Model construction
// ---------------------------------------------------------------------------------------

func newModel(t *testing.T, arch string, cfg models.Config) models.Model {
	t.Helper()
	b, err := models.New(arch, cfg)
	if err != nil {
		t.Fatalf("models.New(%q): %v", arch, err)
	}
	m, ok := b.(models.Model)
	if !ok {
		t.Fatalf("%q is not a plain Model", arch)
	}
	return m
}

func newPipeline(t *testing.T, arch string, cfg models.Config) models.PipelineModel {
	t.Helper()
	b, err := models.New(arch, cfg)
	if err != nil {
		t.Fatalf("models.New(%q): %v", arch, err)
	}
	m, ok := b.(models.PipelineModel)
	if !ok {
		t.Fatalf("%q is not a PipelineModel", arch)
	}
	return m
}

func readLabels(t *testing.T, rel string) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot, rel))
	if err != nil {
		t.Fatalf("labels: %v", err)
	}
	var out []string
	for _, line := range strings.Split(string(raw), "\n") {
		if s := strings.TrimSpace(line); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// ---------------------------------------------------------------------------------------
// Synthetic inputs
// ---------------------------------------------------------------------------------------

// synthImage is a deterministic w×h test picture: smooth gradients, a few solid rectangles
// and per-pixel noise, so resizers and normalisers see real structure.
func synthImage(w, h int, seed int64) *image.NRGBA {
	r := rand.New(rand.NewSource(seed))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i] = uint8((x*255)/max(w, 1) ^ (y & 31))
			img.Pix[i+1] = uint8((y * 255) / max(h, 1))
			img.Pix[i+2] = uint8(128 + 100*math.Sin(float64(x+y)/17))
			img.Pix[i+3] = 255
		}
	}
	for k := 0; k < 6; k++ {
		x0, y0 := r.Intn(w), r.Intn(h)
		x1, y1 := x0+1+r.Intn(max(w/3, 1)), y0+1+r.Intn(max(h/3, 1))
		c := color.NRGBA{uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256)), 255}
		for y := y0; y < y1 && y < h; y++ {
			for x := x0; x < x1 && x < w; x++ {
				img.SetNRGBA(x, y, c)
			}
		}
	}
	for i := 0; i < len(img.Pix); i += 4 {
		n := r.Intn(9) - 4
		for c := 0; c < 3; c++ {
			v := int(img.Pix[i+c]) + n
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			img.Pix[i+c] = uint8(v)
		}
	}
	return img
}

// loadSample decodes the committed test/testdata/sample.jpg (810×1080 photo).
func loadSample(t *testing.T) image.Image {
	t.Helper()
	f, err := os.Open(filepath.Join(repoRoot, "test", "testdata", "sample.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// seedOf derives a deterministic PRNG seed from float data (exact bits).
func seedOf(parts ...[]float32) int64 {
	h := sha256.New()
	var b [4]byte
	for _, p := range parts {
		for _, v := range p {
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			h.Write(b[:])
		}
		h.Write([]byte{0xff})
	}
	s := h.Sum(nil)
	return int64(binary.LittleEndian.Uint64(s[:8]) & (1<<62 - 1))
}

func randNorm(r *rand.Rand, n int, mean, sd float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(mean + sd*r.NormFloat64())
	}
	return out
}

func randUniform(r *rand.Rand, n int, lo, hi float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(lo + (hi-lo)*r.Float64())
	}
	return out
}

// blob parameterises one synthetic SAM mask channel: an ellipse in NORMALISED coordinates
// with a separable ripple, so the same blob rendered at any frame size gives a consistent
// (but multi-component, ragged-edged) mask.
type blob struct{ cx, cy, rx, ry, amp, freq float64 }

func randBlob(r *rand.Rand) blob {
	return blob{
		cx: 0.15 + 0.7*r.Float64(), cy: 0.15 + 0.7*r.Float64(),
		rx: 0.05 + 0.3*r.Float64(), ry: 0.05 + 0.3*r.Float64(),
		amp: 1.2 * r.Float64(), freq: 2 + 7*r.Float64(),
	}
}

// render writes the blob's logits (clamped to ±32, like SAM2's export) into dst[h*w].
func (b blob) render(dst []float32, w, h int) {
	su := make([]float64, w)
	for x := range su {
		u := (float64(x) + 0.5) / float64(w)
		su[x] = math.Sin(2 * math.Pi * b.freq * u)
	}
	for y := 0; y < h; y++ {
		v := (float64(y) + 0.5) / float64(h)
		cv := math.Cos(2 * math.Pi * b.freq * v)
		dv := (v - b.cy) / b.ry
		row := dst[y*w : (y+1)*w]
		for x := range row {
			u := (float64(x) + 0.5) / float64(w)
			du := (u - b.cx) / b.rx
			l := 8*(1-du*du-dv*dv) + 8*b.amp*su[x]*cv
			if l > 32 {
				l = 32
			}
			if l < -32 {
				l = -32
			}
			row[x] = float32(l)
		}
	}
}
