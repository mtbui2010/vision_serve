package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"math"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"visionserve/internal/cli/clireport"
	"visionserve/internal/engine"
	"visionserve/internal/imageproc"
	"visionserve/internal/lifecycle"
	"visionserve/internal/models"
	"visionserve/internal/registry"
	"visionserve/internal/server"
	"visionserve/internal/templates"
	"visionserve/pkg/api"
)

// Bounds on what bench reads or allocates (engineering rule: bound every external input).
const (
	benchMaxImages     = 256
	benchMaxImageBytes = 64 << 20
	benchMaxSide       = 8192
	benchMaxRequests   = 100000
	benchMaxConc       = 256
)

const benchUsage = `visionserve bench <model> [flags]

Measures how fast a model runs on THIS machine: cold load, latency percentiles, throughput at a
given concurrency, memory, and the execution provider actually used. Run it on the device you
deploy to (a Jetson, a PC): numbers from another machine say nothing about it.

  --images DIR        photos to send (cycled; up to 256). Default: a synthetic --size image
  --size WxH          synthetic image size when --images is not given (default 640x480)
  --requests N        timed requests (default 50)
  --concurrency C     requests in flight at once (default 1)
  --warmup N          untimed requests before timing (default 3; the first is reported separately)
  --ep auto|cpu|cuda|tensorrt
                      in-process: force the execution provider chain (auto = the manifest's,
                      cuda → cpu by default). tensorrt = TensorRT → CUDA → CPU; it measured 6.8 mAP
                      lower on GroundingDINO (BUGS_TO_FIX.md #3): check accuracy before using it.
                      With --server the server's own chain applies; --ep then only sets what to expect.
  --server URL        bench a running server (whole HTTP request, multipart upload)
  --in-process        load the model in this process, like visionserve run (the default)
  --reload            with --server: unload and reload the model to measure a cold load
  --models DIR        registry (in-process; default $VISIONSERVE_MODELS or ~/.visionserve/models)
  --json              print one JSON object {"verdict","reason","summary","details"} and nothing else
  --report FILE.html  also write a self-contained HTML report
  --prompt / --box / --point / ...   request options, as for visionserve run. Without one, an
                      open-vocabulary model gets the prompt "person. car. dog. cat." and a SAM
                      model a box over the centre of the image.

Verdict: PASS when every request succeeded; WARN when a GPU was requested (or is present) but the
model ran on the CPU, or TensorRT was requested but CUDA ran; FAIL when requests failed.
Exit status: 0 PASS/WARN, 1 FAIL, 2 usage or setup error.
`

type benchOptions struct {
	model       string
	imagesDir   string
	size        string
	requests    int
	concurrency int
	warmup      int
	ep          string
	serverURL   string
	inProcess   bool
	reload      bool
	modelsDir   string
	out         *clireport.Output
	req         runOptions
}

// parseBenchArgs parses `bench` flags (interleaved with the model positional).
func parseBenchArgs(args []string, stderr io.Writer) (*benchOptions, error) {
	o := &benchOptions{}
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, benchUsage) }
	fs.StringVar(&o.imagesDir, "images", "", "directory of photos to send (cycled)")
	fs.StringVar(&o.size, "size", "", "synthetic image size WxH (default 640x480)")
	fs.IntVar(&o.requests, "requests", 50, "timed requests")
	fs.IntVar(&o.concurrency, "concurrency", 1, "requests in flight at once")
	fs.IntVar(&o.warmup, "warmup", 3, "untimed warm-up requests")
	fs.StringVar(&o.ep, "ep", "auto", "execution provider: auto|cpu|cuda|tensorrt")
	fs.StringVar(&o.serverURL, "server", "", "bench a running server at this URL")
	fs.BoolVar(&o.inProcess, "in-process", false, "load the model in this process (default)")
	fs.BoolVar(&o.reload, "reload", false, "with --server: unload + reload to measure a cold load")
	fs.StringVar(&o.modelsDir, "models", "", "model registry directory")
	o.out = clireport.AddFlags(fs)
	addRequestFlags(fs, &o.req)
	var pos []string
	rem := args
	for len(rem) > 0 {
		if err := fs.Parse(rem); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil, err
			}
			return nil, clireport.Usage(err)
		}
		rem = fs.Args()
		if len(rem) > 0 {
			pos = append(pos, rem[0])
			rem = rem[1:]
		}
	}
	if len(pos) != 1 {
		return nil, clireport.Usagef("usage: visionserve bench <model> [flags] (see visionserve bench --help)")
	}
	o.model = pos[0]
	return o, o.validate()
}

func (o *benchOptions) validate() error {
	switch o.ep {
	case "auto", "cpu", "cuda", "tensorrt":
	default:
		return clireport.Usagef("--ep must be auto, cpu, cuda or tensorrt, got %q", o.ep)
	}
	if o.requests < 1 || o.requests > benchMaxRequests {
		return clireport.Usagef("--requests must be 1..%d, got %d", benchMaxRequests, o.requests)
	}
	if o.concurrency < 1 || o.concurrency > benchMaxConc {
		return clireport.Usagef("--concurrency must be 1..%d, got %d", benchMaxConc, o.concurrency)
	}
	if o.warmup < 0 || o.warmup > benchMaxRequests {
		return clireport.Usagef("--warmup must be 0..%d, got %d", benchMaxRequests, o.warmup)
	}
	if o.imagesDir != "" && o.size != "" {
		return clireport.Usagef("--images and --size are alternatives: pass one")
	}
	if o.serverURL != "" && o.inProcess {
		return clireport.Usagef("--server and --in-process are alternatives: pass one")
	}
	if o.reload && o.serverURL == "" {
		return clireport.Usagef("--reload applies to --server (in-process always measures a cold load)")
	}
	if o.serverURL != "" {
		if o.req.depthFile != "" || len(o.req.templates) > 0 {
			return clireport.Usagef("--depth and --template are in-process only")
		}
		u, err := url.Parse(o.serverURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return clireport.Usagef("--server must be an http(s) URL like http://127.0.0.1:11435, got %q", o.serverURL)
		}
	}
	if o.size != "" {
		if _, _, err := parseBenchSize(o.size); err != nil {
			return clireport.Usagef("--size: %v", err)
		}
	}
	return nil
}

func parseBenchSize(s string) (int, int, error) {
	a, b, ok := strings.Cut(strings.ToLower(strings.TrimSpace(s)), "x")
	if !ok {
		return 0, 0, fmt.Errorf("expected WxH, e.g. 640x480, got %q", s)
	}
	w, err1 := strconv.Atoi(a)
	h, err2 := strconv.Atoi(b)
	if err1 != nil || err2 != nil || w < 16 || h < 16 || w > benchMaxSide || h > benchMaxSide {
		return 0, 0, fmt.Errorf("expected WxH with 16 <= W,H <= %d, got %q", benchMaxSide, s)
	}
	return w, h, nil
}

// benchImage is one input: the decoded image (in-process), its encoded bytes (server) and name.
type benchImage struct {
	name string
	img  image.Image
	raw  []byte
	w, h int
}

func loadBenchImages(dir, size string, needRaw bool) ([]benchImage, string, error) {
	if dir == "" {
		if size == "" {
			size = "640x480"
		}
		w, h, err := parseBenchSize(size)
		if err != nil {
			return nil, "", clireport.Usagef("--size: %v", err)
		}
		img := syntheticImage(w, h)
		bi := benchImage{name: "synthetic-" + size, img: img, w: w, h: h}
		if needRaw {
			var buf bytes.Buffer
			if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
				return nil, "", err
			}
			bi.raw = buf.Bytes()
		}
		return []benchImage{bi}, fmt.Sprintf("synthetic %dx%d image (pass --images DIR for real photos)", w, h), nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, "", clireport.Usagef("--images: %v", err)
	}
	var files []string
	for _, e := range entries {
		switch strings.ToLower(filepath.Ext(e.Name())) {
		case ".jpg", ".jpeg", ".png", ".webp", ".bmp":
			if !e.IsDir() {
				files = append(files, filepath.Join(dir, e.Name()))
			}
		}
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, "", clireport.Usagef("--images %s: no .jpg/.png/.webp/.bmp files", dir)
	}
	truncated := len(files) > benchMaxImages
	if truncated {
		files = files[:benchMaxImages]
	}
	out := make([]benchImage, 0, len(files))
	for _, f := range files {
		st, err := os.Stat(f)
		if err != nil {
			return nil, "", clireport.Usagef("--images: %v", err)
		}
		if st.Size() > benchMaxImageBytes {
			return nil, "", clireport.Usagef("--images: %s is larger than %d MB", f, benchMaxImageBytes>>20)
		}
		raw, err := os.ReadFile(f)
		if err != nil {
			return nil, "", clireport.Usagef("--images: %v", err)
		}
		img, err := imageproc.Decode(raw) // the server's decoder (EXIF orientation), like `run`
		if err != nil {
			return nil, "", clireport.Usagef("--images: %s: %v", f, err)
		}
		bi := benchImage{name: filepath.Base(f), img: img, w: img.Bounds().Dx(), h: img.Bounds().Dy()}
		if needRaw {
			bi.raw = raw
		}
		out = append(out, bi)
	}
	desc := fmt.Sprintf("%d photo(s) from %s, cycled", len(out), dir)
	if truncated {
		desc += fmt.Sprintf(" (first %d only)", benchMaxImages)
	}
	return out, desc, nil
}

// syntheticImage is a deterministic test pattern: gradients and blocks, so preprocessing and
// the model do real work (a flat image can take shortcuts in some kernels).
func syntheticImage(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.RGBA{uint8(255 * x / w), uint8(255 * y / h), uint8((x*7 + y*13) % 256), 255}
			if (x/(w/8+1)+y/(h/6+1))%3 == 0 {
				c = color.RGBA{c.R / 2, 200, c.B / 3, 255}
			}
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

// benchSample is one timed request.
type benchSample struct {
	e2eMs    float64
	serverMs float64
	device   string
	err      error
}

// benchTarget is what bench drives: the model in this process, or a server.
type benchTarget interface {
	where() string
	load(ctx context.Context) (coldMs float64, note string, err error)
	predict(ctx context.Context, img benchImage, req api.PredictJSONRequest) benchSample
	pid() int
	close()
}

// ---- in-process ------------------------------------------------------------------------

type inProcessTarget struct {
	model string
	mgr   *lifecycle.Manager
}

func (t *inProcessTarget) where() string { return "in-process" }

func (t *inProcessTarget) load(ctx context.Context) (float64, string, error) {
	start := time.Now()
	if err := t.mgr.Load(ctx, t.model); err != nil {
		return 0, "", err
	}
	return ms(time.Since(start)), "session(s) created in this process", nil
}

func (t *inProcessTarget) predict(ctx context.Context, img benchImage, q api.PredictJSONRequest) benchSample {
	req := server.Request{PredictJSONRequest: q}
	prompt, err := req.ToPrompt(img.w, img.h)
	if err != nil {
		return benchSample{err: err}
	}
	start := time.Now()
	res, err := server.Predict(ctx, t.mgr, t.model, img.img, prompt)
	e2e := ms(time.Since(start))
	if err != nil {
		return benchSample{err: err}
	}
	return benchSample{e2eMs: e2e, serverMs: res.DurationMs, device: res.Device}
}

func (t *inProcessTarget) pid() int { return os.Getpid() }
func (t *inProcessTarget) close()   { t.mgr.Close() }

// ---- server ------------------------------------------------------------------------------

type serverTarget struct {
	base   string
	model  string
	reload bool
	client *http.Client
}

func (t *serverTarget) where() string { return "server " + t.base }

func (t *serverTarget) models(ctx context.Context) ([]api.ModelInfo, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, t.base+"/api/models", nil)
	resp, err := t.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("GET /api/models: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var infos []api.ModelInfo
	if err := json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&infos); err != nil {
		return nil, fmt.Errorf("GET /api/models: %w", err)
	}
	return infos, nil
}

func (t *serverTarget) post(ctx context.Context, path string, body any) error {
	raw, _ := json.Marshal(body)
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, t.base+path, bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	resp, err := t.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if resp.StatusCode >= 300 {
		return fmt.Errorf("POST %s: HTTP %d: %s", path, resp.StatusCode, strings.TrimSpace(string(b)))
	}
	return nil
}

func (t *serverTarget) load(ctx context.Context) (float64, string, error) {
	infos, err := t.models(ctx)
	if err != nil {
		return 0, "", clireport.Usage(fmt.Errorf("cannot reach the server at %s: %w", t.base, err))
	}
	state := ""
	for _, in := range infos {
		if in.Name == t.model {
			state = in.State
		}
	}
	if state == "" {
		return 0, "", clireport.Usagef("the server at %s does not list %q (a server only sees its own --models registry)", t.base, t.model)
	}
	if state == "loaded" && !t.reload {
		return math.NaN(), "already loaded on the server, not measured (pass --reload to unload and measure a cold load)", nil
	}
	if state == "loaded" {
		if err := t.post(ctx, "/api/unload", api.LoadRequest{Model: t.model}); err != nil {
			return 0, "", err
		}
	}
	start := time.Now()
	if err := t.post(ctx, "/api/load", api.LoadRequest{Model: t.model}); err != nil {
		return 0, "", err
	}
	return ms(time.Since(start)), "POST /api/load on the server", nil
}

// multipartBody renders the request as the multipart form clients send: model, image, options.
func multipartBody(img benchImage, q api.PredictJSONRequest) (*bytes.Buffer, string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	q.ImageBase64 = ""
	raw, err := json.Marshal(q)
	if err != nil {
		return nil, "", err
	}
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, "", err
	}
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := fields[k]
		var s string
		switch x := v.(type) {
		case string:
			s = x
		case float64:
			s = strconv.FormatFloat(x, 'g', -1, 64)
		default:
			s = fmt.Sprint(x)
		}
		if s == "" || (k != "model" && s == "0") {
			continue
		}
		if err := mw.WriteField(k, s); err != nil {
			return nil, "", err
		}
	}
	part, err := mw.CreateFormFile("image", img.name)
	if err != nil {
		return nil, "", err
	}
	if _, err := part.Write(img.raw); err != nil {
		return nil, "", err
	}
	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	return &buf, mw.FormDataContentType(), nil
}

func (t *serverTarget) predict(ctx context.Context, img benchImage, q api.PredictJSONRequest) benchSample {
	q.Model = t.model
	body, ctype, err := multipartBody(img, q)
	if err != nil {
		return benchSample{err: err}
	}
	start := time.Now()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, t.base+"/api/predict", body)
	req.Header.Set("Content-Type", ctype)
	resp, err := t.client.Do(req)
	if err != nil {
		return benchSample{err: err}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 256<<20))
	e2e := ms(time.Since(start))
	if err != nil {
		return benchSample{err: err}
	}
	if resp.StatusCode >= 300 {
		return benchSample{err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(strings.TrimSpace(string(raw)), 300))}
	}
	var res api.Result
	if err := json.Unmarshal(raw, &res); err != nil {
		return benchSample{err: fmt.Errorf("decode response: %w", err)}
	}
	return benchSample{e2eMs: e2e, serverMs: res.DurationMs, device: res.Device}
}

// pid finds the server's process when it listens on this machine (for its memory).
func (t *serverTarget) pid() int {
	u, err := url.Parse(t.base)
	if err != nil {
		return 0
	}
	host, port := u.Hostname(), u.Port()
	if port == "" {
		return 0
	}
	if ip := net.ParseIP(host); host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		return 0
	}
	p, err := strconv.Atoi(port)
	if err != nil {
		return 0
	}
	return pidListeningOn(p)
}

func (t *serverTarget) close() {}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// ---- statistics --------------------------------------------------------------------------

// percentile of sorted samples, linear between closest ranks (numpy's default).
func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return math.NaN()
	}
	if n == 1 {
		return sorted[0]
	}
	pos := p / 100 * float64(n-1)
	lo := int(math.Floor(pos))
	hi := min(lo+1, n-1)
	return sorted[lo] + (sorted[hi]-sorted[lo])*(pos-float64(lo))
}

type latencyStats struct {
	P50, P95, P99, Mean, Min, Max float64
	N                             int
}

func statsOf(samples []float64) latencyStats {
	s := append([]float64(nil), samples...)
	sort.Float64s(s)
	st := latencyStats{N: len(s), P50: percentile(s, 50), P95: percentile(s, 95), P99: percentile(s, 99),
		Mean: math.NaN(), Min: math.NaN(), Max: math.NaN()}
	if len(s) > 0 {
		sum := 0.0
		for _, v := range s {
			sum += v
		}
		st.Mean, st.Min, st.Max = sum/float64(len(s)), s[0], s[len(s)-1]
	}
	return st
}

func (s latencyStats) json() map[string]any {
	return map[string]any{"p50_ms": s.P50, "p95_ms": s.P95, "p99_ms": s.P99, "mean_ms": s.Mean,
		"min_ms": s.Min, "max_ms": s.Max, "n": s.N}
}

// ---- run ---------------------------------------------------------------------------------

// benchResult is everything measured; it becomes the report.
type benchResult struct {
	opts        *benchOptions
	where       string
	images      string
	promptNote  string
	coldMs      float64
	coldNote    string
	firstMs     float64
	samples     []benchSample
	wall        time.Duration
	devices     map[string]int
	rss, rssMax float64
	gpu         gpuMem
	gpuKind     gpuKind
	pid         int
	manifestEPs []string
	arch, task  string
	epChain     []string
}

// runBench is `visionserve bench`.
func runBench(args []string) error {
	o, err := parseBenchArgs(args, os.Stderr)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	res, err := executeBench(context.Background(), o, os.Stderr)
	if err != nil {
		if clireport.ExitCode(err) == clireport.ExitUsage {
			return err
		}
		return clireport.Usage(err)
	}
	return o.out.Emit(os.Stdout, os.Stderr, res.report())
}

// applyBenchEP sets the in-process EP chain before anything loads.
func applyBenchEP(ep string) {
	switch ep {
	case "cpu":
		os.Setenv("VISIONSERVE_EP", "cpu")
	case "cuda":
		os.Setenv("VISIONSERVE_EP", "cuda")
	case "tensorrt":
		os.Setenv("VISIONSERVE_EP", "tensorrt,cuda")
	}
}

// defaultPrompt fills in a prompt for models that cannot run without one, and says so.
func defaultPrompt(q *api.PredictJSONRequest, task, arch string) string {
	if q.Prompt != "" || q.Box != "" || q.Point != "" || q.TemplateName != "" {
		return ""
	}
	switch {
	case task == string(api.TaskOpenVocab):
		q.Prompt = "person. car. dog. cat."
		return `default text prompt "person. car. dog. cat." (pass --prompt to bench your own; latency grows with prompt length)`
	case task == string(api.TaskSegmentation) && isSAM(arch):
		return "default box prompt: the centre half of each image (pass --box or --point to change it)"
	}
	return ""
}

func isSAM(arch string) bool {
	switch arch {
	case "mobile-sam", "efficient-sam", "nano-sam", "sam2":
		return true
	}
	return false
}

// centreBox is the default SAM prompt for an image of w x h.
func centreBox(w, h int) string {
	return fmt.Sprintf("%d,%d,%d,%d", w/4, h/4, w/2, h/2)
}

func executeBench(ctx context.Context, o *benchOptions, progress io.Writer) (*benchResult, error) {
	res := &benchResult{opts: o, devices: map[string]int{}, coldMs: math.NaN(), firstMs: math.NaN()}
	res.gpuKind = probeGPU()

	var target benchTarget
	base := o.req.PredictJSONRequest
	if o.serverURL == "" {
		applyBenchEP(o.ep)
		req, tmpl, err := o.req.request(o.model)
		if err != nil {
			return nil, clireport.Usagef("%v", err)
		}
		base = req.PredictJSONRequest
		reg := registry.New(modelsDir(o.modelsDir))
		warns, err := reg.Scan()
		if err != nil {
			return nil, clireport.Usagef("%v", err)
		}
		e, ok := reg.Get(o.model)
		if !ok {
			for _, w := range warns {
				fmt.Fprintf(progress, "registry warning: %v\n", w)
			}
			return nil, clireport.Usagef("model %q not found in registry (%s)", o.model, reg.Root())
		}
		res.task, res.arch = e.Manifest.Task, e.Manifest.Architecture
		res.manifestEPs = e.Manifest.Runtime.Prefer
		if chain, err := engine.ResolveProviders(e.Manifest.Runtime.Prefer); err == nil {
			for _, p := range chain {
				res.epChain = append(res.epChain, string(p))
			}
		}
		mgr := lifecycle.NewManager(reg)
		if tmpl == nil {
			tmpl = templates.New()
		}
		mgr.SetTemplateStore(tmpl)
		target = &inProcessTarget{model: o.model, mgr: mgr}
	} else {
		st := &serverTarget{base: strings.TrimRight(o.serverURL, "/"), model: o.model, reload: o.reload,
			client: &http.Client{Timeout: 10 * time.Minute}}
		infos, err := st.models(ctx)
		if err != nil {
			return nil, clireport.Usagef("cannot reach the server at %s: %v", st.base, err)
		}
		for _, in := range infos {
			if in.Name == o.model {
				res.task = string(in.Task)
			}
		}
		// The architecture decides the default SAM prompt; the server does not report it, so it
		// is read from the local registry when that has the model (same name, same files).
		reg := registry.New(modelsDir(o.modelsDir))
		if _, err := reg.Scan(); err == nil {
			if e, ok := reg.Get(o.model); ok {
				res.arch = e.Manifest.Architecture
			}
		}
		target = st
	}
	defer target.close()
	res.where = target.where()
	res.promptNote = defaultPrompt(&base, res.task, res.arch)

	imgs, desc, err := loadBenchImages(o.imagesDir, o.size, o.serverURL != "")
	if err != nil {
		return nil, err
	}
	res.images = desc
	reqFor := func(img benchImage) api.PredictJSONRequest {
		q := base
		if res.promptNote != "" && isSAM(res.arch) && q.Box == "" {
			q.Box = centreBox(img.w, img.h)
		}
		return q
	}

	fmt.Fprintf(progress, "bench: loading %s (%s) ...\n", o.model, res.where)
	cold, note, err := target.load(ctx)
	if err != nil {
		if clireport.ExitCode(err) == clireport.ExitUsage {
			return nil, err
		}
		return nil, clireport.Usagef("loading %s failed: %v", o.model, err)
	}
	res.coldMs, res.coldNote = cold, note

	for i := 0; i < o.warmup; i++ {
		img := imgs[i%len(imgs)]
		s := target.predict(ctx, img, reqFor(img))
		if s.err != nil {
			if errors.Is(s.err, models.ErrBadPrompt) {
				return nil, clireport.Usagef("warm-up request failed: %v\n(this model needs a prompt: pass --prompt, --box or --point)", s.err)
			}
			return nil, clireport.Usagef("warm-up request failed: %v\n(if the model needs a prompt, pass --prompt, --box or --point)", s.err)
		}
		if i == 0 {
			res.firstMs = s.e2eMs
		}
		res.devices[s.device]++
	}

	res.pid = target.pid()
	sampler := startMemSampler(res.pid, res.gpuKind, 500*time.Millisecond)
	fmt.Fprintf(progress, "bench: %d requests at concurrency %d ...\n", o.requests, o.concurrency)
	res.samples = make([]benchSample, o.requests)
	jobs := make(chan int)
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < o.concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				img := imgs[i%len(imgs)]
				res.samples[i] = target.predict(ctx, img, reqFor(img))
			}
		}()
	}
	for i := 0; i < o.requests; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	res.wall = time.Since(start)
	res.rss, res.rssMax, res.gpu = sampler.finish()
	if _, peak := processRSS(res.pid); peak > res.rssMax {
		res.rssMax = peak // VmHWM: the true peak since the process started
	}
	res.devices = map[string]int{}
	for _, s := range res.samples {
		if s.err == nil {
			res.devices[s.device]++
		}
	}
	return res, nil
}

// ---- report ------------------------------------------------------------------------------

func (r *benchResult) ok() (e2e, srv []float64, errs []error) {
	for _, s := range r.samples {
		if s.err != nil {
			errs = append(errs, s.err)
			continue
		}
		e2e = append(e2e, s.e2eMs)
		srv = append(srv, s.serverMs)
	}
	return
}

// mainDevice is the device most successful requests reported.
func (r *benchResult) mainDevice() string {
	best, n := "", -1
	for d, c := range r.devices {
		if c > n || (c == n && d < best) {
			best, n = d, c
		}
	}
	if best == "" {
		return "unknown"
	}
	return best
}

// verdict decides PASS / WARN / FAIL and the one-sentence reason.
func (r *benchResult) verdict() (clireport.Verdict, string, []string) {
	e2e, _, errs := r.ok()
	st := statsOf(e2e)
	dev := r.mainDevice()
	tput := float64(len(e2e)) / r.wall.Seconds()
	head := fmt.Sprintf("p50 %s ms, %s req/s on %s", trimFloat(st.P50), trimFloat(tput), dev)
	var warns []string
	if len(errs) > 0 {
		return clireport.Fail, fmt.Sprintf("%d of %d requests failed (first: %s)", len(errs), len(r.samples),
			truncate(errs[0].Error(), 160)), nil
	}
	if len(r.devices) > 1 {
		warns = append(warns, "requests ran on different devices: "+deviceCounts(r.devices))
	}
	if strings.HasPrefix(dev, "mixed(") {
		warns = append(warns, "the session pool runs on different execution providers ("+dev+")")
	}
	onCPU := dev == "cpu"
	switch r.opts.ep {
	case "cuda":
		if onCPU {
			warns = append(warns, "CUDA was requested but the model ran on the CPU")
		}
	case "tensorrt":
		if onCPU {
			warns = append(warns, "TensorRT was requested but the model ran on the CPU")
		} else if !strings.Contains(dev, "trt") {
			warns = append(warns, "TensorRT was requested but CUDA ran (libnvinfer.so.10 missing, or the engine failed to build)")
		}
	case "auto":
		if onCPU && r.gpuKind.present() && preferGPU(r.manifestEPs, r.opts.serverURL != "") {
			warns = append(warns, "a GPU is present but the model ran on the CPU")
		}
	}
	if len(warns) > 0 {
		return clireport.Warn, warns[0] + ": " + head, warns
	}
	return clireport.Pass, head, nil
}

// preferGPU: did the manifest ask for a GPU EP? In server mode the chain is unknown; assume the
// shipped default (cuda → cpu).
func preferGPU(prefer []string, server bool) bool {
	if server && len(prefer) == 0 {
		return true
	}
	for _, p := range prefer {
		switch strings.ToLower(strings.TrimSpace(p)) {
		case "cuda", "tensorrt", "coreml", "directml":
			return true
		}
	}
	return false
}

func (r *benchResult) report() *clireport.Report {
	o := r.opts
	e2e, srv, errs := r.ok()
	se, ss := statsOf(e2e), statsOf(srv)
	tput := math.NaN()
	if r.wall > 0 {
		tput = float64(len(e2e)) / r.wall.Seconds()
	}
	verdict, reason, warns := r.verdict()
	dev := r.mainDevice()
	rep := &clireport.Report{Command: "bench", Subject: o.model, Verdict: verdict, Reason: reason}

	devNote := "requested: " + o.ep
	if len(r.epChain) > 0 {
		devNote += "; chain " + strings.Join(r.epChain, " → ")
	}
	coldText, coldVal := "not measured", any(nil)
	if !math.IsNaN(r.coldMs) {
		coldText, coldVal = fmt.Sprintf("%.2f s", r.coldMs/1000), r.coldMs
	}
	e2eLabel := "latency p50 / p95 / p99"
	e2eNote := "end to end in this process: crop, preprocess, inference, postprocess (decoded image in memory)"
	if o.serverURL != "" {
		e2eNote = "end to end over HTTP: multipart upload, decode, preprocess, inference, postprocess, JSON"
	}
	gpuText, gpuNote := fmtMB(r.gpu.ProcessMB), ""
	switch {
	case r.gpu.ProcessMB >= 0:
		gpuNote = "this process (" + r.gpu.Source + ")"
		if o.serverURL != "" {
			gpuNote = fmt.Sprintf("server PID %d (%s)", r.pid, r.gpu.Source)
		}
	case dev == "cpu" && r.gpuKind.NvidiaSMI && !r.gpuKind.Jetson && o.serverURL == "":
		gpuText, gpuNote = "none", "this process holds no GPU memory (it ran on the CPU)"
	case r.gpu.UsedMB >= 0:
		gpuText = fmtMB(r.gpu.UsedMB) + " / " + fmtMB(r.gpu.TotalMB)
		gpuNote = "whole GPU, every process (" + r.gpu.Source + ")"
		if r.gpuKind.Jetson {
			gpuNote = "system RAM in use, shared by CPU and GPU (" + r.gpu.Source + ")"
		}
	case !r.gpuKind.present():
		gpuText, gpuNote = "n/a", "no NVIDIA GPU found (no nvidia-smi, not a Jetson)"
	default:
		gpuText, gpuNote = "n/a", "could not be read"
	}
	rssNote := "this process: current / peak (VmRSS / VmHWM)"
	if o.serverURL != "" {
		rssNote = fmt.Sprintf("server PID %d: current / peak", r.pid)
		if r.pid == 0 {
			rssNote = "server process not found on this machine (remote, containerised or another user's)"
		}
	}
	// Every summary value carries a note on what exactly it measures: the value goes in the summary
	// table, the notes in the "What the numbers measure" section (clireport fields have no note).
	type row struct {
		f    clireport.Field
		note string
	}
	rows := []row{
		{clireport.Field{Key: "device", Label: "device (EP used)", Value: dev}, devNote},
		{clireport.Field{Key: "cold_load_ms", Label: "cold load", Value: coldVal, Text: coldText}, r.coldNote},
		{clireport.Field{Key: "first_request_ms", Label: "first request", Value: nanNil(r.firstMs), Text: msText(r.firstMs)},
			"the first warm-up request: kernel selection and allocations"},
		{clireport.Field{Key: "latency_ms", Label: e2eLabel, Value: sanitizeJSON(se.json()),
			Text: fmt.Sprintf("%s / %s / %s ms", trimFloat(se.P50), trimFloat(se.P95), trimFloat(se.P99))}, e2eNote},
		{clireport.Field{Key: "server_ms", Label: "server-side p50 / p95", Value: sanitizeJSON(ss.json()),
			Text: fmt.Sprintf("%s / %s ms", trimFloat(ss.P50), trimFloat(ss.P95))},
			"the server's own duration_ms: preprocess + inference + postprocess"},
		{clireport.Field{Key: "throughput_rps", Label: "throughput", Value: nanNil(tput),
			Text: fmt.Sprintf("%s req/s at concurrency %d", trimFloat(tput), o.concurrency)},
			"successful requests per second of the timed phase's wall time"},
		{clireport.Field{Key: "errors", Label: "errors", Value: len(errs), Text: fmt.Sprintf("%d / %d", len(errs), len(r.samples))}, ""},
		{clireport.Field{Key: "rss_mb", Label: "process memory (RSS)",
			Value: map[string]any{"current": nanNil(neg(r.rss)), "peak": nanNil(neg(r.rssMax))},
			Text:  fmtMB(r.rss) + " / " + fmtMB(r.rssMax) + " peak"}, rssNote},
		{clireport.Field{Key: "gpu_memory_mb", Label: "GPU memory", Value: map[string]any{"process": nanNil(neg(r.gpu.ProcessMB)),
			"used": nanNil(neg(r.gpu.UsedMB)), "total": nanNil(neg(r.gpu.TotalMB)), "source": r.gpu.Source},
			Text: gpuText}, gpuNote},
	}
	if r.gpu.LoadPct >= 0 {
		rows = append(rows, row{clireport.Field{Key: "gpu_load_pct", Label: "GPU load (peak)", Value: r.gpu.LoadPct,
			Text: fmt.Sprintf("%.0f %%", r.gpu.LoadPct)}, r.gpu.Source})
	}
	var meaning []clireport.Field
	for _, rw := range rows {
		rep.Summary = append(rep.Summary, rw.f)
		if rw.note != "" {
			meaning = append(meaning, clireport.Field{Label: rw.f.Label, Text: rw.note})
		}
	}

	host := fmt.Sprintf("%s/%s, %d CPUs", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	if r.gpuKind.Model != "" {
		host += ", " + r.gpuKind.Model
	}
	if r.gpuKind.VisibleGPUs != "" {
		host += " (CUDA_VISIBLE_DEVICES=" + r.gpuKind.VisibleGPUs + ")"
	}
	setup := []clireport.Field{
		{Label: "model", Text: o.model + nonEmpty(" ("+r.task+", "+r.arch+")", r.task != "")},
		{Label: "mode", Text: r.where},
		{Label: "host", Text: host},
		{Label: "images", Text: r.images},
		{Label: "requests", Text: fmt.Sprintf("%d timed after %d warm-up, concurrency %d, %.2f s wall",
			len(r.samples), o.warmup, o.concurrency, r.wall.Seconds())},
	}
	if r.promptNote != "" {
		setup = append(setup, clireport.Field{Label: "prompt", Text: r.promptNote})
	}
	if len(r.devices) > 0 {
		setup = append(setup, clireport.Field{Label: "devices reported", Text: deviceCounts(r.devices)})
	}
	rep.Sections = []clireport.Section{
		{Title: "Setup", Rows: setup},
		{Title: "Latency (ms)", Table: &clireport.Table{Header: []string{"", "p50", "p95", "p99", "mean", "min", "max"},
			Rows: [][]string{latRow("end to end", se), latRow("server-side", ss)}},
			Lines: []string{"end to end − server-side = what a request costs outside the model call (HTTP and image " +
				"decode with --server; ROI crop and result mapping in-process)"}},
	}
	if len(e2e) > 0 {
		rep.Sections = append(rep.Sections,
			clireport.Section{Title: "Latency distribution", Image: &clireport.Image{
				Caption: "End-to-end latency histogram with p50 / p95 / p99",
				PNG:     histogramChart(e2e, 30, map[string]float64{"p50": se.P50, "p95": se.P95, "p99": se.P99})}},
			clireport.Section{Title: "Latency per request", Image: &clireport.Image{
				Caption: "Each request in completion order (drift = warm-up, throttling or other load)",
				PNG:     timelineChart(e2e)}})
	}
	rep.Sections = append(rep.Sections, clireport.Section{Title: "What the numbers measure", Rows: meaning})

	// Findings: the verdict's warnings and errors, then remarks.
	for _, w := range warns {
		rep.Add(clireport.Warn, "%s", w)
	}
	if len(errs) > 0 {
		seen := map[string]bool{}
		for _, e := range errs {
			if m := truncate(e.Error(), 200); !seen[m] && len(seen) < 5 {
				seen[m] = true
				rep.Add(clireport.Fail, "request failed: %s", m)
			}
		}
	}
	if strings.Contains(dev, "trt") {
		rep.Add(clireport.Info, "%s. Faster is not the same as as-accurate: check this model's accuracy under "+
			"TensorRT before deploying it. The first request includes the engine build (cached afterwards).", trtCaveat)
	}
	rep.Add(clireport.Info, "Numbers are for THIS machine, power mode and load; a shared or busy GPU inflates and jitters them.")
	rep.NextSteps = r.nextSteps(se, dev)

	samples := make([]any, 0, len(r.samples))
	for _, s := range r.samples {
		if s.err == nil {
			samples = append(samples, []float64{round3(s.e2eMs), round3(s.serverMs)})
		}
	}
	rep.Details = sanitizeJSON(map[string]any{
		"model": o.model, "task": r.task, "architecture": r.arch, "mode": r.where, "host": host,
		"images": r.images, "prompt": r.promptNote, "requests": len(r.samples), "warmup": o.warmup,
		"concurrency": o.concurrency, "ep_requested": o.ep, "ep_chain": r.epChain, "devices": r.devices,
		"wall_s": r.wall.Seconds(), "jetson": r.gpuKind.Jetson, "gpu": r.gpuKind.Model,
		"cuda_visible_devices": r.gpuKind.VisibleGPUs, "warnings": warns,
		"samples_ms": map[string]any{"columns": []string{"end_to_end", "server"}, "rows": samples},
		"version":    Version,
	})
	return rep
}

func (r *benchResult) nextSteps(se latencyStats, dev string) []string {
	o := r.opts
	var out []string
	if dev == "cpu" && r.gpuKind.present() {
		out = append(out, "The model ran on the CPU. Check that ORT_DYLIB_PATH points to a GPU build of ONNX Runtime "+
			"(its directory must hold libonnxruntime_providers_cuda.so) and rerun with VISIONSERVE_TRACE=1 to see why CUDA did not load.")
	}
	if r.gpuKind.Jetson {
		out = append(out, "Jetson: results depend on the power mode. Check it with `sudo nvpmodel -q`; for stable numbers use the "+
			"highest mode and `sudo jetson_clocks`, and say which mode you measured in.")
	}
	if o.concurrency == 1 {
		out = append(out, fmt.Sprintf("Throughput under parallel load: `visionserve bench %s --concurrency 4`.", o.model))
	}
	if se.P99 > 2*se.P50 && se.N >= 20 {
		out = append(out, "p99 is more than twice p50: look for other load on the GPU/CPU, thermal throttling, or kernel tuning "+
			"for each new input size (the first request of every new image size is slow; raise --warmup, or bench with "+
			"photos at your camera's resolution).")
	}
	if o.imagesDir == "" {
		out = append(out, "This used a synthetic image; detection-heavy models do more work on real photos. Pass --images DIR.")
	}
	out = append(out, fmt.Sprintf("Smaller / faster variants: `visionserve optimize %s --target jetson-orin --images DIR` on a PC, "+
		"then bench the result here.", o.model))
	return out
}

func latRow(name string, s latencyStats) []string {
	return []string{name, trimFloat(s.P50), trimFloat(s.P95), trimFloat(s.P99), trimFloat(s.Mean), trimFloat(s.Min), trimFloat(s.Max)}
}

func msText(v float64) string {
	if math.IsNaN(v) {
		return "n/a"
	}
	return trimFloat(v) + " ms"
}

func nanNil(v float64) any {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	return v
}

func neg(v float64) float64 {
	if v < 0 {
		return math.NaN()
	}
	return v
}

func round3(v float64) float64 { return math.Round(v*1000) / 1000 }

func nonEmpty(s string, ok bool) string {
	if ok {
		return s
	}
	return ""
}

// deviceCounts renders {"gpu:0": 48, "cpu": 2} as "gpu:0 ×48, cpu ×2" (most first).
func deviceCounts(m map[string]int) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if m[keys[i]] != m[keys[j]] {
			return m[keys[i]] > m[keys[j]]
		}
		return keys[i] < keys[j]
	})
	parts := make([]string, len(keys))
	for i, k := range keys {
		name := k
		if name == "" {
			name = "unreported"
		}
		parts[i] = fmt.Sprintf("%s ×%d", name, m[k])
	}
	return strings.Join(parts, ", ")
}
