package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"visionserve/internal/cli/clireport"
	"visionserve/pkg/api"
)

func benchExit(err error) int { return clireport.ExitCode(err) }

func TestParseBenchArgs(t *testing.T) {
	o, err := parseBenchArgs([]string{"rf-detr", "--requests", "10", "--concurrency", "3", "--ep", "cuda"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if o.model != "rf-detr" || o.requests != 10 || o.concurrency != 3 || o.ep != "cuda" || o.warmup != 3 {
		t.Fatalf("parsed %+v", o)
	}
	// flags before the positional work too
	if o, err = parseBenchArgs([]string{"--warmup", "0", "m"}, io.Discard); err != nil || o.model != "m" || o.warmup != 0 {
		t.Fatalf("got %+v, %v", o, err)
	}
	bad := [][]string{
		{},                               // no model
		{"a", "b"},                       // two models
		{"m", "--ep", "rocm"},            // unknown EP
		{"m", "--requests", "0"},         // bounds
		{"m", "--concurrency", "100000"}, // bounds
		{"m", "--warmup", "-1"},          // bounds
		{"m", "--images", "d", "--size", "64x64"}, // alternatives
		{"m", "--server", "http://x", "--in-process"},
		{"m", "--reload"},            // needs --server
		{"m", "--server", "ftp://x"}, // not http
		{"m", "--size", "10x10"},     // too small
		{"m", "--size", "640"},       // not WxH
		{"m", "--server", "http://x", "--depth", "d.raw"},
		{"m", "--bogus"},
	}
	for _, a := range bad {
		if _, err := parseBenchArgs(a, io.Discard); benchExit(err) != 2 {
			t.Errorf("%v: want a usage error (exit 2), got %v", a, err)
		}
	}
}

// Reference values: numpy.percentile(np.arange(1, 11), [50, 95, 99]) = 5.5, 9.55, 9.91.
func TestPercentileMatchesNumpy(t *testing.T) {
	s := []float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	st := statsOf(s)
	for _, c := range []struct{ got, want float64 }{{st.P50, 5.5}, {st.P95, 9.55}, {st.P99, 9.91}, {st.Mean, 5.5}, {st.Min, 1}, {st.Max, 10}} {
		if math.Abs(c.got-c.want) > 1e-9 {
			t.Errorf("got %v want %v", c.got, c.want)
		}
	}
	if !math.IsNaN(statsOf(nil).P50) || statsOf([]float64{7}).P99 != 7 {
		t.Error("empty / single-sample percentiles")
	}
}

func fakeResult(ep, dev string, gpu gpuKind, prefer []string, errs int) *benchResult {
	r := &benchResult{opts: &benchOptions{model: "m", ep: ep, concurrency: 1}, devices: map[string]int{},
		gpuKind: gpu, manifestEPs: prefer, wall: time.Second, coldMs: 100, firstMs: 50}
	for i := 0; i < 10; i++ {
		s := benchSample{e2eMs: float64(10 + i), serverMs: float64(8 + i), device: dev}
		if i < errs {
			s = benchSample{err: errors.New("boom")}
		}
		r.samples = append(r.samples, s)
		if s.err == nil {
			r.devices[dev]++
		}
	}
	return r
}

func TestBenchVerdict(t *testing.T) {
	gpu := gpuKind{NvidiaSMI: true}
	none := gpuKind{}
	cases := []struct {
		name, ep, dev string
		gpu           gpuKind
		prefer        []string
		errs          int
		want          clireport.Verdict
	}{
		{"gpu as asked", "cuda", "gpu:0", gpu, []string{"cuda", "cpu"}, 0, clireport.Pass},
		{"any error fails", "auto", "gpu:0", gpu, []string{"cuda", "cpu"}, 1, clireport.Fail},
		{"cuda asked, cpu ran", "cuda", "cpu", gpu, []string{"cuda", "cpu"}, 0, clireport.Warn},
		{"trt asked, cuda ran", "tensorrt", "gpu:0", gpu, nil, 0, clireport.Warn},
		{"trt asked and ran", "tensorrt", "gpu:0+trt", gpu, nil, 0, clireport.Pass},
		{"auto, GPU present, manifest wants cuda, cpu ran", "auto", "cpu", gpu, []string{"cuda", "cpu"}, 0, clireport.Warn},
		{"auto, no GPU on the host", "auto", "cpu", none, []string{"cuda", "cpu"}, 0, clireport.Pass},
		{"auto, cpu-only manifest", "auto", "cpu", gpu, []string{"cpu"}, 0, clireport.Pass},
		{"cpu asked", "cpu", "cpu", gpu, []string{"cuda", "cpu"}, 0, clireport.Pass},
		{"jetson counts as a GPU", "auto", "cpu", gpuKind{Jetson: true}, []string{"cuda"}, 0, clireport.Warn},
	}
	for _, c := range cases {
		v, reason, _ := fakeResult(c.ep, c.dev, c.gpu, c.prefer, c.errs).verdict()
		if v != c.want {
			t.Errorf("%s: verdict %s (%s), want %s", c.name, v, reason, c.want)
		}
	}
	v, reason, _ := fakeResult("cuda", "gpu:0", gpu, nil, 0).verdict()
	if v != clireport.Pass || !strings.Contains(reason, "p50 14.5 ms") || !strings.Contains(reason, "on gpu:0") {
		t.Errorf("headline: %s %s", v, reason)
	}
}

func TestReportJSONIsOneObjectWithTheSharedKeys(t *testing.T) {
	r := fakeResult("cuda", "gpu:0", gpuKind{}, nil, 0)
	r.coldMs = math.NaN() // not measured -> null, not a JSON error
	raw, err := r.report().JSON()
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	dec := json.NewDecoder(bytes.NewReader(raw))
	if err := dec.Decode(&doc); err != nil {
		t.Fatal(err)
	}
	if dec.More() {
		t.Fatal("more than one JSON value on stdout")
	}
	for _, k := range []string{"verdict", "reason", "summary", "details"} {
		if _, ok := doc[k]; !ok {
			t.Errorf("missing %q", k)
		}
	}
	if len(doc) != 4 {
		t.Errorf("unexpected keys: %v", doc)
	}
	sum := doc["summary"].(map[string]any)
	if sum["cold_load_ms"] != nil || sum["device"] != "gpu:0" {
		t.Errorf("summary %v", sum)
	}
}

func TestReportTextAndHTML(t *testing.T) {
	rep := fakeResult("cuda", "cpu", gpuKind{NvidiaSMI: true}, []string{"cuda"}, 0).report()
	var txt bytes.Buffer
	if err := rep.WriteText(&txt); err != nil {
		t.Fatal(err)
	}
	first, _, _ := strings.Cut(txt.String(), "\n")
	if !strings.HasPrefix(first, "WARN: CUDA was requested but the model ran on the CPU") {
		t.Errorf("first line %q", first)
	}
	var h bytes.Buffer
	if err := rep.WriteHTML(&h); err != nil {
		t.Fatal(err)
	}
	s := h.String()
	for _, want := range []string{`class="verdict warn"`, "prefers-color-scheme: dark", "data:image/png;base64,",
		`id="verdict"`, `id="summary"`, `id="details"`, `id="next-steps"`, "What the numbers measure"} {
		if !strings.Contains(s, want) {
			t.Errorf("HTML lacks %q", want)
		}
	}
	// self-contained: nothing fetched from elsewhere
	for _, bad := range []string{"<script", "<link", `src="http`} {
		if strings.Contains(s, bad) {
			t.Errorf("HTML is not self-contained: %q", bad)
		}
	}
}

func TestProbeParsers(t *testing.T) {
	apps := "1234, 796\n99, 10\n1234, 4\n"
	if got := parseComputeApps(apps, 1234); got != 800 {
		t.Errorf("compute apps: %v", got)
	}
	if got := parseComputeApps("1234, [N/A]\n", 1234); got != -1 {
		t.Errorf("an iGPU's [N/A] must read as unknown, got %v", got)
	}
	mem := "0, 9802, 49140\n3, 339, 49140\n"
	if u, tot := parseGPUMemory(mem, "3"); u != 339 || tot != 49140 {
		t.Errorf("visible GPU 3: %v %v", u, tot)
	}
	if u, _ := parseGPUMemory(mem, ""); u != 10141 {
		t.Errorf("all GPUs: %v", u)
	}
	if u, _ := parseGPUMemory("0, [N/A], [N/A]\n", ""); u != -1 {
		t.Errorf("N/A memory: %v", u)
	}
	orin := "RAM 7712/30536MB (lfb 4x4MB) SWAP 0/15268MB (cached 0MB) CPU [2%@729,0%@729] GR3D_FREQ 45% cpu@48C"
	if u, tot, l := parseTegrastats(orin); u != 7712 || tot != 30536 || l != 45 {
		t.Errorf("orin: %v %v %v", u, tot, l)
	}
	// JetPack 7.0 on Thor prints the GPC clocks only: the load is unknown, not 0.
	thor := "RAM 9000/125772MB (lfb 2x4MB) GR3D_FREQ @[314,314,314] cpu@40C"
	if u, _, l := parseTegrastats(thor); u != 9000 || l != -1 {
		t.Errorf("thor: %v %v", u, l)
	}
	thor71 := "RAM 9000/125772MB GR3D_FREQ 99%@[1098,1098,1098]"
	if _, _, l := parseTegrastats(thor71); l != 99 {
		t.Errorf("thor 7.1: %v", l)
	}
	table := "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n" +
		"   0: 0100007F:2DB2 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1004        0 5551234 1 0\n" +
		"   1: 0100007F:2DB2 0100007F:9C40 01 00000000:00000000 00:00000000 00000000  1004        0 5559999 1 0\n"
	if got := listenInodes(table, 11698); len(got) != 1 || got[0] != "5551234" {
		t.Errorf("listen inodes: %v", got)
	}
	if visibleSet("GPU-abc") != nil || len(visibleSet("1,3")) != 2 {
		t.Error("visibleSet")
	}
}

func TestMultipartBodyCarriesTheOptions(t *testing.T) {
	q := api.PredictJSONRequest{Model: "m", Prompt: "cat.", BoxThreshold: 0.3}
	body, ctype, err := multipartBody(benchImage{name: "a.jpg", raw: []byte("JPEG")}, q)
	if err != nil {
		t.Fatal(err)
	}
	_, params, _ := mime.ParseMediaType(ctype)
	mr := multipart.NewReader(body, params["boundary"])
	got := map[string]string{}
	for {
		p, err := mr.NextPart()
		if err != nil {
			break
		}
		b, _ := io.ReadAll(p)
		got[p.FormName()] = string(b)
	}
	if got["model"] != "m" || got["prompt"] != "cat." || got["box_threshold"] != "0.3" || got["image"] != "JPEG" {
		t.Errorf("parts %v", got)
	}
	if _, ok := got["min_size"]; ok {
		t.Error("zero options must not be sent")
	}
}

// A whole server-mode run against a fake server: load, warm-up, concurrent timed requests, the
// device the server reports, and a FAIL when the server errors.
func TestBenchAgainstAFakeServer(t *testing.T) {
	var loads, predicts atomic.Int32
	var failAfter atomic.Int32
	failAfter.Store(1 << 30)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models":
			_ = json.NewEncoder(w).Encode([]api.ModelInfo{{Name: "det", Task: api.TaskDetection, State: "available"}})
		case "/api/load":
			loads.Add(1)
		case "/api/predict":
			if err := r.ParseMultipartForm(1 << 20); err != nil || r.FormValue("model") != "det" {
				http.Error(w, "bad form", http.StatusBadRequest)
				return
			}
			if predicts.Add(1) > failAfter.Load() {
				http.Error(w, `{"error":"boom"}`, http.StatusInternalServerError)
				return
			}
			_ = json.NewEncoder(w).Encode(api.Result{Model: "det", Device: "gpu:0", DurationMs: 2})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	empty := t.TempDir()
	run := func() (*benchResult, error) {
		o, err := parseBenchArgs([]string{"det", "--server", srv.URL, "--requests", "12", "--concurrency", "3",
			"--size", "64x48", "--models", empty}, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		return executeBench(context.Background(), o, io.Discard)
	}
	res, err := run()
	if err != nil {
		t.Fatal(err)
	}
	if loads.Load() != 1 || predicts.Load() != 15 {
		t.Errorf("loads %d predicts %d (want 1 and 3 warm-up + 12)", loads.Load(), predicts.Load())
	}
	if v, reason, _ := res.verdict(); v != clireport.Pass || !strings.Contains(reason, "gpu:0") {
		t.Errorf("verdict %s %s", v, reason)
	}
	if res.mainDevice() != "gpu:0" || math.IsNaN(res.coldMs) {
		t.Errorf("device %s cold %v", res.mainDevice(), res.coldMs)
	}
	failAfter.Store(predicts.Load() + 5 + 3) // let the warm-up pass, then fail some timed requests
	res, err = run()
	if err != nil {
		t.Fatal(err)
	}
	if v, _, _ := res.verdict(); v != clireport.Fail {
		t.Errorf("errors must FAIL, got %s", v)
	}
}

func TestBenchUnknownModelOnServerIsUsageError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]api.ModelInfo{})
	}))
	defer srv.Close()
	o, err := parseBenchArgs([]string{"nope", "--server", srv.URL, "--models", t.TempDir()}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := executeBench(context.Background(), o, io.Discard); benchExit(err) != 2 {
		t.Errorf("want exit 2, got %v", err)
	}
}

func TestLoadBenchImagesBounds(t *testing.T) {
	dir := t.TempDir()
	if _, _, err := loadBenchImages(dir, "", false); benchExit(err) != 2 {
		t.Errorf("empty dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "x.jpg"), []byte("not an image"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := loadBenchImages(dir, "", false); benchExit(err) != 2 {
		t.Errorf("corrupt image: %v", err)
	}
	imgs, desc, err := loadBenchImages("", "", true)
	if err != nil || len(imgs) != 1 || imgs[0].w != 640 || len(imgs[0].raw) == 0 || !strings.Contains(desc, "synthetic") {
		t.Errorf("default synthetic image: %v %v %q", err, len(imgs), desc)
	}
}

func TestDefaultPrompt(t *testing.T) {
	q := api.PredictJSONRequest{}
	if defaultPrompt(&q, "open_vocab", "grounding-dino") == "" || q.Prompt == "" {
		t.Error("open-vocab models get a text prompt")
	}
	q = api.PredictJSONRequest{}
	if defaultPrompt(&q, "segmentation", "mobile-sam") == "" || q.Prompt != "" {
		t.Error("SAM gets a (per-image) box, not text")
	}
	q = api.PredictJSONRequest{Box: "1,2,3,4"}
	if defaultPrompt(&q, "segmentation", "mobile-sam") != "" {
		t.Error("a user prompt is never replaced")
	}
	if defaultPrompt(&api.PredictJSONRequest{}, "segmentation", "background") != "" {
		t.Error("the background model runs without a prompt")
	}
	if centreBox(640, 480) != "160,120,320,240" {
		t.Error(centreBox(640, 480))
	}
}

func TestTuneArgsAndDocker(t *testing.T) {
	if _, err := parseTuneArgs([]string{"--images", "x"}); benchExit(err) != 2 {
		t.Errorf("model must come first: %v", err)
	}
	ta, err := parseTuneArgs([]string{"rf-detr", "--python=/usr/bin/python3", "--models", "/m", "--gpu", "--target", "jetson-orin"})
	if err != nil || ta.python != "/usr/bin/python3" || ta.models != "/m" || !ta.gpu {
		t.Fatalf("%+v %v", ta, err)
	}
	if strings.Join(ta.pass, " ") != "rf-detr --gpu --target jetson-orin" {
		t.Errorf("pass-through %v", ta.pass)
	}
	imgs := t.TempDir()
	out := filepath.Join(t.TempDir(), "sub", "r.html")
	ta = tuneArgs{image: "img:1", gpu: true, pass: []string{"m", "--images", imgs, "--report=" + out, "--target", "cpu"}}
	args, err := buildTuneDockerArgs("optimize", ta, "/reg", "/cache", 1000, 1000)
	if err != nil {
		t.Fatal(err)
	}
	s := strings.Join(args, " ")
	for _, want := range []string{"--gpus all", "type=bind,src=" + imgs + ",dst=/in/1,readonly",
		"type=bind,src=" + filepath.Dir(out) + ",dst=/out/2", "img:1 optimize m --images /in/1 --report /out/2/r.html --target cpu --models /root/.models"} {
		if !strings.Contains(s, want) {
			t.Errorf("docker args lack %q:\n%s", want, s)
		}
	}
	ta = tuneArgs{image: "i", pass: []string{"m", "--labels", "/does/not/exist.json"}}
	if _, err := buildTuneDockerArgs("optimize", ta, "/reg", "/cache", -1, -1); benchExit(err) != 2 {
		t.Errorf("missing input: %v", err)
	}
}

func TestChartsAreValidPNGs(t *testing.T) {
	for name, b := range map[string][]byte{
		"hist":     histogramChart([]float64{1, 2, 2, 3, 10}, 5, map[string]float64{"p50": 2}),
		"timeline": timelineChart([]float64{1, 2, 3}),
		"empty":    histogramChart(nil, 5, nil),
	} {
		if !bytes.HasPrefix(b, []byte("\x89PNG")) {
			t.Errorf("%s: not a PNG", name)
		}
	}
}
