package cli

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestParseCheckArgsSplitsWrapperFlags(t *testing.T) {
	o, err := parseCheckArgs([]string{"--images", "/p", "rf-detr", "--server=http://h:1", "--json",
		"--image", "img:x", "--labels=/l.json", "--models", "/m", "--max-images", "3"})
	if err != nil {
		t.Fatal(err)
	}
	if o.model != "rf-detr" || o.server != "http://h:1" || o.image != "img:x" || o.models != "/m" {
		t.Fatalf("wrapper flags: %+v", o)
	}
	want := "--images /p rf-detr --json --labels /l.json --max-images 3"
	if got := strings.Join(o.pass, " "); got != want {
		t.Fatalf("forwarded %q, want %q", got, want)
	}
}

func TestParseCheckArgsRefusesBadUsage(t *testing.T) {
	for name, args := range map[string][]string{
		"no model":       {"--images", "/p"},
		"two models":     {"a", "b", "--images", "/p"},
		"missing value":  {"rf-detr", "--images"},
		"flag eats name": {"--labels", "rf-detr"},
	} {
		if _, err := parseCheckArgs(args); err == nil {
			t.Errorf("%s: %v accepted", name, args)
		}
	}
}

func TestBuildCheckDockerArgsMounts(t *testing.T) {
	dir := t.TempDir()
	photos := filepath.Join(dir, "photos")
	labels := filepath.Join(dir, "ann", "val.json")
	for _, p := range []string{filepath.Join(photos, "a.jpg"), labels} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	report := filepath.Join(dir, "out.html")
	args, err := buildCheckDockerArgs([]string{"rf-detr", "--images", photos, "--labels", labels, "--json",
		"--report", report}, "img:tag", "/models", "/cache", "http://localhost:11720", 1000, 1001)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(args, " ")
	for _, want := range []string{
		"run --rm --network host --user 1000:1001",
		"--mount type=bind,src=/models,dst=/root/.models,readonly",
		"--mount type=bind,src=" + dir + ",dst=/in/1,readonly",
		"--mount type=bind,src=" + filepath.Join(dir, "ann") + ",dst=/in/2,readonly",
		"--mount type=bind,src=" + dir + ",dst=/out/3 ",
		"img:tag check rf-detr --images /in/1/photos --labels /in/2/val.json --json --report /out/3/out.html " +
			"--models /root/.models --server http://localhost:11720",
	} {
		if !strings.Contains(got+" ", want) {
			t.Errorf("docker args lack %q:\n%s", want, got)
		}
	}
	if !strings.Contains(got, `VISIONSERVE_CHECK_PATHS={"/in/1":"`+dir+`","/in/2":"`+filepath.Join(dir, "ann")+
		`","/out/3":"`+dir+`","/root/.models":"/models"}`) {
		t.Errorf("the container->host path map is missing:\n%s", got)
	}
	if strings.Contains(got, "dst=/out/3,readonly") {
		t.Errorf("the report directory must be writable:\n%s", got)
	}
}

func TestBuildCheckDockerArgsRefusesMissingInputs(t *testing.T) {
	if _, err := buildCheckDockerArgs([]string{"m", "--images", "/nope/photos"}, "i", "/m", "/c", "u", 1, 1); err == nil {
		t.Error("a missing --images must be refused before docker runs")
	}
	if _, err := buildCheckDockerArgs([]string{"m", "--report", "/nope/dir/r.html"}, "i", "/m", "/c", "u", 1, 1); err == nil {
		t.Error("a --report in a missing directory must be refused")
	}
}

func fakeServer(t *testing.T, models string, loadStatus int) string {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"status":"ok"}`)) })
	mux.HandleFunc("GET /api/models", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(models)) })
	mux.HandleFunc("POST /api/load", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(loadStatus) })
	s := httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return s.URL
}

func exitCode(err error) int {
	var e *ExitError
	if errors.As(err, &e) {
		return e.Code
	}
	return -1
}

func TestCheckServer(t *testing.T) {
	listed := fakeServer(t, `[{"name":"rf-detr"}]`, http.StatusNotFound)
	if err := checkServer(listed, "rf-detr", "/m", time.Second); err != nil {
		t.Fatalf("listed model: %v", err)
	}
	err := checkServer(listed, "other", "/m", time.Second)
	if exitCode(err) != 2 || !strings.Contains(err.Error(), "does not serve") || !strings.Contains(err.Error(), "--models /m") {
		t.Fatalf("unknown model: %v", err)
	}
	// Installed after the server started: not listed yet, but /api/load finds it.
	late := fakeServer(t, `[]`, http.StatusOK)
	if err := checkServer(late, "new", "/m", time.Second); err != nil {
		t.Fatalf("model found by /api/load: %v", err)
	}
	down := httptest.NewServer(http.NotFoundHandler())
	url := down.URL
	down.Close()
	err = checkServer(url, "rf-detr", "/reg", time.Second)
	if exitCode(err) != 2 || !strings.Contains(err.Error(), "visionserve serve --models /reg") {
		t.Fatalf("no server must say how to start one: %v", err)
	}
}

func TestCheckExitCodes(t *testing.T) {
	if checkExit(nil, "i") != nil {
		t.Fatal("exit 0 is success")
	}
	for code, want := range map[int]int{1: 1, 2: 2, 125: 2} {
		err := exec.Command("sh", "-c", "exit "+strconv.Itoa(code)).Run()
		if got := exitCode(checkExit(err, "img")); got != want {
			t.Errorf("container exit %d -> %d, want %d", code, got, want)
		}
	}
}

func TestRunCheckUsageErrorsExit2(t *testing.T) {
	if err := runCheck([]string{"--images", "/p"}); exitCode(err) != 2 {
		t.Fatalf("missing model: %v", err)
	}
	if err := runCheck([]string{"--help"}); err != nil {
		t.Fatalf("--help: %v", err)
	}
	if err := Execute([]string{"visionserve", "check", "-h"}); err != nil {
		t.Fatalf("Execute dispatches check: %v", err)
	}
}

func TestCheckModelsDirMustExist(t *testing.T) {
	if _, err := checkModelsDir(filepath.Join(t.TempDir(), "nope")); exitCode(err) != 2 {
		t.Fatalf("missing registry: %v", err)
	}
	d := t.TempDir()
	if got, err := checkModelsDir(d); err != nil || got != d {
		t.Fatalf("checkModelsDir(%s) = %s, %v", d, got, err)
	}
}

// --gpu is a wrapper flag: not forwarded as such, it selects the CUDA image, adds --gpus all and
// --device cuda (unless --device was given).
func TestCheckGPUFlag(t *testing.T) {
	o, err := parseCheckArgs([]string{"rf-detr", "--images", "/tmp/x", "--gpu"})
	if err != nil {
		t.Fatal(err)
	}
	if !o.gpu || hasFlag(o.pass, "--gpu") || !hasFlag(o.pass, "--device") {
		t.Fatalf("gpu=%v pass=%v: want gpu, --gpu not forwarded, --device cuda added", o.gpu, o.pass)
	}
	o, err = parseCheckArgs([]string{"rf-detr", "--gpu", "--device", "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, a := range o.pass {
		if a == "--device" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("pass=%v: an explicit --device must not get a second one", o.pass)
	}
	args := withGPU([]string{"run", "--rm", defaultConvertGPUImage})
	if len(args) < 3 || args[1] != "--gpus" || args[2] != "all" {
		t.Fatalf("withGPU gave %v", args)
	}
}

// sensitivity/optimize: --gpu picks the CUDA converter image unless --image or
// $VISIONSERVE_CONVERT_IMAGE says otherwise, and the docker run gets --gpus all.
func TestTuneGPUImage(t *testing.T) {
	t.Setenv("VISIONSERVE_CONVERT_IMAGE", "")
	ta, err := parseTuneArgs([]string{"rf-detr", "--gpu", "--images", "/tmp/x"})
	if err != nil {
		t.Fatal(err)
	}
	if ta.image != defaultConvertGPUImage {
		t.Fatalf("image %q, want %q", ta.image, defaultConvertGPUImage)
	}
	ta, err = parseTuneArgs([]string{"rf-detr", "--images", "/tmp/x"})
	if err != nil || ta.image != defaultConvertImage {
		t.Fatalf("without --gpu: image %q err %v", ta.image, err)
	}
	t.Setenv("VISIONSERVE_CONVERT_IMAGE", "mine:1")
	if ta, _ = parseTuneArgs([]string{"rf-detr", "--gpu"}); ta.image != "mine:1" {
		t.Fatalf("env image overridden by --gpu: %q", ta.image)
	}
}
