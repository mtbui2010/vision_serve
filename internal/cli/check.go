package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ExitError ends the process with Code (cmd/visionserve honours ExitCode()). An empty Msg prints
// nothing: the command already said what happened (e.g. a FAIL verdict on stdout).
type ExitError struct {
	Code int
	Msg  string
}

func (e *ExitError) Error() string { return e.Msg }

// ExitCode is the process exit status: 1 a FAIL verdict, 2 a usage or setup error.
func (e *ExitError) ExitCode() int { return e.Code }

func setupError(format string, a ...any) error {
	return &ExitError{Code: 2, Msg: fmt.Sprintf(format, a...)}
}

const defaultCheckServer = "http://localhost:11435"

// checkValueFlags are the `check` flags that take a value (everything else is boolean, i.e.
// --json): needed to tell the positional <model> from a flag's value.
var checkValueFlags = map[string]bool{
	"--images": true, "--labels": true, "--reference": true, "--checkpoint": true, "--server": true,
	"--models": true, "--report": true, "--max-images": true, "--labels-max": true, "--prompt": true,
	"--device": true, "--max-map-drop": true, "--threshold": true, "--image": true,
}

// checkInputFlags are host paths the converter image reads: mounted read-only. --report is an
// output: its directory is mounted read-write.
var checkInputFlags = map[string]bool{"--images": true, "--labels": true, "--reference": true, "--checkpoint": true}

type checkOpts struct {
	model, image, models, server string
	gpu                          bool     // --gpu: the CUDA converter image with the host's GPUs
	pass                         []string // forwarded to the converter image (without --image/--models/--server/--gpu)
}

// parseCheckArgs splits the wrapper's own flags (--image, --models, --server) from the ones the
// converter's `check` takes, and finds the positional <model>.
func parseCheckArgs(args []string) (checkOpts, error) {
	var o checkOpts
	for i := 0; i < len(args); i++ {
		a := args[i]
		name, val, hasEq := strings.Cut(a, "=")
		if strings.HasPrefix(a, "-") && checkValueFlags[name] && !hasEq {
			if i+1 >= len(args) {
				return o, fmt.Errorf("%s needs a value", name)
			}
			i++
			val = args[i]
		}
		switch {
		case name == "--image":
			o.image = val
		case name == "--models":
			o.models = val
		case name == "--server":
			o.server = val
		case a == "--gpu":
			o.gpu = true
		case !strings.HasPrefix(a, "-"):
			if o.model != "" {
				return o, fmt.Errorf("unexpected argument %q (one model name only; is a flag before it misspelt?)", a)
			}
			o.model = a
			o.pass = append(o.pass, a)
		case checkValueFlags[name]:
			o.pass = append(o.pass, name, val)
		default:
			o.pass = append(o.pass, a)
		}
	}
	if o.model == "" {
		return o, errors.New("missing <model>: the installed model to check (see `visionserve list`)")
	}
	if o.gpu && !hasFlag(o.pass, "--device") { // run the reference model (--checkpoint) on the GPU too
		o.pass = append(o.pass, "--device", "cuda")
	}
	return o, nil
}

// runCheck: visionserve check <model> --images DIR [flags...]
//
// Runs the converter image's `check` (tiers B1/B2/C of `visionserve convert`, on an installed model)
// against a RUNNING server, exactly like `convert` runs the image: host paths mounted (inputs
// read-only, the --report directory read-write, the registry read-only), --network host so the
// container reaches the server on localhost, the calling user's uid/gid. Exit status: 0 PASS or
// WARN, 1 FAIL, 2 usage or setup error.
func runCheck(args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		fmt.Print(checkUsage)
		return nil
	}
	o, err := parseCheckArgs(args)
	if err != nil {
		return setupError("check: %v\nusage: visionserve check <model> --images DIR [flags]  (see visionserve check --help)", err)
	}
	image := o.image
	if image == "" {
		image = os.Getenv("VISIONSERVE_CONVERT_IMAGE")
	}
	if image == "" {
		image = defaultConvertImage
		if o.gpu {
			image = defaultConvertGPUImage // as `convert --gpu` picks it
		}
	}
	dir, err := checkModelsDir(o.models)
	if err != nil {
		return err
	}
	url := checkServerURL(o.server)
	if err := checkServer(url, o.model, dir, 5*time.Second); err != nil {
		return err
	}
	cacheDir := filepath.Join(userHome(), ".cache", "visionserve-convert")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return setupError("check: create cache dir: %v", err)
	}
	dockerArgs, err := buildCheckDockerArgs(o.pass, image, dir, cacheDir, url, os.Getuid(), os.Getgid())
	if err != nil {
		return setupError("check: %v", err)
	}
	if o.gpu {
		dockerArgs = withGPU(dockerArgs)
	}
	if _, err := exec.LookPath("docker"); err != nil {
		return setupError("check needs Docker on this machine (it runs the converter image %s).\n"+
			"Equivalent command to run where Docker is available:\n  docker %s", image, shellJoin(dockerArgs))
	}
	fmt.Fprintf(os.Stderr, "models: %s\n$ docker %s\n", dir, shellJoin(dockerArgs))
	cmd := exec.Command("docker", dockerArgs...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return checkExit(cmd.Run(), image)
}

// checkExit maps the container's exit status to ours. Docker's own failures (125: the daemon or
// `docker run` failed, e.g. no such image; 126/127: the entrypoint could not run) are setup errors.
func checkExit(err error, image string) error {
	if err == nil {
		return nil
	}
	var ee *exec.ExitError
	if !errors.As(err, &ee) {
		return setupError("check: docker: %v", err)
	}
	switch code := ee.ExitCode(); code {
	case 1:
		return &ExitError{Code: 1}
	case 125, 126, 127:
		return setupError("check: docker could not run %s (exit %d). Build it with `make docker-convert` or "+
			"set VISIONSERVE_CONVERT_IMAGE to an image you have", image, code)
	default:
		return &ExitError{Code: 2}
	}
}

// checkModelsDir is the registry to read the manifest from: --models > $VISIONSERVE_MODELS >
// ~/.visionserve/models, or ~/.visionserve_models when only that exists (the Docker layout, as
// convert picks it). Unlike convert it never creates the directory.
func checkModelsDir(flagVal string) (string, error) {
	dir := expandTilde(flagVal)
	if dir == "" && os.Getenv("VISIONSERVE_MODELS") == "" {
		docker := filepath.Join(userHome(), ".visionserve_models")
		if !isDir(defaultModelsDir()) && isDir(docker) {
			dir = docker
		}
	}
	if dir == "" {
		dir = modelsDir("")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", setupError("check: %v", err)
	}
	if !isDir(abs) {
		return "", setupError("check: model registry %s does not exist (pass --models DIR, the registry the server "+
			"serves)", abs)
	}
	return abs, nil
}

func checkServerURL(flagVal string) string {
	u := flagVal
	if u == "" {
		u = os.Getenv("VISIONSERVE_HOST")
	}
	if u == "" {
		u = defaultCheckServer
	}
	if !strings.Contains(u, "://") {
		u = "http://" + u
	}
	return strings.TrimRight(u, "/")
}

// checkServer fails with a setup error (exit 2) that says what to do unless a VisionServe server
// answers at url and serves model. A model missing from /api/models may still be one installed
// after the server started (it re-scans its registry for an unknown name): POST /api/load decides,
// and only its 404 means "not in this server's registry".
func checkServer(url, model, modelsDir string, timeout time.Duration) error {
	c := &http.Client{Timeout: timeout}
	start := fmt.Sprintf("Start one in another terminal, on the registry that holds %q:\n"+
		"  visionserve serve --models %s\nthen re-run this command (or pass --server URL of a running server).",
		model, modelsDir)
	resp, err := c.Get(url + "/api/health")
	if err != nil {
		return setupError("check: no VisionServe server answers at %s (%v).\n%s", url, err, start)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return setupError("check: %s/api/health answered HTTP %d: not a VisionServe server?\n%s", url, resp.StatusCode, start)
	}
	resp, err = c.Get(url + "/api/models")
	if err != nil {
		return setupError("check: GET %s/api/models: %v", url, err)
	}
	var infos []struct {
		Name string `json:"name"`
	}
	err = json.NewDecoder(resp.Body).Decode(&infos)
	resp.Body.Close()
	if err != nil {
		return setupError("check: GET %s/api/models: %v", url, err)
	}
	names := make([]string, 0, len(infos))
	for _, m := range infos {
		if m.Name == model {
			return nil
		}
		names = append(names, m.Name)
	}
	// Loading can take minutes for a large model on CPU: no client timeout on this one.
	body, _ := json.Marshal(map[string]string{"model": model})
	resp, err = (&http.Client{}).Post(url+"/api/load", "application/json", bytes.NewReader(body))
	if err != nil {
		return setupError("check: POST %s/api/load: %v", url, err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		return nil // known to the server (a load error is reported by the check itself)
	}
	shown := strings.Join(names, ", ")
	if len(names) > 8 {
		shown = strings.Join(names[:8], ", ") + " ..."
	}
	if shown == "" {
		shown = "nothing"
	}
	return setupError("check: the server at %s does not serve %q (it lists: %s). Is it installed "+
		"(`visionserve list --models %s`)? If so, this server serves another registry: restart it with "+
		"--models %s, or pass --server URL of the server that serves it.",
		url, model, shown, modelsDir, modelsDir)
}

// buildCheckDockerArgs turns the forwarded `check` arguments into `docker run` arguments: input
// paths mounted read-only under /in/<n>/ (their parent directory), the --report file's directory
// read-write under /out/<n>/, the registry read-only at /root/.models, and the resolved --server.
// uid/gid < 0 (Windows) omit --user.
func buildCheckDockerArgs(pass []string, image, modelsDir, cacheDir, server string, uid, gid int) ([]string, error) {
	run := []string{"run", "--rm", "--network", "host"}
	if uid >= 0 && gid >= 0 {
		run = append(run, "--user", fmt.Sprintf("%d:%d", uid, gid))
	}
	run = append(run, "-e", "HOME=/tmp", "-e", "HF_HOME=/cache/hf",
		"--mount", bindMount(modelsDir, "/root/.models", true),
		"--mount", bindMount(cacheDir, "/cache", false))
	n := 0
	out := []string{"check"}
	// container prefix -> host path, so the report names the user's paths (check.py host_paths)
	paths := map[string]string{"/root/.models": modelsDir}
	for i := 0; i < len(pass); i++ {
		a := pass[i]
		if !checkInputFlags[a] && a != "--report" {
			out = append(out, a)
			continue
		}
		if i+1 >= len(pass) {
			return nil, fmt.Errorf("%s needs a value", a)
		}
		i++
		abs, err := filepath.Abs(expandTilde(pass[i]))
		if err != nil {
			return nil, err
		}
		n++
		if a == "--report" {
			if !isDir(filepath.Dir(abs)) {
				return nil, fmt.Errorf("--report %s: its directory does not exist", pass[i])
			}
			dst := fmt.Sprintf("/out/%d", n)
			run = append(run, "--mount", bindMount(filepath.Dir(abs), dst, false))
			paths[dst] = filepath.Dir(abs)
			out = append(out, a, dst+"/"+filepath.Base(abs))
			continue
		}
		if !exists(abs) {
			return nil, fmt.Errorf("%s %s: no such file or directory", a, pass[i])
		}
		dst := fmt.Sprintf("/in/%d", n)
		run = append(run, "--mount", bindMount(filepath.Dir(abs), dst, true))
		paths[dst] = filepath.Dir(abs)
		out = append(out, a, dst+"/"+filepath.Base(abs))
	}
	pm, err := json.Marshal(paths)
	if err != nil {
		return nil, err
	}
	run = append(run, "-e", "VISIONSERVE_CHECK_PATHS="+string(pm))
	out = append(out, "--models", "/root/.models", "--server", server)
	return append(append(run, image), out...), nil
}

const checkUsage = `visionserve check <model> --images DIR [flags]

Does the served model behave like your training pipeline? Checks an INSTALLED model, served by a
RUNNING server, with the converter's verification tiers, and prints a verdict first:

  B1  preprocessing: the tensor the server feeds the model vs a reference preprocessing of the same
      photos (your --reference script, the --checkpoint's own pipeline, the architecture's known
      recipe, or else the manifest's declared spec), with the likely cause and fix of a difference
  B2  outputs vs the original model (needs --checkpoint or --reference)
  C   accuracy on your labelled photos (needs --labels)

Flags:
  --images DIR          real photos (B1/B2 use up to --max-images of them; C reads its images here)
  --labels FILE         COCO json (detection); folder-per-class dir or CSV image,label (classification)
  --reference FILE.py   your training transform: preprocess(pil) [and predict(pil)]
  --checkpoint PATH     the checkpoint the ONNX was exported from (RF-DETR .pth, HuggingFace dir)
  --server URL          running server (default $VISIONSERVE_HOST or ` + defaultCheckServer + `)
  --models DIR          registry the server serves (default as for convert)
  --report FILE.html    also write a self-contained HTML report (figures included)
  --json                print one JSON object {"verdict","reason","summary","details"}
  --max-images N        photos for B1/B2 (default 8)      --labels-max N  cap for C (default 200)
  --device D            device for --checkpoint (auto, cpu, cuda)
  --gpu                 run on the GPU: the CUDA converter image (` + defaultConvertGPUImage + `)
                        with the host's GPUs; implies --device cuda unless --device is given
  --image IMAGE         converter image (default $VISIONSERVE_CONVERT_IMAGE or ` + defaultConvertImage + `)

Runs in the converter image (Docker), like convert. Exit status: 0 PASS or WARN, 1 FAIL, 2 usage or
setup error (no server running, model not installed, missing file).

Examples:
  visionserve check rf-detr --images ./photos
  visionserve check my-detector --images ./val/images --labels ./val/_annotations.coco.json \
      --checkpoint ./checkpoint_best_total.pth --report check.html
`

// hasFlag reports whether args already carries flag (as "--flag" or "--flag=value").
func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag || strings.HasPrefix(a, flag+"=") {
			return true
		}
	}
	return false
}
