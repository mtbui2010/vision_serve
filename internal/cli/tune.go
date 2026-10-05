package cli

// `visionserve sensitivity` and `visionserve optimize`: thin host-side wrappers, like `convert`.
// The work (ONNX graph surgery, INT8 calibration, per-layer sensitivity) is Python in the
// converter package (clients/python/visionserve/convert/edge.py); the server binary stays
// Python-free (CLAUDE.md rule 3). Two ways to run it:
//
//   - Docker (default): the converter image, with the registry and the input paths mounted.
//   - --python PY (or $VISIONSERVE_CONVERT_PYTHON): a local Python that has the converter
//     installed (`pip install visionserve[convert]`), e.g. on a machine without Docker.
//
// Both print the same verdict / table / JSON / HTML report, produced on the Python side.

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"visionserve/internal/cli/clireport"
)

const sensitivityUsage = `visionserve sensitivity <model> --images DIR [flags]

Which layers of an installed model lose accuracy when reduced to INT8 (or FP16, INT4)? Each
MatMul / Gemm / Conv is reduced ALONE and the model's outputs are compared with FP32 on your
photos. The summary names the sensitive layers: those are the ones to keep in FP32 (optimize
does it for you). Single-session image models (RF-DETR, classification, depth, ...).

  --images DIR        photos like the ones the model will see (8-32; used for ranges and the comparison)
  --formats LIST      formats to measure per layer: int8 (default), fp16, int4, fp8, fp4 (fp8/fp4 simulated)
  --sens-images N     images per layer (default 8; cost ~ layers x N model runs)
  --threshold E       a layer is "sensitive" when its error alone exceeds E (default 0.05)
  --top N             rows in the table (default 15)
  --save FILE         write the per-layer scores (sensitivity.json); optimize --sensitivity FILE reuses them
  --gpu               measure with ONNX Runtime's CUDA EP when the converter supports it (CPU otherwise)
  --json              one JSON object {"verdict","reason","summary","details"} only
  --report FILE.html  a self-contained HTML report with a bar chart
  --models DIR        registry (default as for convert)
  --python PY         run the converter with this local Python instead of Docker
                      (also $VISIONSERVE_CONVERT_PYTHON)
  --image IMAGE       converter image (default $VISIONSERVE_CONVERT_IMAGE or ` + defaultConvertImage + `)

Exit status: 0 PASS/WARN, 1 FAIL, 2 usage or setup error.
`

const optimizeUsage = `visionserve optimize <model> --target jetson-orin|jetson-thor|cuda|cpu --images DIR [flags]

Builds the reduced-precision variants that make sense for a target (FP16, INT8, mixed INT8/FP16,
...), measures each one against FP32 (file size, output error, accuracy with --labels, latency
on THIS machine) and recommends one inside your budget: the fastest measured, FP32 included
(within 10 %: the smaller file), or the smallest when this host cannot measure the target's EP.

Accuracy is measured here. Latency is measured on THIS machine only: a Jetson's numbers must be
measured on the Jetson. Workflow: optimize on a PC -> copy the model dir to the device ->
visionserve bench <model>-<format> there.

  --target T          jetson-orin | jetson-thor | cuda | cpu (presets: see the guide "edge.md")
  --images DIR        calibration photos (8-32, like the deployment's)
  --labels FILE       COCO annotations json: measure mAP of every variant (served, via a
                      temporary server) and its drop vs FP32
  --label-images DIR  images of --labels (default: found next to the json, or --images)
  --max-labels N      cap on labelled images (default 100)
  --max-drop P        accuracy budget: mAP points a variant may lose vs FP32 (default 1.0)
  --max-output-err E  output-error budget vs FP32 on --images (default 0.05)
  --install           install the recommended variant as <model>-<format> (manifest copied,
                      same preprocessing, precision recorded in it)
  --install-format F  with --install: install this variant instead (e.g. fp16 when size matters
                      more than speed); it must be inside the budget
  --tensorrt          jetson targets: also measure the variants under the TensorRT EP (flagged:
                      it measured 6.8 mAP lower on GroundingDINO, BUGS_TO_FIX.md #3)
  --sensitivity FILE  reuse per-layer scores from visionserve sensitivity --save
  --gpu               measure latency/accuracy on this machine's GPU (CUDA EP); default CPU
  --json, --report FILE.html, --models DIR, --python PY, --image IMAGE   as for sensitivity

Exit status: 0 PASS/WARN, 1 FAIL, 2 usage or setup error.
`

// tuneInPaths are flags whose value is an input path on the host (mounted read-only into the
// container); tuneOutPaths are output files (their directory is mounted read-write).
var (
	tuneInPaths  = map[string]bool{"--images": true, "--labels": true, "--label-images": true, "--sensitivity": true, "--calib": true}
	tuneOutPaths = map[string]bool{"--report": true, "--save": true}
)

func runSensitivity(args []string) error { return runTuneTool("sensitivity", sensitivityUsage, args) }
func runOptimize(args []string) error    { return runTuneTool("optimize", optimizeUsage, args) }

// tuneArgs is the parsed wrapper-only part of a sensitivity / optimize command line.
type tuneArgs struct {
	python, image, models string
	gpu                   bool
	pass                  []string // everything for the Python side, model first
}

func parseTuneArgs(args []string) (tuneArgs, error) {
	t := tuneArgs{python: os.Getenv("VISIONSERVE_CONVERT_PYTHON"), image: os.Getenv("VISIONSERVE_CONVERT_IMAGE")}
	value := func(i *int, a string) (string, error) {
		if _, v, ok := strings.Cut(a, "="); ok {
			return v, nil
		}
		if *i+1 >= len(args) {
			return "", clireport.Usagef("%s needs a value", a)
		}
		*i++
		return args[*i], nil
	}
	for i := 0; i < len(args); i++ {
		a := args[i]
		name, _, _ := strings.Cut(a, "=")
		var err error
		switch name {
		case "--python":
			t.python, err = value(&i, a)
		case "--image":
			t.image, err = value(&i, a)
		case "--models":
			t.models, err = value(&i, a)
		default:
			if a == "--gpu" {
				t.gpu = true
			}
			t.pass = append(t.pass, a)
		}
		if err != nil {
			return t, err
		}
	}
	if len(t.pass) == 0 || strings.HasPrefix(t.pass[0], "-") {
		return t, clireport.Usagef("the model name comes first, e.g. visionserve optimize rf-detr --target jetson-orin --images ./photos")
	}
	if t.image == "" { // no --image and no $VISIONSERVE_CONVERT_IMAGE: --gpu picks the CUDA image, as convert does
		t.image = defaultConvertImage
		if t.gpu {
			t.image = defaultConvertGPUImage
		}
	}
	return t, nil
}

func runTuneTool(tool, usage string, args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		fmt.Print(usage)
		return nil
	}
	t, err := parseTuneArgs(args)
	if err != nil {
		return err
	}
	dir, err := convertModelsDir(t.models)
	if err != nil {
		return clireport.Usagef("%v", err)
	}
	var cmd *exec.Cmd
	if t.python != "" {
		cmd = exec.Command(t.python, append([]string{"-m", "visionserve.convert", tool}, append(t.pass, "--models", dir)...)...)
		cmd.Env = os.Environ()
		if os.Getenv("VISIONSERVE_BIN") == "" { // the temporary server optimize starts is this binary
			if exe, err := os.Executable(); err == nil {
				cmd.Env = append(cmd.Env, "VISIONSERVE_BIN="+exe)
			}
		}
		fmt.Fprintf(os.Stderr, "models: %s\n$ %s\n", dir, shellJoin(cmd.Args))
	} else {
		cacheDir := filepath.Join(userHome(), ".cache", "visionserve-convert")
		if err := os.MkdirAll(cacheDir, 0o755); err != nil {
			return clireport.Usagef("create cache dir: %v", err)
		}
		dargs, err := buildTuneDockerArgs(tool, t, dir, cacheDir, os.Getuid(), os.Getgid())
		if err != nil {
			return err
		}
		if _, err := exec.LookPath("docker"); err != nil {
			return clireport.Usagef("%s runs the converter: it needs Docker (image %s) or a local Python with the converter "+
				"installed (--python PY, or $VISIONSERVE_CONVERT_PYTHON).\nEquivalent Docker command:\n  docker %s",
				tool, t.image, shellJoin(dargs))
		}
		fmt.Fprintf(os.Stderr, "models: %s\n$ docker %s\n", dir, shellJoin(dargs))
		cmd = exec.Command("docker", dargs...)
	}
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			code := ee.ExitCode()
			if code == 1 || code == 2 {
				return &clireport.ExitError{Code: code, Reported: true} // the tool printed the verdict / error
			}
			return clireport.Usagef("%s exited with status %d", tool, code)
		}
		return clireport.Usagef("%s: %v", tool, err)
	}
	return nil
}

// buildTuneDockerArgs mounts the registry (read-write: --install and --save write to it), input
// paths read-only and output files' directories read-write, rewriting each path. --gpu adds
// --gpus all (withGPU, as convert does); parseTuneArgs already picked the CUDA image for it.
func buildTuneDockerArgs(tool string, t tuneArgs, modelsDir, cacheDir string, uid, gid int) ([]string, error) {
	run := []string{"run", "--rm", "--network", "host"}
	if uid >= 0 && gid >= 0 {
		run = append(run, "--user", fmt.Sprintf("%d:%d", uid, gid))
	}
	run = append(run, "-e", "HOME=/tmp", "-e", "HF_HOME=/cache/hf",
		"--mount", bindMount(modelsDir, "/root/.models", false),
		"--mount", bindMount(cacheDir, "/cache", false))
	n := 0
	mount := func(hostPath string, out bool) (string, error) {
		abs, err := filepath.Abs(hostPath)
		if err != nil {
			return "", err
		}
		root := filepath.Dir(abs)
		if !out && isDir(abs) {
			root = abs
		}
		rel, err := filepath.Rel(root, abs)
		if err != nil {
			return "", err
		}
		n++
		dst := fmt.Sprintf("/in/%d", n)
		if out {
			dst = fmt.Sprintf("/out/%d", n)
		}
		run = append(run, "--mount", bindMount(root, dst, !out))
		if rel == "." {
			return dst, nil
		}
		return dst + "/" + filepath.ToSlash(rel), nil
	}
	out := []string{tool, t.pass[0]}
	for i := 1; i < len(t.pass); i++ {
		a := t.pass[i]
		name, val, hasEq := strings.Cut(a, "=")
		in, isOut := tuneInPaths[name], tuneOutPaths[name]
		if !in && !isOut {
			out = append(out, a)
			continue
		}
		if !hasEq {
			if i+1 >= len(t.pass) {
				return nil, clireport.Usagef("%s needs a value", a)
			}
			i++
			val = t.pass[i]
		}
		if in && !exists(val) {
			return nil, clireport.Usagef("%s %s: no such file or directory", name, val)
		}
		if isOut {
			if err := os.MkdirAll(filepath.Dir(absOr(val)), 0o755); err != nil {
				return nil, clireport.Usagef("%s: %v", name, err)
			}
		}
		p, err := mount(val, isOut)
		if err != nil {
			return nil, err
		}
		out = append(out, name, p)
	}
	out = append(out, "--models", "/root/.models")
	args := append(append(run, t.image), out...)
	if t.gpu {
		args = withGPU(args)
	}
	return args, nil
}

func absOr(p string) string {
	if a, err := filepath.Abs(p); err == nil {
		return a
	}
	return p
}
