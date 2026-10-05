package cli

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// defaultConvertImage is the converter image. It is SEPARATE from the server image on purpose:
// converting needs Python + PyTorch/TensorFlow (several GB), and the server must stay a lean,
// Python-free Go binary (CLAUDE.md rule 3). The converter runs once and exits.
const defaultConvertImage = "mtbui2010/visionserve-convert:latest"

// defaultConvertGPUImage is the CUDA variant (onnxruntime-gpu + PyTorch CUDA). `convert --gpu` selects it
// and runs it with `--gpus all`; sensitivity / mixed-precision runs are far faster there.
const defaultConvertGPUImage = "mtbui2010/visionserve-convert:latest-gpu"

// convertPathFlags are the converter flags whose value is a HOST path that must be mounted into
// the container. The positional <source> is handled separately.
var convertPathFlags = map[string]bool{
	"--labels": true, "--script": true, "--weights": true,
	"--images": true, "--eval": true, "--eval-images": true, "--reference-script": true,
	"--calib": true,
}

// runConvert: visionserve convert <format> <source> [flags...]
//
// A thin host-side wrapper around `docker run <converter image>`: it mounts the checkpoint (and
// any --labels/--script/--weights/--eval path) read-only, mounts the model registry read-write, runs the
// conversion as the calling user so the installed files are not root-owned, and passes every
// other flag through. The converter itself exports, checks parity and the I/O contract, and
// installs through `visionserve pull <folder>` — the same validation as a hand-made folder.
func runConvert(args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		fmt.Print(convertUsage)
		return nil
	}
	image := os.Getenv("VISIONSERVE_CONVERT_IMAGE")
	var models string
	var pass []string
	gpu := false
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--gpu":
			gpu = true
		case a == "--image" && i+1 < len(args):
			image = args[i+1]
			i++
		case strings.HasPrefix(a, "--image="):
			image = strings.TrimPrefix(a, "--image=")
		case a == "--models" && i+1 < len(args):
			models = args[i+1]
			i++
		case strings.HasPrefix(a, "--models="):
			models = strings.TrimPrefix(a, "--models=")
		default:
			pass = append(pass, a)
		}
	}
	if image == "" {
		image = defaultConvertImage
		if gpu {
			image = defaultConvertGPUImage
		}
	}
	dir, err := convertModelsDir(models)
	if err != nil {
		return err
	}
	cacheDir := filepath.Join(userHome(), ".cache", "visionserve-convert")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return fmt.Errorf("convert: create cache dir: %w", err)
	}
	dockerArgs, err := buildConvertDockerArgs(pass, image, dir, cacheDir, os.Getuid(), os.Getgid())
	if err != nil {
		return err
	}
	if gpu {
		dockerArgs = withGPU(dockerArgs)
	}
	if _, err := exec.LookPath("docker"); err != nil {
		return fmt.Errorf("convert needs Docker on this machine (the converter is the %s image).\n"+
			"Equivalent command to run where Docker is available:\n  docker %s", image, shellJoin(dockerArgs))
	}
	fmt.Fprintf(os.Stderr, "models: %s\n$ docker %s\n", dir, shellJoin(dockerArgs))
	cmd := exec.Command("docker", dockerArgs...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("convert failed: %w", err)
	}
	return nil
}

// withGPU gives the container the host's NVIDIA GPUs: `--gpus all` right after `run`.
func withGPU(dockerArgs []string) []string {
	if len(dockerArgs) == 0 || dockerArgs[0] != "run" {
		return dockerArgs
	}
	out := append([]string{"run", "--gpus", "all"}, dockerArgs[1:]...)
	return out
}

// convertModelsDir picks the registry to install into. Without --models / $VISIONSERVE_MODELS it
// prefers the Docker deployment's ~/.visionserve_models (deploy/README.md) when that exists and
// the native default does not, because a converted model is useless in a registry the running
// container never sees. The chosen directory is always printed.
func convertModelsDir(flagVal string) (string, error) {
	// `--models=~/x` reaches us unexpanded (the shell only expands a leading ~ of a word).
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
		return "", err
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return "", fmt.Errorf("convert: create models dir %s: %w", abs, err)
	}
	return abs, nil
}

// buildConvertDockerArgs turns `<format> <source> [flags]` into `docker run` arguments. Host paths
// (the source and path-valued flags) are mounted read-only under /in/<n>/ and rewritten; a source
// that is not an existing path (e.g. a HuggingFace hub id) is passed through unchanged. uid/gid < 0
// (Windows) omit --user.
func buildConvertDockerArgs(pass []string, image, modelsDir, cacheDir string, uid, gid int) ([]string, error) {
	if len(pass) < 2 || strings.HasPrefix(pass[0], "-") || strings.HasPrefix(pass[1], "-") {
		return nil, fmt.Errorf("usage: visionserve convert <format> <source> --name NAME [flags]\n%s", convertUsage)
	}
	// --network host: the verification tiers call the user's running server (default
	// http://localhost:11435), which from inside a bridged container would not be "localhost".
	// --mount rather than -v: -v splits on ':', so a host path containing one (a timestamped run
	// directory, a Windows drive letter) would be mangled.
	run := []string{"run", "--rm", "--network", "host"}
	if uid >= 0 && gid >= 0 { // os.Getuid() is -1 on Windows, where Docker Desktop maps ownership itself
		run = append(run, "--user", fmt.Sprintf("%d:%d", uid, gid))
	}
	run = append(run, "-e", "HOME=/tmp", "-e", "HF_HOME=/cache/hf",
		"--mount", bindMount(modelsDir, "/root/.models", false),
		"--mount", bindMount(cacheDir, "/cache", false))
	mounts := 0
	// mount exposes hostPath read-only and returns its in-container path. up is how many
	// directory levels above the path to mount: 1 = its parent (the default), 2 = its
	// grandparent, for inputs that reference siblings of their own directory.
	mount := func(hostPath string, up int) (string, error) {
		abs, err := filepath.Abs(hostPath)
		if err != nil {
			return "", err
		}
		root := abs
		for i := 0; i < up; i++ {
			root = filepath.Dir(root)
		}
		rel, err := filepath.Rel(root, abs)
		if err != nil {
			return "", err
		}
		mounts++
		dst := fmt.Sprintf("/in/%d", mounts)
		run = append(run, "--mount", bindMount(root, dst, true))
		return dst + "/" + filepath.ToSlash(rel), nil
	}

	out := []string{pass[0]}
	if exists(pass[1]) {
		p, err := mount(pass[1], 1)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	} else {
		out = append(out, pass[1])
	}
	for i := 2; i < len(pass); i++ {
		a := pass[i]
		name, val, hasEq := strings.Cut(a, "=")
		if !convertPathFlags[name] {
			out = append(out, a)
			continue
		}
		if !hasEq {
			if i+1 >= len(pass) {
				return nil, fmt.Errorf("%s needs a value", a)
			}
			i++
			val = pass[i]
		}
		if !exists(val) {
			return nil, fmt.Errorf("%s %s: no such file", name, val)
		}
		up := 1
		if name == "--eval" && !isDir(val) {
			// A COCO json finds its images in a sibling of its own directory
			// (annotations/instances_val2017.json -> ../val2017/, ../images/): mount the dataset root.
			up = 2
		}
		p, err := mount(val, up)
		if err != nil {
			return nil, err
		}
		out = append(out, name, p)
	}
	out = append(out, "--models", "/root/.models")
	return append(append(run, image), out...), nil
}

// bindMount renders one `docker run --mount` value. The value is CSV: a field holding a comma or
// a double quote is quoted, with inner quotes doubled.
func bindMount(src, dst string, readonly bool) string {
	field := func(s string) string {
		if strings.ContainsAny(s, ",\"") {
			return `"` + strings.ReplaceAll(s, `"`, `""`) + `"`
		}
		return s
	}
	v := "type=bind," + field("src="+src) + "," + field("dst="+dst)
	if readonly {
		v += ",readonly"
	}
	return v
}

// shellJoin renders args as a POSIX shell command line that can be pasted back verbatim.
func shellJoin(args []string) string {
	out := make([]string, len(args))
	for i, a := range args {
		out[i] = shellQuote(a)
	}
	return strings.Join(out, " ")
}

func shellQuote(s string) string {
	if s == "" {
		return "''"
	}
	safe := true
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("_@%+=:,./-", r)) {
			safe = false
			break
		}
	}
	if safe {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// expandTilde expands a leading "~" or "~/" to the user's home directory.
func expandTilde(p string) string {
	if p == "~" {
		return userHome()
	}
	if strings.HasPrefix(p, "~/") || strings.HasPrefix(p, "~"+string(filepath.Separator)) {
		return filepath.Join(userHome(), p[2:])
	}
	return p
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }
func isDir(p string) bool  { st, err := os.Stat(p); return err == nil && st.IsDir() }

func userHome() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	return "/root"
}

const convertUsage = `visionserve convert <format> <source> --name NAME [flags]

Converts a checkpoint to ONNX + manifest.yaml and installs it into the model registry, using the
converter image (Docker required; the server image stays Python-free). A running server picks the
new model up without a restart.

Formats:
  rfdetr       RF-DETR training checkpoint (.pth) — variant, resolution and class names are read from it
  hf           HuggingFace model dir or hub id (grounding-dino, siglip, clip, rt_detr, image
               classification, depth estimation)
  torchscript  TorchScript .pt                      (needs --task --input)
  pytorch      state_dict + --script build.py       (needs --task --input --weights)
  tensorflow   SavedModel directory                 (needs --task --input)
  keras        .keras / .h5                         (needs --task --input)
  tflite       float .tflite                        (needs --task --input)

Common flags:
  --name NAME        registry name (required)
  --license ID       license of the ORIGINAL model: Apache-2.0 | MIT | BSD-3-Clause | BSD-2-Clause.
                     AGPL (Ultralytics YOLO, FastSAM, YOLO-World) is always refused.
  --labels FILE      class names, one per line, in output order
  --task T           classification | detection | depth | embed   (formats without an architecture)
  --input WxH        input resolution, e.g. 224x224
  --mean / --std     normalisation after /255 (default ImageNet)
  --force            replace an installed model of the same name
  --dry-run          convert and check only; print the manifest
  --models DIR       registry (default $VISIONSERVE_MODELS, else ~/.visionserve/models, or
                     ~/.visionserve_models when only that exists — the Docker layout)
  --image IMAGE      converter image (default $VISIONSERVE_CONVERT_IMAGE or ` + defaultConvertImage + `)
  --gpu              run the CUDA converter image (` + defaultConvertGPUImage + `) with --gpus all:
                     the precision step (sensitivity, mixed precision) and the verification run on the GPU

Reduced precision (opt-in; see the "Reduced precision" guide): --precision fp16|int8|int4|mixed,
--sensitivity, --formats, --max-output-err, --calib DIR, --ep auto|cpu|cuda.

Every conversion is checked before install: the ONNX output must match the original framework's
(parity), and its tensor shapes must match what the VisionServe architecture decodes.

Examples:
  visionserve convert rfdetr ./checkpoint_best_total.pth --name my-detector
  visionserve convert hf google/vit-base-patch16-224 --name vit
  visionserve convert keras ./model.keras --name my-cls --task classification --input 224x224 \
      --license Apache-2.0 --labels ./classes.txt
`
