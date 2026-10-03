package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Host paths must be mounted read-only and rewritten to their in-container path; everything else
// (including a hub id that is not a local path) passes through untouched.
func TestBuildConvertDockerArgsMountsHostPaths(t *testing.T) {
	dir := t.TempDir()
	ckpt := filepath.Join(dir, "ckpt.pth")
	labels := filepath.Join(dir, "sub", "classes.txt")
	for _, p := range []string{ckpt, labels} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	args, err := buildConvertDockerArgs(
		[]string{"rfdetr", ckpt, "--name", "det", "--labels=" + labels, "--force"},
		"img:tag", "/models", "/cache", 1000, 1001)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(args, " ")
	for _, want := range []string{
		"run --rm --network host --user 1000:1001",
		"--mount type=bind,src=/models,dst=/root/.models",
		"--mount type=bind,src=" + dir + ",dst=/in/1,readonly",
		"--mount type=bind,src=" + filepath.Join(dir, "sub") + ",dst=/in/2,readonly",
		"img:tag rfdetr /in/1/ckpt.pth --name det --labels /in/2/classes.txt --force --models /root/.models",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("docker args lack %q:\n%s", want, got)
		}
	}
}

func TestBuildConvertDockerArgsPassesHubIDs(t *testing.T) {
	args, err := buildConvertDockerArgs([]string{"hf", "google/vit-base-patch16-224", "--name", "vit"},
		"img", "/m", "/c", 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(args, " "); !strings.Contains(got, "img hf google/vit-base-patch16-224 --name vit") ||
		strings.Contains(got, "/in/") {
		t.Fatalf("hub id should pass through unmounted:\n%s", got)
	}
}

func TestBuildConvertDockerArgsRejectsMissingPathFlag(t *testing.T) {
	if _, err := buildConvertDockerArgs([]string{"pytorch", "x", "--script", "/nope/build.py"},
		"img", "/m", "/c", 1, 1); err == nil {
		t.Fatal("a --script that does not exist must be refused before docker runs")
	}
	if _, err := buildConvertDockerArgs([]string{"rfdetr"}, "img", "/m", "/c", 1, 1); err == nil {
		t.Fatal("missing <source> must be a usage error")
	}
}

// Review #9: a COCO --eval json keeps its images in a SIBLING of its own directory
// (annotations/instances_val2017.json -> ../val2017/). Mounting only the json's parent hid them.
func TestBuildConvertDockerArgsEvalMountsCOCOLayout(t *testing.T) {
	root := t.TempDir()
	ann := filepath.Join(root, "annotations", "instances_val2017.json")
	img := filepath.Join(root, "val2017", "000000000139.jpg")
	for _, p := range []string{ann, img} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	args, err := buildConvertDockerArgs([]string{"hf", "org/model", "--name", "m", "--eval", ann},
		"img", "/m", "/c", 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(args, " ")
	if !strings.Contains(got, "type=bind,src="+root+",dst=/in/1,readonly") ||
		!strings.Contains(got, "--eval /in/1/annotations/instances_val2017.json") {
		t.Fatalf("--eval must mount the dataset root (the json's grandparent):\n%s", got)
	}
}

// Review #10: `-v src:dst:ro` breaks on a host path containing ':'; --mount does not.
// Windows (Getuid() == -1) must not get `--user -1:-1`.
func TestBuildConvertDockerArgsMountSyntaxAndUser(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "run:2026-10-03")
	ckpt := filepath.Join(dir, "ckpt.pth")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ckpt, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	args, err := buildConvertDockerArgs([]string{"rfdetr", ckpt, "--name", "d"}, "img", "/m:x", "/c", -1, -1)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(args, " ")
	for _, want := range []string{
		"--mount type=bind,src=" + dir + ",dst=/in/1,readonly",
		"--mount type=bind,src=/m:x,dst=/root/.models",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("args lack %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, " -v ") || strings.Contains(got, "--user") {
		t.Errorf("no -v mounts and no --user when Getuid() < 0:\n%s", got)
	}
}

func TestShellQuoteCommand(t *testing.T) {
	got := shellJoin([]string{"run", "--mount", "type=bind,src=/a b/it's,dst=/in/1", "plain/arg-1.0:x"})
	want := `run --mount 'type=bind,src=/a b/it'\''s,dst=/in/1' plain/arg-1.0:x`
	if got != want {
		t.Fatalf("shellJoin:\n got %s\nwant %s", got, want)
	}
}

func TestConvertModelsDirExpandsTilde(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	got, err := convertModelsDir("~/vs-models")
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(home, "vs-models"); got != want {
		t.Fatalf("--models ~/vs-models resolved to %s, want %s", got, want)
	}
}

func TestBindMountQuotesCSVFields(t *testing.T) {
	if got, want := bindMount("/data/a,b", "/in/1", true), `type=bind,"src=/data/a,b",dst=/in/1,readonly`; got != want {
		t.Fatalf("bindMount = %s, want %s", got, want)
	}
}
