package cli

// Host probes for `visionserve bench`: process memory, GPU memory and the kind of GPU the host
// has. Every probe is best effort: a missing tool or file gives "unknown", never an error, and
// what was used is reported next to the number.
//
//   - nvidia-smi (desktop / server NVIDIA GPUs): per-process GPU memory for the bench process
//     (in-process) or the server's PID (local server), plus the GPU's total memory used.
//   - Jetson: the GPU shares system RAM, so there is no separate "VRAM", and nvidia-smi does not
//     report memory there (Orin/JetPack 6: a stub; Thor/JetPack 7: "Not Supported"). tegrastats'
//     RAM line (whole system, CPU + GPU) and GR3D_FREQ (GPU load; JetPack 7.0 on Thor prints only
//     the clocks, so the load is then unknown) are sampled instead; /sys/.../gpu.0/load is the
//     fallback for the load when tegrastats is missing.
//   - process RSS / peak RSS from /proc/<pid>/status (Linux); on Jetson this includes most of
//     the CUDA allocations, since they live in the same physical memory.

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// gpuKind describes what the probes could find.
type gpuKind struct {
	Jetson      bool   // /etc/nv_tegra_release or a Jetson device-tree model
	Model       string // "NVIDIA Jetson AGX Orin Developer Kit", "NVIDIA RTX A6000", ...
	NvidiaSMI   bool
	Tegrastats  string // path, "" when absent
	VisibleGPUs string // CUDA_VISIBLE_DEVICES as set
}

func (g gpuKind) present() bool { return g.Jetson || g.NvidiaSMI }

func probeGPU() gpuKind {
	g := gpuKind{VisibleGPUs: os.Getenv("CUDA_VISIBLE_DEVICES")}
	if raw, err := os.ReadFile("/proc/device-tree/model"); err == nil {
		m := strings.TrimRight(string(raw), "\x00\n ")
		if strings.Contains(strings.ToLower(m), "jetson") || strings.Contains(strings.ToLower(m), "nvidia") {
			g.Jetson, g.Model = true, m
		}
	}
	if _, err := os.Stat("/etc/nv_tegra_release"); err == nil {
		g.Jetson = true
	}
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader").Output()
		if err == nil && len(strings.TrimSpace(string(out))) > 0 {
			g.NvidiaSMI = true
			if g.Model == "" {
				g.Model = gpuNames(string(out), g.VisibleGPUs)
			}
		}
	}
	for _, p := range []string{"/usr/bin/tegrastats", "/usr/sbin/tegrastats"} {
		if _, err := os.Stat(p); err == nil {
			g.Tegrastats = p
			break
		}
	}
	if g.Tegrastats == "" {
		if p, err := exec.LookPath("tegrastats"); err == nil {
			g.Tegrastats = p
		}
	}
	return g
}

// gpuNames picks the names of the visible GPUs from "index, name" lines.
func gpuNames(out, visible string) string {
	keep := visibleSet(visible)
	var names []string
	for _, ln := range strings.Split(strings.TrimSpace(out), "\n") {
		idx, name, ok := strings.Cut(ln, ",")
		if !ok {
			continue
		}
		if keep != nil && !keep[strings.TrimSpace(idx)] {
			continue
		}
		names = append(names, strings.TrimSpace(name))
	}
	return strings.Join(names, ", ")
}

// visibleSet is the set of physical GPU indices in CUDA_VISIBLE_DEVICES, or nil (all) when it is
// unset or uses UUIDs.
func visibleSet(visible string) map[string]bool {
	if strings.TrimSpace(visible) == "" {
		return nil
	}
	keep := map[string]bool{}
	for _, v := range strings.Split(visible, ",") {
		v = strings.TrimSpace(v)
		if _, err := strconv.Atoi(v); err != nil {
			return nil
		}
		keep[v] = true
	}
	return keep
}

// gpuMem is one GPU-memory reading.
type gpuMem struct {
	ProcessMB float64 // memory of the measured PID (nvidia-smi compute apps); <0 unknown
	UsedMB    float64 // memory used on the visible GPU(s) / system RAM on Jetson; <0 unknown
	TotalMB   float64
	LoadPct   float64 // GPU load (Jetson GR3D_FREQ / sysfs); <0 unknown
	Source    string
}

func unknownGPUMem() gpuMem { return gpuMem{ProcessMB: -1, UsedMB: -1, TotalMB: -1, LoadPct: -1} }

// parseComputeApps reads `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`
// and returns the MiB used by pid (summed over GPUs), or -1 when pid is not listed or the
// driver reports [N/A] (iGPUs).
func parseComputeApps(out string, pid int) float64 {
	total, found := 0.0, false
	for _, ln := range strings.Split(out, "\n") {
		p, mem, ok := strings.Cut(ln, ",")
		if !ok {
			continue
		}
		if strings.TrimSpace(p) != strconv.Itoa(pid) {
			continue
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(mem), 64)
		if err != nil {
			continue
		}
		total += v
		found = true
	}
	if !found {
		return -1
	}
	return total
}

// parseGPUMemory reads `--query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`
// and sums the visible GPUs.
func parseGPUMemory(out, visible string) (used, total float64) {
	keep := visibleSet(visible)
	used, total = -1, -1
	for _, ln := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Split(ln, ",")
		if len(f) != 3 {
			continue
		}
		if keep != nil && !keep[strings.TrimSpace(f[0])] {
			continue
		}
		u, err1 := strconv.ParseFloat(strings.TrimSpace(f[1]), 64)
		t, err2 := strconv.ParseFloat(strings.TrimSpace(f[2]), 64)
		if err1 != nil || err2 != nil {
			continue
		}
		if used < 0 {
			used, total = 0, 0
		}
		used += u
		total += t
	}
	return used, total
}

func smiGPUMem(pid int, visible string) gpuMem {
	m := unknownGPUMem()
	m.Source = "nvidia-smi"
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if pid > 0 {
		if out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-compute-apps=pid,used_memory",
			"--format=csv,noheader,nounits").Output(); err == nil {
			m.ProcessMB = parseComputeApps(string(out), pid)
		}
	}
	if out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=index,memory.used,memory.total",
		"--format=csv,noheader,nounits").Output(); err == nil {
		m.UsedMB, m.TotalMB = parseGPUMemory(string(out), visible)
	}
	return m
}

// tegrastats line, e.g.
// "RAM 7712/30536MB (lfb 4x4MB) SWAP 0/15268MB ... GR3D_FREQ 45% ..." (Orin, JetPack 6) or
// "... GR3D_FREQ 45%@[1300,1300] ..." (some releases print the GPC clocks).
var (
	tegraRAM  = regexp.MustCompile(`RAM (\d+)/(\d+)MB`)
	tegraGR3D = regexp.MustCompile(`GR3D_FREQ (\d+)%`)
)

func parseTegrastats(line string) (usedMB, totalMB, loadPct float64) {
	usedMB, totalMB, loadPct = -1, -1, -1
	if m := tegraRAM.FindStringSubmatch(line); m != nil {
		usedMB, _ = strconv.ParseFloat(m[1], 64)
		totalMB, _ = strconv.ParseFloat(m[2], 64)
	}
	if m := tegraGR3D.FindStringSubmatch(line); m != nil {
		loadPct, _ = strconv.ParseFloat(m[1], 64)
	}
	return
}

// sysfsGPULoad reads the Jetson GPU load (0-1000) from sysfs, or -1.
func sysfsGPULoad() float64 {
	for _, p := range []string{"/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load",
		"/sys/devices/platform/bus@0/17000000.gpu/load", "/sys/devices/17000000.ga10b/load"} {
		raw, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(string(raw)), 64)
		if err == nil {
			return v / 10
		}
	}
	return -1
}

// memSampler samples process RSS and GPU memory while the timed phase runs, keeping the peaks.
type memSampler struct {
	pid  int
	gpu  gpuKind
	stop chan struct{}
	done chan struct{}

	mu      sync.Mutex
	rssMB   float64 // last
	rssPeak float64
	gpuLast gpuMem
	gpuPeak gpuMem
	tegra   *exec.Cmd
}

func startMemSampler(pid int, gpu gpuKind, every time.Duration) *memSampler {
	s := &memSampler{pid: pid, gpu: gpu, stop: make(chan struct{}), done: make(chan struct{}),
		rssMB: -1, rssPeak: -1, gpuLast: unknownGPUMem(), gpuPeak: unknownGPUMem()}
	if gpu.Jetson && gpu.Tegrastats != "" {
		s.startTegrastats(every)
	}
	go s.loop(every)
	return s
}

// startTegrastats runs `tegrastats --interval N` (a child this process owns and stops) and
// keeps the peak of its readings.
func (s *memSampler) startTegrastats(every time.Duration) {
	cmd := exec.Command(s.gpu.Tegrastats, "--interval", strconv.Itoa(int(every.Milliseconds())))
	out, err := cmd.StdoutPipe()
	if err != nil {
		return
	}
	if err := cmd.Start(); err != nil {
		return
	}
	s.tegra = cmd
	go func() {
		sc := bufio.NewScanner(out)
		for sc.Scan() {
			u, t, l := parseTegrastats(sc.Text())
			s.mu.Lock()
			s.gpuLast = gpuMem{ProcessMB: -1, UsedMB: u, TotalMB: t, LoadPct: l, Source: "tegrastats (system RAM, shared by CPU and GPU)"}
			s.gpuPeak = peakGPU(s.gpuPeak, s.gpuLast)
			s.mu.Unlock()
		}
	}()
}

func peakGPU(a, b gpuMem) gpuMem {
	out := b
	out.ProcessMB = maxF(a.ProcessMB, b.ProcessMB)
	out.UsedMB = maxF(a.UsedMB, b.UsedMB)
	out.LoadPct = maxF(a.LoadPct, b.LoadPct)
	if b.TotalMB < 0 {
		out.TotalMB = a.TotalMB
	}
	if b.Source == "" {
		out.Source = a.Source
	}
	return out
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func (s *memSampler) sample() {
	rss, _ := processRSS(s.pid)
	var g gpuMem
	haveGPU := false
	switch {
	case s.gpu.Jetson && s.tegra != nil:
		// tegrastats feeds the readings (startTegrastats)
	case s.gpu.Jetson:
		g = unknownGPUMem()
		g.LoadPct, g.Source, haveGPU = sysfsGPULoad(), "sysfs gpu load", true
	case s.gpu.NvidiaSMI:
		g, haveGPU = smiGPUMem(s.pid, s.gpu.VisibleGPUs), true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if rss >= 0 {
		s.rssMB = rss
		s.rssPeak = maxF(s.rssPeak, rss)
	}
	if haveGPU {
		s.gpuLast = g
		s.gpuPeak = peakGPU(s.gpuPeak, g)
	}
}

func (s *memSampler) loop(every time.Duration) {
	defer close(s.done)
	s.sample()
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-s.stop:
			s.sample()
			return
		case <-t.C:
			s.sample()
		}
	}
}

// finish stops sampling and returns (last RSS, peak RSS, peak GPU reading).
func (s *memSampler) finish() (float64, float64, gpuMem) {
	close(s.stop)
	<-s.done
	if s.tegra != nil && s.tegra.Process != nil {
		_ = s.tegra.Process.Kill() // our own child
		_ = s.tegra.Wait()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.rssMB, s.rssPeak, s.gpuPeak
}

// fmtMB renders MiB as "812 MB" / "1.2 GB", or "n/a".
func fmtMB(mb float64) string {
	switch {
	case mb < 0:
		return "n/a"
	case mb >= 1024:
		return fmt.Sprintf("%.2f GB", mb/1024)
	default:
		return fmt.Sprintf("%.0f MB", mb)
	}
}
