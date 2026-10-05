//go:build linux

package cli

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// processRSS returns the resident set size and its peak (VmRSS, VmHWM) of pid in MiB, from
// /proc/<pid>/status. (-1, -1) when it cannot be read (another user's process, no procfs).
func processRSS(pid int) (rss, peak float64) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return -1, -1
	}
	defer f.Close()
	rss, peak = -1, -1
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		fields := strings.Fields(v)
		if len(fields) == 0 {
			continue
		}
		kb, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			continue
		}
		switch k {
		case "VmRSS":
			rss = kb / 1024
		case "VmHWM":
			peak = kb / 1024
		}
	}
	return rss, peak
}

// pidListeningOn finds the process listening on a local TCP port, by matching the socket inode
// of /proc/net/tcp{,6} against /proc/<pid>/fd. It only sees processes this user may inspect;
// 0 when not found (a server in a container, another user's server).
func pidListeningOn(port int) int {
	inodes := map[string]bool{}
	for _, f := range []string{"/proc/net/tcp", "/proc/net/tcp6"} {
		raw, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, ino := range listenInodes(string(raw), port) {
			inodes[ino] = true
		}
	}
	if len(inodes) == 0 {
		return 0
	}
	procs, err := os.ReadDir("/proc")
	if err != nil {
		return 0
	}
	for _, p := range procs {
		pid, err := strconv.Atoi(p.Name())
		if err != nil {
			continue
		}
		fds, err := os.ReadDir(filepath.Join("/proc", p.Name(), "fd"))
		if err != nil {
			continue
		}
		for _, fd := range fds {
			link, err := os.Readlink(filepath.Join("/proc", p.Name(), "fd", fd.Name()))
			if err != nil || !strings.HasPrefix(link, "socket:[") {
				continue
			}
			if inodes[strings.TrimSuffix(strings.TrimPrefix(link, "socket:["), "]")] {
				return pid
			}
		}
	}
	return 0
}

// listenInodes returns the inodes of sockets in LISTEN state (st 0A) on port in a
// /proc/net/tcp table.
func listenInodes(table string, port int) []string {
	want := fmt.Sprintf(":%04X", port)
	var out []string
	for i, ln := range strings.Split(table, "\n") {
		if i == 0 {
			continue // header
		}
		f := strings.Fields(ln)
		if len(f) < 10 {
			continue
		}
		if strings.HasSuffix(f[1], want) && f[3] == "0A" {
			out = append(out, f[9])
		}
	}
	return out
}
