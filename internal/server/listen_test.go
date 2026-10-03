package server

import "testing"

// The loopback note names the port actually used, not the default one.
func TestListenScope(t *testing.T) {
	for addr, want := range map[string]string{
		"127.0.0.1:11435": " (this machine only; --addr :11435 accepts other hosts)",
		"127.0.0.1:11651": " (this machine only; --addr :11651 accepts other hosts)",
		"localhost:8080":  " (this machine only; --addr :8080 accepts other hosts)",
		"[::1]:9000":      " (this machine only; --addr :9000 accepts other hosts)",
		":11651":          "",
		"0.0.0.0:11435":   "",
		"192.168.1.5:80":  "",
		"not-an-addr":     "",
	} {
		if got := listenScope(addr); got != want {
			t.Errorf("listenScope(%q) = %q, want %q", addr, got, want)
		}
	}
}
