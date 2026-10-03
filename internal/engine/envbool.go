package engine

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
)

// boolEnv reads an on/off VISIONSERVE_* switch: unset or empty is false; 1/0, true/false, on/off
// and yes/no (any case) are accepted. A value that does not parse keeps the default (off) and is
// reported once per process through bad, with keptOff saying what stays off.
func boolEnv(name, keptOff string, bad *sync.Once) bool {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return false
	}
	switch strings.ToLower(v) {
	case "on", "yes":
		return true
	case "off", "no":
		return false
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		bad.Do(func() {
			fmt.Fprintf(os.Stderr, "engine: ignoring %s=%q (want 1/0, true/false, on/off); %s\n", name, v, keptOff)
		})
		return false
	}
	return b
}
