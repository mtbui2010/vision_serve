//go:build !linux

package cli

// processRSS is not implemented off Linux (no /proc): memory is reported as unknown.
func processRSS(int) (rss, peak float64) { return -1, -1 }

// pidListeningOn is not implemented off Linux.
func pidListeningOn(int) int { return 0 }
