//go:build race

package preprocess_test

// raceEnabled: under -race the sweeps run their short sets (the race detector slows imaging
// ~25x); the full bit-equivalence sweep runs in the normal test run.
const raceEnabled = true
