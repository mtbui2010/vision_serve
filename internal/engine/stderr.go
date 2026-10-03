package engine

import "regexp"

// Why session creation captures stderr at all
//
// ONNX Runtime reports two things only on stderr, through its default logger:
//   - the red errors it prints while a GPU EP fails over to the next one (normal fallback noise
//     we hide when a later EP succeeds, and reprint when every EP fails), and
//   - the evidence that it silently DROPPED the EP we registered and runs on CPU (epWasDropped) —
//     the only way to keep `device` honest.
//
// The preferred fix would be to configure ORT's logging instead: a logging callback, or at least
// a per-session severity. The binding (yalue/onnxruntime_go v1.13.0) exposes neither: its C
// wrapper calls CreateEnv(ORT_LOGGING_LEVEL_ERROR, ...) with no custom logger, and SessionOptions
// has no log-severity setter. So the capture swaps fd 2 for the duration of the session creation
// (captureStderr, linux only), with these rules:
//   - only lines in ORT's log format are kept; every other line is passed straight through to the
//     real stderr as it arrives — other goroutines' logs are not swallowed or held for minutes;
//   - fd 2 is restored by a defer, so a panic cannot leave the process writing into a dead pipe;
//   - creations are still serialized (stderrMu): fd 2 is process-wide, and ORT's lines carry no
//     session id, so two overlapping captures could not tell whose EP a "fell back to CPU" line
//     is about — and would mark the wrong session as CPU;
//   - the CPU EP is never captured (it cannot be dropped and has nothing to fall back to), so
//     CPU-only hosts load models fully in parallel.
//
// A multi-line ORT message's continuation lines do not carry the prefix and are passed through:
// the capture errs on the side of showing too much, never on hiding someone else's output.

// ortLogLine matches a line of ORT's default log sink: an optional timestamp, then
// "[<severity>:onnxruntime:<logger id>, <location>] <message>".
var ortLogLine = regexp.MustCompile(`^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\S* )?\[[VIWEF]:onnxruntime:`)

// isORTLogLine reports whether line was written by ORT's logger (see ortLogLine).
func isORTLogLine(line string) bool { return ortLogLine.MatchString(line) }
