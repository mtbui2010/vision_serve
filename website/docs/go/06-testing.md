# 6. Testing

!!! abstract "What you'll learn"
    - How `go test` finds and runs tests, and the flags you will use most.
    - Table-driven tests and `t.Run` subtests, the standard Go test style.
    - Helpers that keep tests clean: `t.Helper`, `t.TempDir`, `t.Cleanup`, `t.Setenv`, `t.Skip`.
    - Oracle tests, golden files in `testdata/`, fakes, benchmarks and the race detector,
      each shown on a real test in this repository.

CLAUDE.md asks for **tests for the pre/postprocess of every model**, "the most error-prone
part". Go makes that cheap: the test tool is built in, and no framework is needed.

## The basics

A test lives in a file ending in `_test.go`, next to the code it tests, in the same
package. A test is a function `TestXxx(t *testing.T)`:

=== "Go"

    ```go title="internal/vision/nms/nms_test.go (lines 13-22)"
    func TestSuppressesOverlapSameClass(t *testing.T) {
    	dets := []api.Detection{
    		{BBox: [4]float64{0, 0, 100, 100}, Class: "cat", Conf: 0.9},
    		{BBox: [4]float64{10, 10, 100, 100}, Class: "cat", Conf: 0.8}, // high IoU -> suppressed
    	}
    	keep := Detections(dets, Options{IoU: 0.5})
    	if len(keep) != 1 || keep[0].Conf != 0.9 {
    		t.Fatalf("want 1 box kept with conf 0.9, got %+v", keep)
    	}
    }
    ```

=== "pytest"

    ```python
    def test_suppresses_overlap_same_class():
        dets = [Detection(bbox=(0, 0, 100, 100), cls="cat", conf=0.9),
                Detection(bbox=(10, 10, 100, 100), cls="cat", conf=0.8)]
        keep = nms.detections(dets, Options(iou=0.5))
        assert len(keep) == 1 and keep[0].conf == 0.9, keep
    ```

[nms_test.go#L13-L22 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/nms/nms_test.go#L13-L22)

There is no `assert`. You compare with ordinary `if` statements and report with:

- `t.Errorf(...)`: record a failure and **continue** (to see several failures at once);
- `t.Fatalf(...)`: record a failure and **stop this test** (when continuing makes no sense).

Because the test is in package `nms`, it can call unexported functions too. When you want
to test only the public surface, name the package `nms_test` instead; the
`internal/vision/preprocess` tests do this (`package preprocess_test` in
[sync_test.go#L1](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/sync_test.go#L1)).

| Command | What it does |
|---|---|
| `go test ./...` | every test in the module |
| `go test ./internal/vision/nms/` | one package |
| `go test ./internal/vision/nms/ -run Containment -v` | tests whose name matches the regexp, with output |
| `go test ./... -count=1` | ignore the cache (Go caches passing results and prints `(cached)`) |
| `go test -cover ./internal/vision/nms/` | statement coverage (`coverage: 94.5% of statements`) |
| `go test -race ./internal/lifecycle/` | with the data-race detector (chapter 5) |

## Table-driven tests and subtests

The idiomatic Go test lists its cases in a slice of anonymous structs and loops over them.
This is `pytest.mark.parametrize` without the decorator:

```go title="internal/server/handlers_test.go (lines 28-51, trimmed)"
func TestStatusOf(t *testing.T) {
	maxBytes := &http.MaxBytesError{Limit: 10}
	cases := []struct {
		name string
		err  error
		want int
	}{
		{"not found (wrapped by the runtime)", fmt.Errorf("lifecycle: model %q: %w", "x", lifecycle.ErrModelNotFound), 404},
		{"invalid request (wrapped)", fmt.Errorf("lifecycle: template: %w", lifecycle.ErrInvalidRequest), 400},
		{"overloaded (wrapped)", fmt.Errorf("lifecycle: queue full: %w", lifecycle.ErrOverloaded), 503},
		// ...
		{"client gone", errClientGone, 499},
		{"anything else", errors.New("onnx: run failed"), 500},
	}
	for _, c := range cases {
		if got := statusOf(c.err); got != c.want {
			t.Errorf("%s: status %d, want %d", c.name, got, c.want)
		}
	}
```

[handlers_test.go#L28-L51 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/server/handlers_test.go#L28-L51)

Adding a case is one line. With `t.Run(name, func(t *testing.T) {...})` each case becomes a
named **subtest** that reports separately and can be run alone:

```go title="internal/models/groundingdino/postprocess_test.go (lines 46-68)"
func TestPhraseSpansEdgeCases(t *testing.T) {
	tok := testTokenizer(t)
	for _, tc := range []struct {
		name, prompt string
		want         []string
	}{
		{"no trailing period", "cup", []string{"cup"}},
		{"double period", "cup.. remote.", []string{"cup", "remote"}},
		{"single class", "water bottle.", []string{"water bottle"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spans := phraseSpans(tok.Encode(tc.prompt).InputIDs, tok, nil)
			if len(spans) != len(tc.want) {
				t.Fatalf("got %d spans, want %d: %v", len(spans), len(tc.want), spans)
			}
			for i, w := range tc.want {
				if spans[i].text != w {
					t.Errorf("span %d = %q, want %q", i, spans[i].text, w)
				}
			}
		})
	}
}
```

[postprocess_test.go#L46-L68 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/groundingdino/postprocess_test.go#L46-L68)

```console
$ go test ./internal/models/groundingdino -run 'TestPhraseSpansEdgeCases/double_period' -v
=== RUN   TestPhraseSpansEdgeCases/double_period
--- PASS: TestPhraseSpansEdgeCases (0.03s)
    --- PASS: TestPhraseSpansEdgeCases/double_period (0.00s)
```

Spaces in a subtest name become `_` on the command line.

## Test helpers

The `testing.T` value has a few methods that remove most boilerplate:

| Method | Use | Example in the repo |
|---|---|---|
| `t.Helper()` | marks a helper so failures point at the *caller's* line | `testTokenizer`, `writeTestModel` |
| `t.Skip(...)` | skip when a resource is missing, instead of failing | weights or `vocab.txt` not downloaded |
| `t.TempDir()` | a fresh directory, deleted after the test | fake model registries |
| `t.Cleanup(f)` | run `f` after the test (like a fixture's teardown) | close a `Manager` |
| `t.Setenv(k, v)` | set an env variable for this test only | `VISIONSERVE_POOL_THREADS` |

A helper that loads a tokenizer, and skips if the file is absent:

```go title="internal/models/groundingdino/postprocess_test.go (lines 17-24)"
func testTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := LoadTokenizer(filepath.Join("..", "..", "..", "models", "grounding-dino", "vocab.txt"))
	if err != nil {
		t.Skipf("vocab.txt unavailable: %v", err)
	}
	return tok
}
```

[postprocess_test.go#L17-L24 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/groundingdino/postprocess_test.go#L17-L24)

`go test` runs with the package directory as the working directory, which is why the path
climbs `../../..` to the repository root.

The lifecycle tests build a whole model registry on disk in a temporary directory, then
create a real `Manager` whose ONNX sessions are fakes:

```go title="internal/lifecycle/fixtures_test.go (lines 175-183)"
// newFakeManager is a real Manager over reg whose sessions are fakeEngines.
func newFakeManager(t *testing.T, reg *registry.Registry) (*Manager, *fakeOpener) {
	t.Helper()
	op := &fakeOpener{}
	m := NewManager(reg)
	m.openRunnable = op.open
	t.Cleanup(m.Close)
	return m, op
}
```

[fixtures_test.go#L175-L183 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/fixtures_test.go#L175-L183)

`m.openRunnable = op.open` is a common Go testing **seam**: production code calls a
function stored in a field or package variable, and the test replaces it. You will find
several, e.g. `var createORTSession = createSession` in the engine
([ort.go#L263-L264](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/ort.go#L263-L264))
and `newEngineSession` swapped in
[pool_threads_test.go#L56-L74](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/pool_threads_test.go#L56-L74)
together with `t.Setenv`. It is Go's version of `unittest.mock.patch`, but explicit. The
HTTP tests use the other approach, a fake behind an interface (chapter 3).

## Oracle tests: fast code against obvious code

The NMS uses a grid index to be fast on 16 800 SCRFD proposals. That code is tricky, so the
test keeps the textbook all-pairs NMS as an **oracle** and checks that both keep exactly the
same boxes, on random data, for every option combination:

```go title="internal/vision/nms/nms_test.go (lines 137-171, trimmed)"
func TestMatchesNaive(t *testing.T) {
	for _, n := range []int{300, 3000} {
		r := rand.New(rand.NewSource(int64(n)))
		dets := make([]api.Detection, n)
		// ... random boxes, two classes, plus a giant box, a zero-area box, negative coordinates
		for _, thr := range []float64{0.3, 0.5, 0.7} {
			for _, o := range []Options{
				{IoU: thr},
				{IoU: thr, ClassAgnostic: true},
				{IoU: thr, Containment: true},
				{IoU: thr, ClassAgnostic: true, Containment: true, TopK: 17},
				{IoU: thr, TopK: 1},
			} {
				got, want := Detections(dets, o), naive(dets, o)
				if len(got) != len(want) {
					t.Fatalf("n=%d %+v: kept %d, naive kept %d", n, o, len(got), len(want))
				}
				// ... compare box by box
			}
		}
	}
}
```

[nms_test.go#L137-L171 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/nms/nms_test.go#L137-L171)

`rand.NewSource(seed)` makes the "random" data the same on every run, so a failure is
reproducible. This pattern is very useful for model code: keep a slow, obviously correct
reference (often a port of the Python implementation) and test the optimised version
against it.

## Golden files in `testdata/`

A directory named `testdata` is ignored by the Go build, so it is the standard place for
test inputs and expected outputs. A **golden file** stores the current output; the test
compares against it, and a flag regenerates it when a change is intended:

```go title="internal/vision/preprocess/sync_test.go (lines 14, 70-121, trimmed)"
var update = flag.Bool("update", false, "rewrite testdata/geometry_sync.json from the Go implementation")

func TestGeometrySyncCorpus(t *testing.T) {
	// ...
	var got []geometryCase
	for _, sm := range geometrySpecs {
		s := specFromMap(sm)
		for i, sz := range geometrySizes {
			ten, meta, err := s.Apply(nrgbaImage(sz[0], sz[1], int64(i)))
			// ...
			got = append(got, geometryCase{Spec: sm, Image: sz, Shape: ten.Shape, Meta: meta})
		}
	}
	path := filepath.Join("testdata", "geometry_sync.json")
	// ...
	if *update {
		// ... write got to path
		return
	}
	raw, err := os.ReadFile(path)
	// ... unmarshal and compare case by case
}
```

[sync_test.go#L14-L121 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/sync_test.go#L14-L121)

This one does double duty: the Python converter's tests read the **same** JSON file, so the
Go and Python preprocessing cannot drift apart. The package `internal/models/golden` pins
the end-to-end pre/postprocess output of most model packages the same way, with synthetic
ONNX outputs of the real shapes; regenerate with `go test ./internal/models/golden -update`
only when a behaviour change is intended
([harness_test.go#L1-L15](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/golden/harness_test.go#L1-L15)).

## Benchmarks

A function `BenchmarkXxx(b *testing.B)` runs its loop `b.N` times; Go picks `b.N` until the
timing is stable:

```go title="internal/vision/nms/nms_test.go (lines 83-89)"
func BenchmarkNMS16800(b *testing.B) {
	dets := benchDets(16800)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Detections(dets, Options{IoU: 0.4})
	}
}
```

[nms_test.go#L83-L89 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/nms/nms_test.go#L83-L89)

`b.ResetTimer()` excludes the setup. Benchmarks only run with `-bench`; `-run XXX` (a regexp
that matches no test) skips the ordinary tests:

```console
$ go test ./internal/vision/nms -bench NMS16800 -run XXX -benchmem
BenchmarkNMS16800-48    	      20	  59304845 ns/op	 3323646 B/op	    4025 allocs/op
```

About 59 ms per call for 16 800 boxes on the development machine (48 is the number of
CPUs). The comment above `Detections` records that the previous all-pairs version took 2.1 s.
`internal/models/classification/preprocess_bench_test.go` benchmarks JPEG decoding and
preprocessing the same way.

## Tests that only build under some conditions

Some preprocessing sweeps are 25 times slower under the race detector. Two tiny files,
selected by a **build tag** (chapter 7), tell the test which mode it runs in:

```go title="internal/vision/preprocess/race_on_test.go"
//go:build race

package preprocess_test

// raceEnabled: under -race the sweeps run their short sets (the race detector slows imaging
// ~25x); the full bit-equivalence sweep runs in the normal test run.
const raceEnabled = true
```

[race_on_test.go](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/race_on_test.go),
[race_off_test.go](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/race_off_test.go)
(`//go:build !race`, `const raceEnabled = false`).

## Try it

1. Run one test, then one subtest, then everything in a package with coverage:
   ```bash
   go test ./internal/vision/nms/ -run TestContainment -v
   go test ./internal/models/groundingdino -run 'TestPhraseSpansEdgeCases/single_class' -v
   go test -cover ./internal/vision/nms/
   ```
2. Add a case to a table. In `TestPhraseSpansEdgeCases`, add
   `{"three classes", "cup. remote. phone.", []string{"cup", "remote", "phone"}},` and rerun.
   Then change one expected word to see the failure message, and undo.
3. Run the benchmark and compare with and without `-benchmem`:
   ```bash
   go test ./internal/vision/nms -bench NMS16800 -run XXX -benchmem
   ```
4. Run the golden-file test; it checks the corpus Python also reads:
   ```bash
   go test ./internal/vision/preprocess -run GeometrySync -v
   ```
5. Run the whole suite as CI does (`.github/workflows/ci.yml`):
   ```bash
   go vet ./... && go test ./... -count=1
   ```
   Tests that need ONNX Runtime or downloaded weights skip themselves unless
   `ORT_DYLIB_PATH` is set and the weights exist.

## Recap

- Tests are `TestXxx(t *testing.T)` functions in `_test.go` files; compare with `if`, report
  with `t.Errorf` (continue) or `t.Fatalf` (stop).
- Write table-driven tests; use `t.Run` to name and isolate cases.
- `t.Helper`, `t.Skip`, `t.TempDir`, `t.Cleanup`, `t.Setenv` replace most fixtures.
- Replace a dependency with a fake behind an interface, or swap a function variable (a seam).
- Keep a slow oracle to test fast code; keep golden files in `testdata/` with an `-update` flag.
- `-bench` runs benchmarks, `-race` finds data races, `-count=1` bypasses the cache.
