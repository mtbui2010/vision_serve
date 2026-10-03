# 1. Packages, modules, building

!!! abstract "What you'll learn"
    - How Go code is grouped into packages and a module, and how imports work.
    - Why some names start with a capital letter (exported) and others do not.
    - Variables, `:=`, constants and `iota`.
    - Slices, arrays and maps, compared with Python lists, numpy arrays and dicts.
    - `for`, `range`, `switch`, and functions that return several values.

## Packages and the module

In Go, **one directory is one package**. Every `.go` file in `internal/vision/nms/` starts
with `package nms`, and together they form that package. There is no `__init__.py`: the
directory is enough.

The **module** is the whole repository. Its name comes from the first line of `go.mod`:
`module visionserve`. An import path is the module name plus the directory, so the
package in `internal/vision/nms/` is imported as `visionserve/internal/vision/nms`.

A package named `main` with a function `main()` becomes a program. VisionServe has
exactly one:

```go title="cmd/visionserve/main.go (lines 1-40, trimmed)"
// Command visionserve is the lightweight binary (server + CLI) for VisionServe.
package main

import (
	"fmt"
	"os"
	"strings"
	"syscall"

	"visionserve/internal/cli"

	// Blank-import the model packages so their init() registers a factory in the registry.
	// Adding a new model = add one import line here (do NOT modify other core code).
	_ "visionserve/internal/models/background"
	_ "visionserve/internal/models/classification"
	// ...
	_ "visionserve/internal/models/textalign"
)

func main() {
	ensureSignalSafe()
	if err := cli.Execute(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
```

[main.go#L1-L40 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/cmd/visionserve/main.go#L1-L40)

Things to notice:

- The import block lists the standard library first (`fmt`, `os`), then this module's
  packages. `goimports` sorts them for you.
- You call a package's function with its last path element: `cli.Execute(...)`,
  `fmt.Fprintln(...)`.
- `_ "visionserve/internal/models/classification"` is a **blank import**: "load this
  package for its side effects, I will not call it by name". Chapter 3 shows the side
  effect (model registration).
- You can rename an import. The classification model does this to keep lines short:

```go title="internal/models/classification/preprocess.go (lines 3-9)"
import (
	"image"

	"visionserve/internal/engine"
	"visionserve/internal/models"
	prep "visionserve/internal/vision/preprocess"
)
```

[preprocess.go#L3-L9 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/classification/preprocess.go#L3-L9)

Outside packages (`github.com/disintegration/imaging`, `gopkg.in/yaml.v3`) are listed in
`go.mod` with a version; `go.sum` stores their hashes so every build gets the same bytes.
It is the same idea as a `requirements.txt` with pinned hashes, but the Go tool manages it
for you (`go mod tidy`).

!!! warning "Go refuses unused imports and variables"
    An import you do not use, or a local variable you assign but never read, is a
    **compile error**, not a warning. This feels strict at first; it keeps the code clean.
    Use `_` when you really want to throw a value away: `_, err := f()`.

## Exported and unexported names

Go has no `public` / `private` keywords and no `_name` convention. The rule is the
**first letter**:

- `Detections`, `Options`, `IoU` start with a capital letter: **exported**, usable from
  other packages as `nms.Detections`.
- `box`, `fmax`, `gridMin` start with a lower-case letter: **unexported**, visible only
  inside package `nms`.

```go title="internal/vision/nms/nms.go (trimmed)"
// Options configures Detections.
type Options struct {
	// IoU is the suppression threshold: a box is suppressed when its overlap with an
	// already-kept, higher-scoring box is STRICTLY greater than IoU.
	IoU float64
	// ClassAgnostic lets boxes of different Class suppress each other. The default
	// (false) only suppresses within the same Class.
	ClassAgnostic bool
	// ...
}

func Detections(dets []api.Detection, o Options) []api.Detection {
	// ...
}

type box struct {
	x1, y1, x2, y2, area float64
	cls                  int
}
```

[nms.go#L15-L30](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/nms/nms.go#L15-L30),
[#L42](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/nms/nms.go#L42),
[#L120-L123](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/nms/nms.go#L120-L123)

The same rule applies to struct fields. This matters for JSON: `encoding/json` can only
see exported fields, which is why every field of `api.Result` starts with a capital letter
(chapter 2).

## Variables, `:=` and constants

```go
var sum float64          // declare with a type; starts at the zero value 0
k := cfg.MaxDet          // declare AND assign; the type (int) is inferred
k = 5                    // plain assignment to an existing variable
```

`:=` is the form you will see most. It only works inside functions and needs at least one
new variable on the left. Every type has a **zero value** (`0`, `""`, `false`, `nil`), so a
declared variable is never "undefined" like a Python name before assignment.

Constants are computed at compile time. Go has no enums; a group of constants with
`iota` (0, 1, 2, ... inside one `const` block) plays that role:

```go title="internal/vision/preprocess/apply.go (lines 12-19)"
// padKind says how render fills the frame outside the content.
type padKind int

const (
	noPad     padKind = iota // the content covers the whole frame
	pixelPad                 // PadValue is a pixel gray level, normalised like the image
	tensorPad                // PadValue is written into the normalised tensor as is
)
```

[apply.go#L12-L19 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/apply.go#L12-L19)

More often the project uses *string* constants of a named type, so the JSON and YAML stay
readable: `TaskDetection Task = "detection"` in
[pkg/api/types.go#L9-L18](https://github.com/mtbui2010/vision_serve/blob/main/pkg/api/types.go#L9-L18)
or `ProviderCUDA Provider = "cuda"` in
[engine/provider.go#L15-L22](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/provider.go#L15-L22).

## Functions with several results

A Go function can return several values. Geometry helpers use it a lot. Here the results
are even **named**, which documents them and declares them as variables:

```go title="internal/vision/preprocess/geometry.go (lines 9-25)"
// LetterboxSize: scale = min(W/w, H/h), new sides rounded half up (int(x+0.5), at least 1),
// centred: pad = (W-nw)/2, (H-nh)/2 floored.
func LetterboxSize(w, h, W, H int) (nw, nh int, scale float64, padX, padY int) {
	scale = float64(W) / float64(w)
	if s := float64(H) / float64(h); s < scale {
		scale = s
	}
	nw = int(float64(w)*scale + 0.5)
	nh = int(float64(h)*scale + 0.5)
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	return nw, nh, scale, (W - nw) / 2, (H - nh) / 2
}
```

[geometry.go#L9-L25 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/geometry.go#L9-L25)

Notice:

- `float64(W)`: Go never converts numbers implicitly. `W / w` with two `int`s is integer
  division, so the code converts first. In Python `/` always gives a float; in Go you say
  which one you want.
- `if s := ...; s < scale {`: an `if` can start with a short statement. `s` only exists
  inside the `if`. You will see this pattern everywhere, especially with errors:
  `if err := f(); err != nil { ... }`.
- The caller unpacks like a Python tuple: `nw, nh, scale, px, py := LetterboxSize(ow, oh, W, H)`
  ([apply.go#L44](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/apply.go#L44)).

## Slices: Go's lists (and numpy views)

A **slice** `[]float32` is a growable sequence of one type. It is the closest thing to a
Python list, but typed, and it behaves like a numpy *view*: slicing does not copy.

Softmax in the classification model shows the everyday operations:

=== "Go"

    ```go title="internal/models/classification/postprocess.go (lines 83-111)"
    // softmax converts logits to probabilities (numerically stable via max subtraction).
    func softmax(logits []float32) []float32 {
    	if len(logits) == 0 {
    		return nil
    	}

    	// Find max for numerical stability.
    	maxVal := logits[0]
    	for _, v := range logits[1:] {
    		if v > maxVal {
    			maxVal = v
    		}
    	}

    	out := make([]float32, len(logits))
    	var sum float64
    	for i, v := range logits {
    		e := math.Exp(float64(v) - float64(maxVal))
    		out[i] = float32(e)
    		sum += e
    	}
    	if sum > 0 {
    		inv := float32(1.0 / sum)
    		for i := range out {
    			out[i] *= inv
    		}
    	}
    	return out
    }
    ```

=== "Python (numpy)"

    ```python
    def softmax(logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return None
        e = np.exp(logits.astype(np.float64) - logits.max())
        return (e / e.sum()).astype(np.float32)
    ```

[postprocess.go#L83-L111 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/classification/postprocess.go#L83-L111)

There is no vectorised maths in Go: you write the loop. That is fine. A Go loop compiles to
machine code, so it is in the same league as the C loop inside numpy, not a slow Python loop.

| Python / numpy | Go |
|---|---|
| `[0.0] * n` / `np.zeros(n)` | `make([]float32, n)` (filled with zeros) |
| `xs.append(v)` | `xs = append(xs, v)` (always assign the result back) |
| `len(xs)` | `len(xs)` |
| `xs[1:]`, `xs[a:b]` | `xs[1:]`, `xs[a:b]` (a view: shares memory, like numpy) |
| `for v in xs:` | `for _, v := range xs {` |
| `for i, v in enumerate(xs):` | `for i, v := range xs {` |
| `for i in range(n):` | `for i := 0; i < n; i++ {` |
| `None` list | `nil` slice (length 0, safe to `range` over and `append` to) |

`for` is Go's only loop keyword. `for cond { }` is a while loop, and `for { }` loops forever
until `return` or `break`.

### Tensors are flat slices plus a shape

ONNX Runtime wants one flat buffer and a shape, so `engine.Tensor` is exactly that, like
`arr.ravel()` next to `arr.shape`:

```go title="internal/engine/tensor.go (lines 8-13)"
type Tensor struct {
	Data    []float32
	DataI64 []int64 // used when Dtype == "i64"
	Shape   []int64
	Dtype   string // "" or "f32" => float32; "i64" => int64
}
```

[tensor.go#L8-L13 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/engine/tensor.go#L8-L13)

The preprocessing code writes an NCHW tensor by cutting the flat buffer into three channel
planes. Each plane is a slice that *shares memory* with `data`, just like
`r, g, b = data.reshape(3, -1)` gives three numpy views:

```go title="internal/vision/preprocess/apply.go (lines 199-211)"
	default: // NCHW
		r, g, bl := data[:plane], data[plane:2*plane], data[2*plane:]
		for y := 0; y < ch; y++ {
			row := pix[(sy+y)*stride+sx*4 : (sy+y)*stride+(sx+cw)*4]
			o := (dy+y)*outW + dx
			for x := 0; x < cw; x++ {
				p := row[x*4 : x*4+3 : x*4+3]
				r[o] = lut[0][p[0]]
				g[o] = lut[1][p[1]]
				bl[o] = lut[2][p[2]]
				o++
			}
		}
```

[apply.go#L199-L211 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/apply.go#L199-L211)

`pix` is the image's raw RGBA bytes (4 per pixel), `lut` a 256-entry lookup table that
already holds `(p/255 - mean) / std` for every byte value.

### Arrays: fixed size

`[4]float64` (with a number) is an **array**: fixed length, copied on assignment, and
comparable with `==`. The project uses it for boxes, `BBox [4]float64` in `api.Detection`,
so a box can never have 3 or 5 numbers. Slices (`[]T`, no number) are what you use
everywhere else.

## Maps: Go's dicts

A map has one key type and one value type: `map[string]int`. The license gate is a map
literal, and reading it uses the **comma-ok** form, Go's answer to `key in d`:

```go title="internal/registry/manifest.go (lines 29-47, trimmed)"
var licenseAllowlist = map[string]string{
	"apache-2.0":   "Apache-2.0",
	"mit":          "MIT",
	"bsd-3-clause": "BSD-3-Clause",
	"bsd-2-clause": "BSD-2-Clause",
}

// ...
func canonicalLicense(declared string) (string, bool) {
	canon, ok := licenseAllowlist[strings.ToLower(strings.TrimSpace(declared))]
	return canon, ok
}
```

[manifest.go#L29-L47 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/manifest.go#L29-L47)

=== "Go"

    ```go
    canon, ok := licenseAllowlist[key]   // ok is false when key is missing
    v := m[key]                          // missing key gives the zero value, no KeyError
    m[key] = v                           // insert or overwrite
    delete(m, key)
    for k, v := range m { ... }          // random order!
    ```

=== "Python"

    ```python
    canon = allowlist.get(key); ok = key in allowlist
    v = m.get(key, 0)
    m[key] = v
    del m[key]
    for k, v in m.items(): ...           # insertion order
    ```

!!! note "Map order is random on purpose"
    Go randomises map iteration order, so code cannot depend on it by accident. When the
    order matters, collect the keys and sort them, as `models.Registered()` does
    ([model.go#L323-L333](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/model.go#L323-L333)).

## `switch`

A `switch` has no fall-through (no `break` needed), and a `switch` with no value is a
clean if/else-if chain. The classification postprocess checks the output shape this way:

```go title="internal/models/classification/postprocess.go (lines 30-42)"
	// Accept [1, C] or [C] — some ONNX exports drop the batch dimension.
	var numClasses int
	switch len(out.Shape) {
	case 2:
		if out.Shape[0] != 1 {
			return models.Result{}, fmt.Errorf("classification: expected batch size 1, got shape %v", out.Shape)
		}
		numClasses = int(out.Shape[1])
	case 1:
		numClasses = int(out.Shape[0])
	default:
		return models.Result{}, fmt.Errorf("classification: unexpected output tensor shape %v (expected [1,C] or [C])", out.Shape)
	}
```

[postprocess.go#L30-L42 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/classification/postprocess.go#L30-L42)

## Try it

1. Run the NMS tests and read one of them next to the code:
   ```bash
   go test ./internal/vision/nms/ -run TestEmptyAndSortedOutput -v
   ```
2. Write a tiny program that calls an internal package. Create `cmd/play/main.go`
   (delete it afterwards; do not commit it):
   ```go title="cmd/play/main.go"
   package main

   import (
   	"fmt"

   	"visionserve/internal/vision/nms"
   	"visionserve/pkg/api"
   )

   func main() {
   	dets := []api.Detection{
   		{BBox: [4]float64{0, 0, 100, 100}, Class: "cat", Conf: 0.9},
   		{BBox: [4]float64{10, 10, 100, 100}, Class: "cat", Conf: 0.8},
   		{BBox: [4]float64{300, 300, 50, 50}, Class: "dog", Conf: 0.7},
   	}
   	keep := nms.Detections(dets, nms.Options{IoU: 0.5})
   	for i, d := range keep {
   		fmt.Printf("%d: %s %.2f %v\n", i, d.Class, d.Conf, d.BBox)
   	}
   }
   ```
   ```console
   $ go run ./cmd/play
   0: cat 0.90 [0 0 100 100]
   1: dog 0.70 [300 300 50 50]
   ```
   The second cat box overlaps the first one too much and is suppressed.
3. Break it on purpose. Add these two lines after `keep := ...` and run again:
   ```go
   unused := len(keep)
   var b nms.box
   ```
   ```console
   $ go run ./cmd/play
   # visionserve/cmd/play
   cmd/play/main.go:17:2: unused declared and not used
   cmd/play/main.go:18:6: b declared and not used
   cmd/play/main.go:18:12: undefined: nms.box
   ```
   Unused variables are errors, and `box` is invisible outside package `nms` because it
   starts with a lower-case letter. Remove the lines, then `rm -r cmd/play`.

## Recap

- One directory = one package; the module name (`visionserve`) + directory = import path.
- `package main` + `func main()` = a program. `_` imports a package only for its side effects.
- Capital first letter = exported. This applies to functions, types and struct fields.
- `:=` declares and assigns; every type has a zero value; `iota` builds enum-like constants.
- Slices are typed lists that share memory when sliced (like numpy views); `[N]T` arrays
  are fixed size; maps are dicts with random iteration order.
- Functions return several values; `if x := ...; cond {}` keeps helper variables local.
