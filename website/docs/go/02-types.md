# 2. Types, structs, methods

!!! abstract "What you'll learn"
    - Structs, Go's version of a dataclass, and how the unified `Result` schema is one.
    - Struct tags: how the same struct becomes JSON on the wire and is read from YAML manifests.
    - Zero values, and why the project designs types so that "empty" means "default".
    - Methods, and when a method takes a pointer (`*T`) or a copy (`T`).
    - Embedding (composition instead of inheritance) and type aliases.

## Structs

A struct is a fixed set of named, typed fields. It is what you would write as a
`@dataclass` (or a pydantic model) in Python. The whole public API of VisionServe is a
handful of structs in `pkg/api`:

=== "Go"

    ```go title="pkg/api/types.go (lines 52-58)"
    // Detection is a bbox with class + confidence.
    // BBox is ALWAYS in ORIGINAL image coordinates: [x, y, w, h] (top-left corner + width/height).
    type Detection struct {
    	BBox  [4]float64 `json:"bbox"`
    	Class string     `json:"class"`
    	Conf  float64    `json:"conf"`
    }
    ```

=== "Python"

    ```python
    @dataclass
    class Detection:
        """BBox is ALWAYS in ORIGINAL image coordinates: [x, y, w, h]."""
        bbox: tuple[float, float, float, float]
        class_: str          # JSON key "class"
        conf: float
    ```

[types.go#L52-L58 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/pkg/api/types.go#L52-L58)

You build a struct value with a **composite literal**, naming the fields:

```go
d := api.Detection{BBox: [4]float64{10, 20, 30, 40}, Class: "cat", Conf: 0.91}
d.Conf = 0.95             // fields are read and written with a dot
fmt.Printf("%+v\n", d)    // {BBox:[10 20 30 40] Class:cat Conf:0.95}
```

Fields you leave out get their zero value. Every task returns the same `Result` struct;
a detector fills `Detections`, a classifier `Classifications`, a depth model `DepthMap`:

```go title="pkg/api/types.go (lines 20-44, trimmed)"
// Result is the normalized output — a unified schema across tasks.
type Result struct {
	Task            Task             `json:"task"`
	Model           string           `json:"model"`
	Device          string           `json:"device,omitempty"` // "cpu" | "gpu:0" | "gpu:0+trt"
	Hint            string           `json:"hint,omitempty"`   // setup recommendation (e.g. install TRT)
	Detections      []Detection      `json:"detections,omitempty"`
	Masks           []Mask           `json:"masks,omitempty"`
	// ...
	Classifications []Classification `json:"classifications,omitempty"` // top-K class predictions
	// ...
	DurationMs      float64          `json:"duration_ms"`
	// ...
}
```

[types.go#L20-L44 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/pkg/api/types.go#L20-L44)

## Struct tags: JSON out, YAML in

The text in back-quotes after a field is a **tag**. Go ignores it; libraries read it.
`encoding/json` uses `json:"..."` to pick the key name, and `omitempty` drops the field
when it holds its zero value (an empty slice, `""`, `0`). That is how one `Result` type
serves every task without sending empty `"masks": []` to a detection client.

The manifest parser uses the same trick with `yaml:"..."` tags. Nested YAML maps become
nested structs, which can be written inline:

```go title="internal/registry/manifest.go (lines 113-229, trimmed)"
type Manifest struct {
	Name      string `yaml:"name"`
	Task      string `yaml:"task"`
	License   string `yaml:"license"`
	ModelFile string `yaml:"model_file"`
	// ...
	Input struct {
		Width     int    `yaml:"width"`
		Height    int    `yaml:"height"`
		Layout    string `yaml:"layout"`
		Letterbox bool   `yaml:"letterbox"`
		// ...
		Normalize  struct {
			Mean []float32 `yaml:"mean"`
			Std  []float32 `yaml:"std"`
		} `yaml:"normalize"`
	} `yaml:"input"`
	// ...
}
```

[manifest.go#L113-L229 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/manifest.go#L113-L229)

So this manifest fragment fills `m.Input.Normalize.Mean`:

```yaml
input:
  width: 224
  normalize:
    mean: [0.485, 0.456, 0.406]
```

Reading the file is then two calls. Note `&m`: the decoder needs the *address* of `m` so
it can write into it (more on pointers below):

```go title="internal/registry/manifest.go (lines 231-240)"
// LoadManifest reads + parses + validates a manifest.yaml file.
func LoadManifest(path string) (*Manifest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("registry: failed to read manifest %s: %w", path, err)
	}
	var m Manifest
	if err := yaml.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("registry: failed to parse YAML %s: %w", path, err)
	}
```

[manifest.go#L231-L251 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/manifest.go#L231-L251)

!!! tip "Unknown YAML keys are ignored"
    A misspelt key (`idle_unload_second:`) does not fail; it simply leaves the field at its
    zero value. When a manifest "does nothing", compare its keys with the tags above. The
    tag `yaml:"-"` (used on `dir` at
    [L222-L223](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/manifest.go#L222-L223))
    means "never read this field from YAML".

## Zero values: empty means default

Every Go type has a zero value, and a struct's zero value has every field at zero. The
project leans on this deliberately. The preprocessing spec says so in its doc comment:

```go title="internal/vision/preprocess/spec.go (lines 96-137, trimmed)"
// Spec declares one model's preprocessing. The zero value of each field is the common case.
// ...
type Spec struct {
	// Resize: the geometry; "" = the architecture's default (Arch.Resolve fills it in).
	Resize Mode
	// Width, Height: the target size (unused by None). LongSide/LongSidePad scale by
	// min(Width/w, Height/h).
	Width, Height int
	// ...
	// Resample: "" = the mode's default (see Mode).
	Resample Resample
	// Mean, Std: per-channel normalisation, RGB order (see the formula above).
	Mean, Std []float32
	// NoRescale: keep pixels in 0..255 instead of dividing by 255 (HuggingFace do_rescale=False).
	NoRescale bool
	// Layout: "" = NCHW.
	Layout Layout
	// ...
}
```

[spec.go#L96-L137 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/spec.go#L96-L137)

`NoRescale bool` is named so that its zero value (`false`) is the usual case (divide by 255).
A manifest that does not mention it gets the right behaviour without a default value
anywhere. In Python you would write `no_rescale: bool = False`; in Go you choose the field's
*meaning* so that the zero value is the default.

## Named types and methods

You can define a new type on top of a basic one and attach methods to it. The resize
mode is a string, but a *typed* one:

```go title="internal/vision/preprocess/spec.go (lines 22-29, 57, 62-64, trimmed)"
// Mode is how an image is brought to the model's input size. Only modes that a served
// architecture really uses exist; each one reproduces its upstream recipe exactly.
type Mode string

const (
	// Squash resizes to exactly Width×Height; the aspect ratio is not kept (RF-DETR, MiDaS,
	// EfficientNet, SAM2, …). Default resample: bilinear.
	Squash Mode = "squash"
	// ...
)

// Pads reports whether the mode keeps the aspect ratio and fills an area outside the resized
// image — what the legacy `input.letterbox: true` ("keep aspect ratio + pad") describes.
func (m Mode) Pads() bool { return m == Letterbox || m == TopLeftPad || m == LongSidePad }
```

[spec.go#L22-L29](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/spec.go#L22-L29),
[#L62-L64](https://github.com/mtbui2010/vision_serve/blob/main/internal/vision/preprocess/spec.go#L62-L64)

`(m Mode)` before the name is the **receiver**: Go's `self`, but written explicitly and
named by you (usually one or two letters). You call it as `spec.Resize.Pads()`. A function
that expects a `Mode` will not accept a plain `string` variable by mistake, which catches
bugs a type checker in Python would only catch with `Literal[...]` or an `Enum`.

A named type can even control how it is decoded. `wholeNumber` refuses `1.5` where an
integer is expected, because yaml.v3 would otherwise truncate it silently:

```go title="internal/registry/manifest.go (lines 401-415)"
// wholeNumber is an int that refuses a YAML float: yaml.v3 truncates `1.5` into an int field
// silently, and a thread count of 1 written as 1.5 should be an error, not a guess.
type wholeNumber int

func (w *wholeNumber) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind != yaml.ScalarNode || n.ShortTag() != "!!int" {
		return fmt.Errorf("line %d: %q is not an integer", n.Line, n.Value)
	}
	var v int
	if err := n.Decode(&v); err != nil {
		return err
	}
	*w = wholeNumber(v)
	return nil
}
```

[manifest.go#L401-L415 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/manifest.go#L401-L415)

This is the Go equivalent of a pydantic validator. yaml.v3 sees the method and calls it.

## Pointer or value receiver?

Notice `(m Mode)` above but `(w *wholeNumber)` here. The star makes the receiver a
**pointer**:

| Receiver | What the method gets | Use it when |
|---|---|---|
| `func (s Spec) Apply(...)` | a **copy** of the value | the method only reads; the type is small or meant to be a value |
| `func (m *Manifest) validate()` | the **address** of the original | the method changes the value, the struct is large, or it holds a mutex |

In Python every object is passed by reference, so `self.license = ...` always changes the
original. In Go a value receiver changes only its copy. `validate` must write the
canonical license back, so it takes a pointer:

```go title="internal/registry/manifest.go (lines 263-269)"
	// License: required + must be in the permissive allowlist (case-insensitive match,
	// stored back in canonical SPDX form so later == comparisons see one spelling).
	canonLicense, ok := canonicalLicense(m.License)
	if !ok {
		return fmt.Errorf("license %q is not allowed — only permissive licenses accepted (Apache-2.0/MIT/BSD); AGPL is strictly forbidden", m.License)
	}
	m.License = canonLicense
```

[manifest.go#L253-L269 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/manifest.go#L253-L269)

`preprocess.Spec`, `preprocess.Meta` and `engine.Tensor` use value receivers: they are
plain data, and `Tensor` only holds slice *headers* (pointer + length), so copying it does
not copy the pixels. `lifecycle.Manager` and `engine.Session` always use pointer receivers:
they contain a `sync.Mutex`, and copying a mutex breaks it (`go vet` warns about this).

!!! note "Pointers in one paragraph"
    `&x` takes the address of `x`; `*T` is the type "pointer to T"; `p.Field` works through
    a pointer without writing `(*p).Field`. A pointer can be `nil`. The project uses that for
    *optional* values: `Preprocess *preprocess.Spec` in `models.Config` is `nil` when the
    manifest has no `preprocess:` block, like `Optional[Spec] = None`
    ([model.go#L73-L77](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/model.go#L73-L77)).
    There is no pointer arithmetic in normal Go code.

## Embedding: composition, not inheritance

Go has no classes and no inheritance. Instead, a struct can **embed** another type by
writing the type without a field name. The embedded type's fields and methods are
*promoted*: you can use them as if they were declared on the outer struct.

The HTTP request type embeds the public JSON request:

```go title="internal/server/request.go (lines 47-51)"
type Request struct {
	api.PredictJSONRequest

	data *formData // nil for a Request built in code (`visionserve run`)
}
```

[request.go#L47-L51 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/server/request.go#L47-L51)

So the handler can write `q.Model` and `q.Encoding`
([handlers.go#L163-L164](https://github.com/mtbui2010/vision_serve/blob/main/internal/server/handlers.go#L163-L164))
although those fields belong to `api.PredictJSONRequest`.

The tests use embedding to build a variant of a fake model that adds one method:

```go title="internal/lifecycle/fixtures_test.go (lines 34-42)"
// exclusivePipe is a testPipe that asks the runtime to serialize its Infer (models.Exclusive).
type exclusivePipe struct{ testPipe }

func (p *exclusivePipe) Exclusive() bool { return true }

// notExclusivePipe implements models.Exclusive but says no: it must run concurrently.
type notExclusivePipe struct{ testPipe }

func (p *notExclusivePipe) Exclusive() bool { return false }
```

[fixtures_test.go#L34-L42 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/lifecycle/fixtures_test.go#L34-L42)

`exclusivePipe` has every method of `testPipe` (`Name`, `Task`, `Roles`, `Infer`) plus
`Exclusive`. In Python you would subclass; in Go you embed. The difference: there is no
`super()` and no overriding chain. If you need to change a method, you define it on the
outer type and, if needed, call `p.testPipe.Infer(...)` explicitly.

## Type aliases

`type A = B` (with `=`) is an **alias**: a second name for the very same type. Without `=`
(`type Mode string`) you get a *new* type. The model packages use aliases so that model code
can write `models.Result` while there is still only one schema, the one in `pkg/api`:

```go title="internal/models/model.go (lines 19-28)"
// The types below are aliases to the public schema in pkg/api, so model
// implementations can use short names while keeping a SINGLE unified schema.
type (
	Task           = api.Task
	Result         = api.Result
	Detection      = api.Detection
	Mask           = api.Mask
	Grasp          = api.Grasp
	Classification = api.Classification
)
```

[model.go#L19-L28 on GitHub](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/model.go#L19-L28)

`models.PreprocessMeta = preprocess.Meta` works the same way
([meta.go#L9](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/meta.go#L9)).
This is how CLAUDE.md's rule "do NOT invent a per-model schema" is enforced by the type
system: a `models.Result` *is* an `api.Result`.

??? note "Generics, briefly"
    Go has had generics since 1.18. The project uses them sparingly, e.g. MobileSAM's
    automatic mask generator collects results of any type `T`:
    ```go title="internal/models/mobilesam/automask.go (lines 77-82)"
    // amgOut is one final-pass result after emit has converted it.
    type amgOut[T any] struct {
    	v     T
    	valid bool
    	err   error
    }
    ```
    [automask.go#L77-L82](https://github.com/mtbui2010/vision_serve/blob/main/internal/models/mobilesam/automask.go#L77-L82).
    `[T any]` is like `Generic[T]` in Python typing. You will rarely need to write generic code
    for a model.

## Try it

1. Read a type's documentation and fields:
   ```bash
   go doc ./pkg/api Result
   go doc ./internal/vision/preprocess Spec
   ```
2. See `omitempty` and the zero value at work. Put this in `cmd/play/main.go` (throwaway):
   ```go title="cmd/play/main.go"
   package main

   import (
   	"encoding/json"
   	"fmt"

   	"visionserve/pkg/api"
   )

   func main() {
   	var empty api.Result // the zero value
   	res := api.Result{
   		Task:  api.TaskDetection,
   		Model: "rf-detr",
   		Detections: []api.Detection{
   			{BBox: [4]float64{10, 20, 30, 40}, Class: "cat", Conf: 0.91},
   		},
   	}
   	for _, r := range []api.Result{empty, res} {
   		b, err := json.Marshal(r)
   		if err != nil {
   			panic(err) // fine in a throwaway program, never in the server
   		}
   		fmt.Println(string(b))
   	}
   	fmt.Printf("%+v\n", res.Detections[0])
   }
   ```
   ```console
   $ go run ./cmd/play
   {"task":"","model":"","duration_ms":0}
   {"task":"detection","model":"rf-detr","detections":[{"bbox":[10,20,30,40],"class":"cat","conf":0.91}],"duration_ms":0}
   {BBox:[10 20 30 40] Class:cat Conf:0.91}
   ```
   `task`, `model` and `duration_ms` have no `omitempty`, so they always appear; the empty
   `detections`, `masks`, ... vanish.
3. Run the license-gate tests; they build `Manifest` values in code and call the
   pointer-receiver `validate()`:
   ```bash
   go test ./internal/registry -run 'TestValidate' -v
   ```
   Then change `"AGPL-3.0"` in `TestValidateRejectsAGPL`
   ([manifest_test.go#L10](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/manifest_test.go#L10))
   to `"MIT"` and watch it fail. Undo the change.

## Recap

- A struct is a typed record; composite literals `T{Field: v}` build values, missing fields are zero.
- Tags (`json:"bbox,omitempty"`, `yaml:"model_file"`) map fields to wire and file formats.
- Design fields so the zero value is the sensible default.
- Methods have an explicit receiver. Use `*T` to modify, for big structs, and for anything
  holding a mutex; use `T` for small read-only data.
- Embedding promotes the inner type's fields and methods: composition instead of inheritance.
- `type A = B` is an alias (same type); `type A B` is a new type.
