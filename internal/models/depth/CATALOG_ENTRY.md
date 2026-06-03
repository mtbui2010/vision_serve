# Catalog Entries — Depth Models

These entries should be added to the `builtin` slice in
`internal/catalog/catalog.go` once the HuggingFace sources are confirmed.

## depth-anything-v2

```go
{
    Name:         "depth-anything-v2",
    Task:         "depth",
    License:      "Apache-2.0",
    Architecture: "depth-anything-v2",
    Description:  "Depth Anything V2 Small — monocular depth estimation (518×518).",
    HFRepo:       "onnx-community/depth-anything-v2-small-hf",
    Files: []catalog.File{
        {
            Role:          "model",
            HFFilename:    "onnx/model.onnx",
            LocalFilename: "model.onnx",
        },
    },
    InputWidth:        518,
    InputHeight:       518,
    InputLayout:       "NCHW",
    Letterbox:         false,
    Normalize: &catalog.Normalize{
        Mean: []float32{0.485, 0.456, 0.406},
        Std:  []float32{0.229, 0.224, 0.225},
    },
    PostprocessType:   "depth",
    RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:          true,
},
```

## midas

```go
{
    Name:         "midas",
    Task:         "depth",
    License:      "MIT",
    Architecture: "midas",
    Description:  "MiDaS Small — lightweight monocular depth estimation (256×256).",
    HFRepo:       "isl-org/MiDaS",
    Files: []catalog.File{
        {
            Role:          "model",
            HFFilename:    "midas_v21_small_256.onnx", // TODO: confirm exact ONNX filename
            LocalFilename: "model.onnx",
        },
    },
    InputWidth:        256,
    InputHeight:       256,
    InputLayout:       "NCHW",
    Letterbox:         false,
    Normalize: &catalog.Normalize{
        Mean: []float32{0.485, 0.456, 0.406},
        Std:  []float32{0.229, 0.224, 0.225},
    },
    PostprocessType:   "depth",
    RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:          false,
    Note:              "Exact ONNX filename in isl-org/MiDaS not yet confirmed — verify before pulling.",
},
```
