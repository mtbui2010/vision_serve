# Catalog Entries — Classification Models

These entries should be added to the `builtin` slice in
`internal/catalog/catalog.go` once a shared ImageNet-1k labels file is
embedded (similar to coco91.txt — embed `labels/imagenet1k.txt`).

## efficientnet-b0

```go
{
    Name:         "efficientnet-b0",
    Task:         "classification",
    License:      "Apache-2.0",
    Architecture: "efficientnet",
    Description:  "EfficientNet-B0 — ImageNet-1k image classifier (224×224).",
    HFRepo:       "onnx-community/efficientnet-b0",
    Files: []catalog.File{
        {
            Role:          "model",
            HFFilename:    "model.onnx",
            LocalFilename: "model.onnx",
        },
    },
    InputWidth:        224,
    InputHeight:       224,
    InputLayout:       "NCHW",
    Letterbox:         false,
    Normalize: &catalog.Normalize{
        Mean: []float32{0.485, 0.456, 0.406},
        Std:  []float32{0.229, 0.224, 0.225},
    },
    PostprocessType:   "classification",
    MaxDetections:     5,
    LabelsFile:        "imagenet1k.txt",
    EmbeddedLabels:    imagenet1k, // embed labels/imagenet1k.txt (add //go:embed directive)
    RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:          true,
},
```

## mobilenet-v3

```go
{
    Name:         "mobilenet-v3",
    Task:         "classification",
    License:      "Apache-2.0",
    Architecture: "mobilenet-v3",
    Description:  "MobileNet-V3 Small — lightweight ImageNet-1k classifier (224×224).",
    HFRepo:       "onnx-community/mobilenet_v3_small",
    Files: []catalog.File{
        {
            Role:          "model",
            HFFilename:    "model.onnx",
            LocalFilename: "model.onnx",
        },
    },
    InputWidth:        224,
    InputHeight:       224,
    InputLayout:       "NCHW",
    Letterbox:         false,
    Normalize: &catalog.Normalize{
        Mean: []float32{0.485, 0.456, 0.406},
        Std:  []float32{0.229, 0.224, 0.225},
    },
    PostprocessType:   "classification",
    MaxDetections:     5,
    LabelsFile:        "imagenet1k.txt",
    EmbeddedLabels:    imagenet1k, // embed labels/imagenet1k.txt (add //go:embed directive)
    RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:          true,
},
```

## Setup note

To embed the labels file, add to `catalog.go`:

```go
//go:embed labels/imagenet1k.txt
var imagenet1k string
```

Then place the 1000-line ImageNet class name file at
`internal/catalog/labels/imagenet1k.txt`.
