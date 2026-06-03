# RT-DETR catalog entry

Add the following entry to the builtin slice in `internal/catalog/catalog.go`:

```go
{
    Name:         "rt-detr",
    Task:         "detection",
    License:      "Apache-2.0",
    Architecture: "rt-detr",
    Description:  "RT-DETR-l (COCO) — real-time NMS-free detector, 640×640.",
    HFRepo:       "onnx-community/RT-DETR-l-hf",
    Files: []File{
        {Role: "model", HFFilename: "onnx/model.onnx", LocalFilename: "model.onnx"},
    },
    InputWidth:      640,
    InputHeight:     640,
    InputLayout:     "NCHW",
    Letterbox:       true,
    Normalize:       &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
    PostprocessType: "rt-detr",
    BoxFormat:       "cxcywh",
    ConfThreshold:   0.5,
    MaxDetections:   300,
    LabelsFile:      "coco80.txt",
    RuntimePrefer:   []string{"tensorrt", "cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:        true,
},
```

Also add the blank import to `cmd/visionserve/main.go` alongside the rf-detr import:

```go
_ "visionserve/internal/models/rtdetr"
```
