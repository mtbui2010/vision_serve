# SCRFD Catalog Entry

Paste this block into the model catalog (e.g. `cmd/visionserve/catalog.go`) and add
the blank import to `main.go`.

```go
{
    Name:         "scrfd",
    Task:         "detection",
    License:      "MIT",
    Architecture: "scrfd",
    Description:  "SCRFD-10GF — InsightFace face detector, 640×640, MIT license.",
    HFRepo:       "deepinsight/insightface",
    Files: []File{
        {Role: "model", HFFilename: "models/buffalo_l/det_10g.onnx", LocalFilename: "det_10g.onnx"},
    },
    InputWidth:      640,
    InputHeight:     640,
    InputLayout:     "NCHW",
    Letterbox:       true,
    Normalize: &Normalize{
        Mean: []float32{127.5, 127.5, 127.5},
        Std:  []float32{128.0, 128.0, 128.0},
    },
    PostprocessType: "scrfd",
    ConfThreshold:   0.5,
    MaxDetections:   1000,
    RuntimePrefer:   []string{"tensorrt", "cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified: false,
    Note: "buffalo_l pack path unverified. Check deepinsight/insightface HF repo for exact file path.",
},
```

Add to `main.go`:

```go
_ "visionserve/internal/models/scrfd"
```
