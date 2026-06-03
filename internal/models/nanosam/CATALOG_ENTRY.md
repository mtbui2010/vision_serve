# NanoSAM catalog entry

Add to the model catalog (`internal/catalog/catalog.go` or equivalent) when the catalog
pull mechanism is wired up. Also add the blank import to `cmd/visionserve/main.go`.

## catalog.go entry

```go
{
    Name:         "nano-sam",
    Task:         "segmentation",
    License:      "MIT",
    Architecture: "nano-sam",
    Description:  "NanoSAM — NVIDIA edge-optimized SAM (ResNet-18 encoder), MIT license.",
    HFRepo:       "NVIDIA-AI-IOT/nanosam", // HF path TBD — weights are on GitHub, not HF
    Files: []File{
        {
            Role:          "encoder",
            HFFilename:    "nanosam_encoder.onnx",
            LocalFilename: "nanosam_encoder.onnx",
            ManifestRole:  "encoder",
        },
        {
            Role:          "decoder",
            HFFilename:    "nanosam_decoder.onnx",
            LocalFilename: "nanosam_decoder.onnx",
            ManifestRole:  "decoder",
        },
    },
    InputWidth:        1024,
    InputHeight:       1024,
    InputLayout:       "NCHW",
    PostprocessType:   "sam",
    RuntimePrefer:     []string{"tensorrt", "cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:          false,
    Note: "Weights available at github.com/NVIDIA-AI-IOT/nanosam — not on HuggingFace. " +
          "License MIT. Verify encoder input tensor name ('input' assumed) and decoder " +
          "tensor names against real ONNX files before setting Verified=true.",
},
```

## main.go blank import

```go
_ "visionserve/internal/models/nanosam"
```
