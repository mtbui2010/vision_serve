# Catalog entry — paddle-ocr

Add the following entry to the model catalog and the blank import to `cmd/visionserve/main.go`.

## Catalog struct

```go
{
    Name:         "paddle-ocr",
    Task:         "detection",
    License:      "Apache-2.0",
    Architecture: "paddle-ocr",
    Description:  "PP-OCRv4 — Chinese+English OCR (text detection + recognition, no direction cls).",
    HFRepo:       "paddlepaddle/pp-ocrv4",
    Files: []File{
        {Role: "det", HFFilename: "ch_PP-OCRv4_det_infer.onnx", LocalFilename: "det_model.onnx", ManifestRole: "det"},
        {Role: "rec", HFFilename: "ch_PP-OCRv4_rec_infer.onnx", LocalFilename: "rec_model.onnx", ManifestRole: "rec"},
        {Role: "keys", HFFilename: "ppocr_keys_v1.txt", LocalFilename: "ppocr_keys_v1.txt"},
    },
    InputWidth:       960,
    InputHeight:      960,
    InputLayout:      "NCHW",
    Letterbox:        false,
    PostprocessType:  "paddle-ocr",
    ConfThreshold:    0.3,
    RuntimePrefer:    []string{"cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:         false,
    Note: "Verify HF repo path for ONNX files. ppocr_keys_v1.txt must be present in model dir.",
},
```

## Blank import for main.go

```go
// in cmd/visionserve/main.go, add:
_ "visionserve/internal/models/paddleocr"
```
