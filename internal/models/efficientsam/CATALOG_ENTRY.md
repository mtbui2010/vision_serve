# EfficientSAM — catalog entry

Add this entry to the catalog slice in `cmd/visionserve/catalog.go` (or wherever your
catalog is defined):

```go
{
    Name:         "efficient-sam",
    Task:         "segmentation",
    License:      "Apache-2.0",
    Architecture: "efficient-sam",
    Description:  "EfficientSAM ViT-Tiny — promptable segmentation, lighter than MobileSAM.",
    HFRepo:       "onnx-community/EfficientSAM",
    Files: []File{
        {Role: "encoder", HFFilename: "encoder_model.onnx",  LocalFilename: "efficient_sam_encoder.onnx", ManifestRole: "encoder"},
        {Role: "decoder", HFFilename: "decoder_model.onnx",  LocalFilename: "efficient_sam_decoder.onnx", ManifestRole: "decoder"},
    },
    InputWidth:          1024,
    InputHeight:         1024,
    InputLayout:         "NCHW",
    PostprocessType:     "sam",
    RuntimePrefer:       []string{"cuda", "cpu"},
    IdleUnloadSeconds:   300,
    Verified:            false,
    Note: "Verify HF file names match encoder/decoder split before using. " +
          "Encoder takes raw pixel/255 at original size (graph resizes + normalizes); " +
          "I/O verified on the local export — see efficientsam.go.",
},
```

Also add the blank import to `cmd/visionserve/main.go`:

```go
_ "visionserve/internal/models/efficientsam"
```
