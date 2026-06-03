{
    Name: "sam2",
    Task: "segmentation",
    License: "Apache-2.0",
    Architecture: "sam2",
    Description: "SAM2-Tiny — promptable segmentation with multi-scale features.",
    HFRepo: "jf-11/sam2-image-onnx",
    Files: []File{
        {Role:"encoder",HFFilename:"sam2_tiny_encoder.onnx",LocalFilename:"sam2_tiny_encoder.onnx",ManifestRole:"encoder"},
        {Role:"decoder",HFFilename:"sam2_tiny_decoder.onnx",LocalFilename:"sam2_tiny_decoder.onnx",ManifestRole:"decoder"},
    },
    InputWidth:1024, InputHeight:1024, InputLayout:"NHWC",
    PostprocessType:"sam",
    RuntimePrefer:[]string{"tensorrt","cuda","cpu"},
    IdleUnloadSeconds:300,
    Verified: false,
    Note: "Community ONNX export — verify HF repo license before use. Tensor names may vary; check against your ONNX file.",
},
// Also add to cmd/visionserve/main.go: _ "visionserve/internal/models/sam2"
