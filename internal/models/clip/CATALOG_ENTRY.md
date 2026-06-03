{
    Name: "clip",
    Task: "embed",
    License: "MIT",
    Architecture: "clip",
    Description: "CLIP ViT-B/32 image encoder — 512-d embeddings for zero-shot classification.",
    HFRepo: "onnx-community/clip-vit-base-patch32",
    Files: []File{
        {Role:"model",HFFilename:"vision_model/model.onnx",LocalFilename:"model.onnx"},
    },
    InputWidth:224, InputHeight:224, InputLayout:"NCHW",
    Letterbox: false,
    Normalize: &Normalize{
        Mean: []float32{0.48145466, 0.4578275, 0.40821073},
        Std: []float32{0.26862954, 0.26130258, 0.27577711},
    },
    PostprocessType: "embed",
    RuntimePrefer: []string{"tensorrt","cuda","cpu"},
    IdleUnloadSeconds: 300,
    Verified: false,
    Note: "V1: image encoder only. Text encoder (BPE tokenizer) planned for v2.",
},
// main.go: _ "visionserve/internal/models/clip"
