package engine

// Tensor is a flat data buffer + shape, independent of ONNX Runtime.
//
// Most models use float32 (normalized image input, RF-DETR/SAM logits & boxes).
// Some models (e.g. GroundingDINO text tokens, attention masks) need int64 inputs:
// set DataI64 and Dtype="i64" for those. Outputs are currently always float32.
type Tensor struct {
	Data    []float32
	DataI64 []int64 // used when Dtype == "i64"
	Shape   []int64
	Dtype   string // "" or "f32" => float32; "i64" => int64
}

// F32 builds a float32 tensor.
func F32(data []float32, shape ...int64) Tensor {
	return Tensor{Data: data, Shape: shape}
}

// I64 builds an int64 tensor (for token ids / masks).
func I64(data []int64, shape ...int64) Tensor {
	return Tensor{DataI64: data, Shape: shape, Dtype: "i64"}
}

// isI64 reports whether the tensor carries int64 data.
func (t Tensor) isI64() bool { return t.Dtype == "i64" }

// NumElements returns the total element count from the shape.
func (t Tensor) NumElements() int64 {
	if len(t.Shape) == 0 {
		return 0
	}
	n := int64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Dim returns the size of dimension i (negative = from the end). 0 if out of range.
func (t Tensor) Dim(i int) int64 {
	if i < 0 {
		i += len(t.Shape)
	}
	if i < 0 || i >= len(t.Shape) {
		return 0
	}
	return t.Shape[i]
}
