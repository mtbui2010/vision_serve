package groundingdino

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"sync"
)

// nonZeroMarker is the ONNX op_type string of the node that gives the BROKEN export away.
//
// The defective community export builds the text self-attention mask with a Python loop that
// torch.onnx.export unrolled, and that loop is driven by torch.nonzero(special_tokens_mask)
// -> a NonZero node. A correct, vectorized export has no NonZero at all: it compares token
// indices pairwise and reduces (ReduceMax/ReduceMin). GroundingDINO uses NonZero NOWHERE
// else, so the presence of this op_type in the graph is a direct, causal signal that the
// mask is the baked-loop one — not a proxy or a version guess.
var nonZeroMarker = []byte("NonZero")

// exportCache memoizes SupportsJointTextPass per file path (weights are immutable once
// downloaded, and a model may be constructed repeatedly across reloads).
var exportCache sync.Map // path -> exportProbe

type exportProbe struct {
	joint bool
	err   error
}

// SupportsJointTextPass reports whether the ONNX graph at path rebuilds the block-diagonal
// text self-attention mask for an ARBITRARY number of "."-separated class phrases — i.e.
// whether the whole prompt may be scored in ONE pass.
//
// It answers by scanning the serialized graph for the NonZero op_type (see nonZeroMarker).
// The scan streams the file and stops at the first match; op_type strings live in the node
// section, which protobuf field ordering puts ahead of the weights (in the defective export
// the marker sits at byte ~26k of 719 MB). A full miss reads the whole file — measured at
// 180 ms warm for the 695 MB fixed export — and happens once per model load, on a file ONNX
// Runtime is about to read anyway.
//
// A read error is reported, and callers must then choose the SAFE (per-phrase) path: a false
// "joint" answer silently drops classes, while a false "per-phrase" answer only costs time.
func SupportsJointTextPass(path string) (bool, error) {
	if v, ok := exportCache.Load(path); ok {
		p := v.(exportProbe)
		return p.joint, p.err
	}
	joint, err := probeJointTextPass(path)
	exportCache.Store(path, exportProbe{joint, err})
	return joint, err
}

func probeJointTextPass(path string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, fmt.Errorf("grounding-dino: cannot probe export %s: %w", path, err)
	}
	defer f.Close()

	const chunk = 1 << 20
	overlap := len(nonZeroMarker) - 1
	buf := make([]byte, chunk+overlap)
	n := 0 // bytes carried over from the previous chunk
	for {
		read, err := io.ReadFull(f, buf[n:])
		n += read
		if bytes.Contains(buf[:n], nonZeroMarker) {
			return false, nil // baked-loop mask -> per-phrase passes only
		}
		if err != nil { // io.EOF / io.ErrUnexpectedEOF => last chunk
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return true, nil
			}
			return false, fmt.Errorf("grounding-dino: cannot probe export %s: %w", path, err)
		}
		// Carry the tail so a marker straddling a chunk boundary is still found.
		copy(buf, buf[n-overlap:n])
		n = overlap
	}
}
