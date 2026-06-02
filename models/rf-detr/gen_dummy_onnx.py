#!/usr/bin/env python3
"""Sinh file ONNX DUMMY cho RF-DETR — CHỈ là fixture test pipeline, KHÔNG phải runtime.

VisionServe là binary Go; script Python này là CÔNG CỤ DEV để tái tạo file
`rf-detr-base.onnx` nhỏ (~vài KB) dùng test luồng preprocess→ORT→postprocess→map-ngược
mà không cần tải weights RF-DETR thật (~108MB). KHÔNG kéo Python vào runtime (CLAUDE.md).

Dummy mô phỏng ĐÚNG hợp đồng I/O của RF-DETR thật (đã xác minh trên weights thật,
xem README.md):
    input  : "input"  [1, 3, 560, 560] f32
    outputs: "logits" [1, Q, 91] f32   (91 = không gian class COCO "paper", idx 0 = N/A)
             "boxes"  [1, Q, 4]  f32   (cxcywh, chuẩn hóa [0,1] theo ảnh đầu vào)

Output là HẰNG SỐ (bỏ qua input) — chỉ để kiểm tra luồng, không phải suy luận thật.
Fire sẵn 2 detection để `visionserve run rf-detr` ra kết quả có nghĩa:
    query 0 -> class 3  ("car"), conf ~1.00, box (0.5, 0.5, 0.2, 0.3)
    query 1 -> class 17 ("cat"), conf ~0.82, box (0.25,0.25,0.1, 0.1)

Chạy: python3 gen_dummy_onnx.py   (cần `onnx`, chỉ lúc dev). Ghi đè rf-detr-base.onnx.
"""
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

Q, C = 5, 91  # 5 queries, 91 class (khớp pred_logits của RF-DETR thật)

# logits: nền rất âm (sigmoid ~ 0), trừ 2 query fire sẵn.
logits = np.full((1, Q, C), -10.0, dtype=np.float32)
logits[0, 0, 3] = 12.0   # car  -> sigmoid ~ 1.000
logits[0, 1, 17] = 1.5   # cat  -> sigmoid ~ 0.818

# boxes: cxcywh chuẩn hóa [0,1]; chỉ 2 query đầu có giá trị.
boxes = np.zeros((1, Q, 4), dtype=np.float32)
boxes[0, 0] = (0.5, 0.5, 0.2, 0.3)
boxes[0, 1] = (0.25, 0.25, 0.1, 0.1)

logits_const = helper.make_node(
    "Constant", [], ["logits"],
    value=numpy_helper.from_array(logits, name="logits_val"),
)
boxes_const = helper.make_node(
    "Constant", [], ["boxes"],
    value=numpy_helper.from_array(boxes, name="boxes_val"),
)

# input khai báo nhưng KHÔNG dùng (ONNX cho phép) — giữ đúng chữ ký để engine dò I/O.
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 560, 560])
out_logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, Q, C])
out_boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, Q, 4])

graph = helper.make_graph(
    [logits_const, boxes_const], "rf_detr_dummy",
    [inp], [out_logits, out_boxes],
)
model = helper.make_model(graph, producer_name="visionserve-dummy",
                          opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9  # tương thích onnxruntime cũ hơn
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(__file__), "rf-detr-base.onnx")
onnx.save(model, out_path)
print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes) — outputs logits[1,{Q},{C}] + boxes[1,{Q},4]")
