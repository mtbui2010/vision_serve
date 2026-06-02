# Đặc tả Manifest ("Modelfile cho CV")

Mỗi model là một thư mục con trong registry (`./models/<name>/`) chứa `manifest.yaml`
+ file ONNX (không commit) + (tuỳ chọn) file labels.

## Ví dụ đầy đủ

```yaml
name: rf-detr                 # bắt buộc — định danh model trong registry
task: detection               # bắt buộc — detection | segmentation | open_vocab
license: Apache-2.0           # bắt buộc — CHỈ permissive (Apache-2.0/MIT/BSD); cấm AGPL
architecture: rf-detr         # tuỳ chọn — loại factory (mặc định = name)
model_file: rf-detr-base.onnx # bắt buộc — đường dẫn tương đối tới .onnx

input:
  width: 560                  # bắt buộc > 0
  height: 560                 # bắt buộc > 0
  layout: NCHW                # NCHW | NHWC
  letterbox: true             # giữ tỉ lệ + pad
  normalize:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

postprocess:
  type: detr                  # gợi ý decode (detr/sam/...)
  box_format: cxcywh          # cxcywh | xyxy (chuẩn hóa [0,1])
  conf_threshold: 0.5
  max_detections: 300

labels: coco.txt              # tuỳ chọn — mỗi dòng một class

runtime:
  prefer: [tensorrt, cuda, cpu]   # fallback chain EP (CPU luôn được thêm ở cuối)
  idle_unload_seconds: 300        # 0 = không bao giờ tự unload
```

## Quy tắc validate (registry từ chối nếu vi phạm)

| Trường | Ràng buộc |
|--------|-----------|
| `license` | phải ∈ {Apache-2.0, MIT, BSD-3-Clause, BSD-2-Clause}. **AGPL bị từ chối.** |
| `task` | ∈ {detection, segmentation, open_vocab} |
| `model_file` | bắt buộc khai báo (đường dẫn) |
| `input.width/height` | > 0 |
| `input.layout` | NCHW / NHWC (hoặc bỏ trống) |
| `runtime.prefer` | mỗi EP ∈ {tensorrt, cuda, cpu} |

Manifest không hợp lệ về **cấu trúc** bị **bỏ qua** lúc scan (gom vào cảnh báo), không làm sập server.

### Weights tồn tại ≠ validate

Sự **tồn tại của file `.onnx`** KHÔNG phải điều kiện validate cấu trúc. Model có manifest
hợp lệ nhưng chưa tải weights vẫn được **liệt kê** với trạng thái:

| Trạng thái | Ý nghĩa |
|-----------|---------|
| `not_downloaded` (`list`: `missing`) | manifest hợp lệ, chưa có file `.onnx` |
| `available` (`list`: `ready`) | weights đã có, sẵn sàng load |
| `loaded` | đang trong bộ nhớ |

Việc thiếu weights chỉ báo lỗi rõ ràng khi **load/predict** (giống Ollama: thấy model
trước khi `pull`).
