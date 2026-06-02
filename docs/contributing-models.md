# Thêm model mới vào VisionServe

> Mục tiêu thiết kế: **thêm model = thêm package, KHÔNG sửa core** (`server`/`engine`/`lifecycle`).

## ⚖️ Ràng buộc license (BẮT BUỘC)

Chỉ chấp nhận model **license permissive**: Apache-2.0 / MIT / BSD.
**TUYỆT ĐỐI KHÔNG** thêm model AGPL (YOLO/Ultralytics, FastSAM, YOLO-World). Manifest
phải khai báo `license` đúng — registry sẽ từ chối nếu không nằm trong allowlist.

## 4 bước

### 1. Tạo package `internal/models/<name>/`

Implement interface `Model` (xem `internal/models/model.go`):

```go
package mymodel

import (
    "image"
    "visionserve/internal/engine"
    "visionserve/internal/models"
)

func init() { models.Register("my-arch", New) }

type myModel struct{ cfg models.Config }

func New(cfg models.Config) (models.Model, error) { return &myModel{cfg: cfg}, nil }

func (m *myModel) Name() string          { return m.cfg.Name }
func (m *myModel) Task() models.Task     { return models.TaskDetection }
func (m *myModel) InputName() string     { return "" } // "" = engine tự dò từ ONNX
func (m *myModel) OutputNames() []string { return nil }

func (m *myModel) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
    // resize/letterbox/normalize → tensor; lưu scale/pad vào meta
}
func (m *myModel) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
    // decode output → Result; BBox PHẢI map về toạ độ ảnh GỐC qua meta
}
```

### 2. Đăng ký import blank trong `cmd/visionserve/main.go`

```go
import _ "visionserve/internal/models/mymodel"
```

(Đây là chỗ DUY NHẤT trong core cần đụng tới — chỉ một dòng import.)

### 3. Tạo thư mục registry `models/<name>/`

- `manifest.yaml` (xem [manifest-spec.md](manifest-spec.md)) — `architecture` khớp tên đã `Register`.
- `README.md` hướng dẫn tải/export weights ONNX (**không commit** file lớn).
- (tuỳ chọn) file labels.

### 4. Viết test cho pre/postprocess

Đây là phần dễ sai nhất (CLAUDE.md). Test tối thiểu:
- letterbox/normalize ra đúng shape + giá trị mẫu.
- postprocess map box về toạ độ ảnh gốc đúng (kiểm tra trường hợp có padding).

## Lưu ý quan trọng

- **KHÔNG đoán format output ONNX.** Xác minh shape thật trước khi viết postprocess.
  Chưa rõ → viết stub + `TODO`, đừng bịa.
- RF-DETR là **NMS-free** — đừng áp NMS của YOLO. Model anchor-based mới có thể dùng
  `imageproc.NMS`.
- `Infer` (gọi session) KHÔNG thuộc model — engine+lifecycle lo. Model chỉ pre/post.
