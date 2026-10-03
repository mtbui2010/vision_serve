# Đề xuất refactor VisionServe (2026-10)

Mục tiêu: các module **nhỏ, một trách nhiệm, dễ hiểu, dễ thay thế**, và **tiết kiệm bộ nhớ, nhanh
hơn**, nhưng không đổi API HTTP và không đổi kết quả đã đo (mAP trong các manifest).

Đề xuất này dựa trên đợt rà soát toàn bộ project (4 phần: lõi runtime, các model vision,
open-vocab/grasp, catalog + SDK Python). Các lỗi tìm được đã được sửa riêng; tài liệu này chỉ
bàn về **cấu trúc**, tức nguyên nhân khiến các lỗi đó lặp lại.

---

## 1. Vì sao cần refactor: các lỗi đều có chung một gốc

| Hiện tượng (lỗi thật đã gặp) | Gốc về cấu trúc |
|---|---|
| SCRFD trả 0 khuôn mặt; EfficientSAM/SAM2/NanoSAM trả mask sai kích thước; PaddleOCR box sai | Mỗi model tự viết lại decode, mask, RLE, NMS (RLE có **5 bản sao**, `threshold→bitmap` có 6), và test dùng shape giả thay vì shape thật |
| Letterbox thay vì squash (−7.35 mAP); tokenizer pad sai (−4.3 mAP); CLIP crop sai (cosine 0.88) | Tiền xử lý rải rác trong từng package, không có một chỗ mô tả "model này được train với tiền xử lý gì" |
| Gỡ model khi đang có request dùng → panic / treo | Vòng đời session không có khái niệm "đang được dùng" (đã sửa bằng lease) |
| `PipelineMu` toàn cục làm router mặc định chỉ xử lý 1 request mỗi lần | Chính sách tuần tự hóa nằm trong một package model thay vì ở runtime |
| Mỗi option request mới phải sửa ở 4 chỗ (JSON, multipart, CLI, `Prompt`) | `models.Prompt` là một túi chứa mọi tham số của mọi model |
| Tải song song cùng model làm hỏng file; `pull --force` từ symlink xóa trắng weights | Tải và cài đặt không theo kiểu giao dịch (transaction) |
| Allowlist license nằm cả ở Go lẫn Python, có thể lệch nhau | Quy tắc trùng lặp giữa hai ngôn ngữ, không có test đồng bộ |

---

## 2. Kiến trúc đề xuất

```
cmd/visionserve            entrypoint (chỉ wiring)
internal/
  httpapi/                 HTTP: routes, decode request → Request, encode Result, map lỗi→status
  runtime/                 (lifecycle hiện tại) load/lease/unload, reaper, singleflight, admission
  engine/                  ONNX Runtime: Session, Pool, EP chain — không biết gì về model
  registry/  catalog/      manifest + tải/cài đặt (transactional)
  vision/                  THƯ VIỆN DÙNG CHUNG cho mọi model (thuần Go, có benchmark):
    geom/                    Box, Meta (scale/pad/crop), Map/Clamp, IoU
    preprocess/              Spec (squash|letterbox|center_crop|keep_aspect) → tensor, một implementation duy nhất
    nms/                     NMS O(n log n) + top-k, class-aware/agnostic
    mask/                    Bitmap, Threshold, RLE (một encoder), Upsample, BBox
    text/                    tokenizer BERT/BPE/Unigram + test vector chung
  models/                  mỗi model = một package NHỎ, chỉ chứa phần riêng của nó:
    detr/                    RF-DETR + RT-DETR (một decoder, hai đăng ký)
    sam/                     MobileSAM/EfficientSAM/SAM2/NanoSAM = encoder/decoder adapter + vision/mask
    gdino/                   GroundingDINO: tokenize → pass (chunk ≤256 token) → spans
    ...
  pipeline/                các model ghép: router, grounded-sam, grasp — chỉ ghép các "stage"
```

Nguyên tắc chính:

1. **Một model không tự viết lại hình học hay mask.** Package model chỉ giữ phần thật sự riêng
   (tên input/output, cách đọc output). Mọi thứ khác lấy từ `vision/*`, nơi có test vector đối
   chiếu với implementation tham chiếu (PIL, transformers, InsightFace).
2. **Tiền xử lý là dữ liệu, không phải code.** Mỗi manifest khai báo `preprocess: {resize:
   squash|letterbox|center_crop|keep_aspect, size, multiple_of, mean, std, resample}`.
   `vision/preprocess` là implementation duy nhất. Endpoint `/api/preprocess` và converter đọc
   **cùng một spec**, nên tầng kiểm tra B1 so sánh được trực tiếp.
3. **Pipeline là ghép các stage có interface nhỏ:**
   `Detector(img, words) → []Det`, `Rescorer(img, dets, words) → []Det`,
   `Segmenter(img, boxes) → []Mask`, `GraspPlanner(mask, depth) → []Grasp`.
   Router, Grounded-SAM, grasp-gd, gdino-siglip trở thành **các cấu hình** của những stage này,
   thay vì 4 file `Infer` dài viết tay. Thay một thành phần (ví dụ SigLIP → CLIP) chỉ là đổi
   một stage.
4. **Runtime sở hữu mọi chính sách đồng thời.** Lease theo session (đã làm), singleflight lúc
   load (đã làm), giới hạn số request chờ cho mỗi model (admission control), và tuần tự hóa theo
   từng session. Bỏ `groundingdino.PipelineMu` toàn cục.
5. **Option theo model có kiểu rõ ràng.** `Request{Model, Image, Text, Boxes, Points, Options
   map[string]any}`. Mỗi model khai báo và kiểm tra `Options` của mình (một struct có tag), nên
   server không cần biết `gripper_min` hay `crop_temp` là gì.

---

## 3. Chi tiết từng module

### 3.1 `engine` (đã gọn, chỉ cần chuẩn hóa)
- `Session` và `Pool` dùng chung hợp đồng `ErrClosed` (đã làm).
- `recover` trong job wrapper của worker: một panic trên thread đã pin hiện giết cả process.
- Thay việc đổi fd 2 (stderr) lúc tạo session bằng logging callback của ORT. Hiện việc đổi fd 2
  nuốt mất log của các goroutine khác trong suốt thời gian build TensorRT, và khóa toàn cục
  `stderrMu` khiến `--preload` song song thực chất chạy tuần tự.

### 3.2 `runtime` (lifecycle)
- Tách `manager.go` (khoảng 700 dòng) thành `load.go` (build + singleflight + một helper
  `newRunnable(path, n, providers)`, thay cho 3 bản sao "tạo N session, đóng hết nếu lỗi"),
  `lease.go`, `reaper.go`, `explain.go`.
- Lưu snapshot manifest vào Session lúc load. Explain hiện đọc lại registry, nên có thể lệch nếu
  manifest trên đĩa đã đổi.
- Cache sha256 của weights theo (size, mtime, inode). Hiện **mỗi lần load lại sau khi bị reaper
  gỡ, và mỗi lần gọi `/api/preprocess`, đều hash lại 695 MB (3–4 s)**.

### 3.3 `httpapi` (server)
- Một hàm `decodeRequest(r) → Request` cho cả JSON lẫn multipart (thay `parsePredictRequest`
  trả về 6 giá trị). Một `decodeImage` có giới hạn byte, pixel và EXIF (đã làm). Một `writeError`
  ánh xạ lỗi có kiểu sang status (`ErrNotFound`→404, `ErrBadRequest`→400, …); hiện mọi lỗi đều
  là 500.
- Encode tùy chọn base64/binary cho `depth_map`/`embeddings`. Đo trên mảng 1920×1080: JSON số
  334 ms, 21 MB; base64 56 ms, 10 MB.
- Mặc định chỉ listen `127.0.0.1` (như Ollama). Docker vẫn truyền `--addr :11435` một cách tường minh.

### 3.4 `vision/*` (thư viện dùng chung, chỗ nhận lại nhiều nhất)
| Hàm hiện có | Số bản sao | Gom về |
|---|---:|---|
| RLE column-major | 5 (hai thứ tự tham số khác nhau!) | `mask.EncodeRLE` |
| threshold → bitmap + bbox | 6 | `mask.Threshold` |
| upsample low-res logits | 0 (thiếu, nên SAM2/NanoSAM sai) | `mask.Upsample` |
| sigmoid, clamp box, `toInputXYWH` | 4 / 3 / 2 | `geom` |
| NMS | 2 (một bản O(n²): 3 s cho 16.8k box) | `nms` |
| vòng lặp HWC→CHW + normalize viết tay | 6 | `preprocess` |
| `firstName`, `shapesOf`, `max` | ≥5 | `vision/util` |

### 3.5 Open-vocabulary
- **Một "crop namer" duy nhất.** `hybrid.rescore` và `textalign.nameOpenCrops` đang lặp lại
  chuỗi crop → embed → cosine → softmax → floor, và `cropTemp` đã lệch giữa hai nơi
  (0.05 so với 0.02).
- **Một cache embedding theo từng từ** (LRU có giới hạn, key = templates + từ). Hiện có 3 cache
  FIFO, mỗi cache key theo **cả danh sách từ**, nên một tổ hợp mới phải embed lại mọi từ
  (khoảng 4.3 ms mỗi từ).
- **Một `rfdetr.SplitOutputs(outs, nLabels, dFeat)`** thay cho 3 heuristic khác nhau dùng để nhận
  diện output logits/boxes/features.
- **Cấu hình GroundingDINO đặt ở một chỗ:** ngưỡng, vocab, chia chunk prompt.

### 3.6 Catalog / cài đặt
- **Một đường tải duy nhất:** ghi file tạm, hash trong lúc tải, kiểm tra Content-Length, phát
  hiện trang lỗi HTML, verify, fsync, rồi rename, dưới lock theo từng model, và resume bằng Range.
- **Cài đặt kiểu giao dịch, giống nhau ở Go và Python:** dựng thư mục tạm, kiểm tra SameFile,
  backup bản cũ rồi đổi chỗ, phục hồi nếu lỗi.
- **Sinh manifest bằng cách marshal `registry.Manifest`,** thay vì ghép chuỗi YAML bằng tay
  (kiểu này dễ đặt sai field vào nhầm block).

### 3.7 SDK Python / converter
- **Gom quy tắc dùng chung:**
  - **License:** dùng một allowlist, có test đồng bộ với bản Go (hoặc lấy từ lệnh `visionserve
    licenses`).
  - **Kiểm tra tên model:** dùng một regex duy nhất cho cả Go lẫn Python.
  - **Quét license:** một hàm `license_scan(source, module)` cho mọi định dạng.
- **SDK:** tách depth tương đối (của model) khỏi depth metric (từ sensor); bỏ các bản sao xử lý
  ndarray/grasp; thêm `Result.to_json()`.

---

## 4. Bộ nhớ và tốc độ: các điểm nóng đã đo

| Điểm nóng | Đo được | Đề xuất | Lợi ích dự kiến |
|---|---|---|---|
| MobileSAM automask (không prompt) | 1.0 GB / 17 s ở 640×480; 4.8 GB / 34 s ở 3200×2400 | lọc và NMS ở độ phân giải thấp, lọc trước bằng IoU của bbox, chỉ upsample mask được giữ | >10× bộ nhớ, NMS nhanh hơn nhiều |
| `imageproc.NMS` | 3.05 s cho 16.8k box | chỉ duyệt các phần tử sau khi đã sắp xếp + top-k | khoảng O(n log n) |
| Hash sha256 mỗi lần load / preprocess | 3–4 s cho 695 MB | cache theo (size, mtime, inode) | khoảng 0 s |
| Grasp search | 120–140 ms, 28 MB mỗi mask lớn | kiểm tra độ mở tay kẹp trước, dùng top-K heap | 2–4× |
| Background `method=sam` | encoder chạy 6 lần mỗi ảnh | encode 1 lần, decode 6 lần | bớt 5 lượt encoder |
| textalign mode `exact` | 21.8 ms Go thuần | đưa sang ORT như fast-path (BUGS_TO_FIX #4) | khoảng 1–3 ms |
| Mask/depth trả về dạng JSON số | 334 ms cho 2 MP | base64 tùy chọn | khoảng 6× |
| Allocation theo từng pixel (`At()`/`Set()`) | nhiều chỗ | đọc thẳng `Pix` theo stride | 5–20× ở các vòng lặp đó |
| Không có admission control | bộ nhớ tăng theo số request đang chờ | semaphore theo model, đặt **trước** bước decode | chặn OOM khi tải cao |

---

## 5. Lộ trình (mỗi bước là một PR nhỏ, có test, không đổi API)

1. **`vision/mask`, `vision/geom`, `vision/nms`:** chuyển các bản sao sang dùng thư viện chung,
   kèm test vector lấy từ implementation tham chiếu. Rủi ro thấp; xóa được khoảng 800 dòng.
2. **`vision/preprocess` + spec trong manifest:** chuyển `letterbox`/`crop`/`keep_aspect` hiện
   có sang `preprocess:` (giữ alias cũ). Chạy lại tầng B của converter trên mọi model để chứng
   minh tensor không đổi.
3. **`runtime`:** tách file, cache sha256, admission control, chuyển chính sách tuần tự hóa vào
   runtime (bỏ `PipelineMu`).
4. **`httpapi`:** gom `Request`, map lỗi→status, encode base64 tùy chọn.
5. **`pipeline/`:** viết lại router, grounded-sam, grasp-gd, gdino-siglip thành cấu hình các
   stage. Kiểm tra bằng protocol held-out: con số trong manifest không được thay đổi.
6. **Catalog transactional + đồng bộ quy tắc Go/Python.**

Mỗi bước đều phải có phép đo trước/sau: test tương đương, tầng B của converter, hoặc protocol
held-out. Đây chính là bài học của BUGS_TO_FIX: tối ưu mà không có test tương đương thì thực
chất là viết lại.
