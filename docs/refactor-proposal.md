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

---

## 6. Trạng thái thực hiện (nhánh `refactor/2026-10`, 2026-10-03)

Cả 6 bước của lộ trình đã xong trên nhánh `refactor/2026-10` (chưa push). Việc được chia thành 7
luồng song song (A–G); mỗi luồng chỉ được merge sau khi qua cổng tương đương ở mục cuối.

| Bước | Luồng | Commit chính |
|---|---|---|
| 1. `vision/mask`, `vision/geom`, `vision/nms` | A | e8a6824, c634c59, 973ef1c, f96a0b4, 1d01418; 4bdd0b9 gộp rf-detr/rt-detr thành một package `detr` |
| 2. `vision/preprocess` + block `preprocess:` | F | 73b91d0, bb5c6d5, 37567c9, 7a1d410, 8bd148c — các trường `input.*` cũ vẫn là alias |
| 3. `runtime` (lifecycle, engine) | B, E | 93cfbf4 (tách file), 28c7d4e (cache sha256), 88b0a09 + e111928 (admission), 92434b9 + ecb7a63 (bỏ `PipelineMu`, khóa theo từng model), 46fc739 (đọc I/O ONNX từ header protobuf) |
| 4. `httpapi` (server) | C | 6fd30af (một request model, lỗi có kiểu → status, hủy theo ctx), 517fc45 (base64) |
| 5. `pipeline/` | G | 76e9ca2, 68f3789 (cache embedding theo từ, một CropNamer), e9f5558 (router / grounded-sam / grasp thành cấu hình stage), 01edc6c |
| 6. Catalog + đồng bộ Go/Python | D | be43477, 8018da5 (pull tiếp tục được), c293758 (dọn staging), 420d9a2 |

### Điểm nóng của mục 4

| Điểm nóng | Trạng thái |
|---|---|
| Hash sha256 mỗi lần load | **Xong**: cache theo (size, mtime, ctime, dev, inode) |
| `imageproc.NMS` | **Xong**: `vision/nms` dùng lưới |
| Mask/depth dạng JSON số | **Xong**: `encoding=base64` phía server; trong SDK Python là opt-in (`base64_arrays=True`) |
| Allocation theo từng pixel | **Xong** trong `vision/preprocess` (đọc `Pix` theo stride) |
| Admission control | **Xong**: `max(32, 2×slots)` request mỗi model, `VISIONSERVE_MAX_QUEUE`, 503 + `Retry-After` |
| MobileSAM automask (không prompt) | **Xong**. Lọc + NMS ở khung 256 px, lọc trước bằng bbox, chỉ decode lại full-res cho mask được giữ: đã có từ efcf9de (3200×2400: 58 s / +3.7 GB → 19 s / +1.5 GB; 640×480 không đổi). Phần thời gian còn lại là do luồng ORT: 4 decoder trong pool, mỗi cái 24 luồng spin, khoảng 600 CPU-giây mỗi request. 5bea01e giới hạn mỗi session trong pool còn `NumCPU/(4n)` luồng (`VISIONSERVE_POOL_THREADS` để chỉnh). bbcf880 encode RLE ngay khi mỗi mask xong. Đo lại: 640×480: 14–15 s → **3.8 s**, peak RSS ~1.0 GB (0.63 GB là model đã load); 3200×2400: 17–18 s → **4.7–5.0 s**, peak 2.3 GB → **1.6–1.8 GB**. Bộ nhớ (2026-10-04, 07053e8, cb2b4f3, 9d7d5a1): đo lại thì phần tăng lớn nhất không nằm ở mask full-res mà ở ORT: arena CPU cộng memory pattern làm từ request thứ hai trở đi arena giữ hai bản vùng nhớ trung gian (cũng gặp ở grounding-dino 2.24 → 3.47 GB, rf-detr 0.32 → 0.48 GB). Session CPU giờ tắt memory pattern (arena vẫn bật; tắt arena thì tốn bộ nhớ hơn và chậm hơn 15–20 %). Lượt full-res: mỗi worker (4, một cho mỗi decoder) có một buffer logit nằm ngoài heap Go, decoder ghi thẳng vào (`RunInto`), xong lượt thì trả về OS; bitmap cũng dùng lại theo worker. Grasp và background automask xử lý từng mask ngay khi xong (`InferMasksEach`) thay vì giữ ~30 bitmap 7,7 MB. Peak RSS sau 5 request (server mới mỗi lần): automask 640×480 0.99 → **0.66 GB**, 3200×2400 1.63 → **0.80 GB**; grasp 1.00 → **0.66** / 2.06 → **0.87 GB**; background `method=automask` 1.18 → **0.82** / 1.98 → **0.98 GB** (model đã load: 0.36 GB, background 0.54 GB vì có MiDaS). Thời gian không đổi trong sai số (automask 3200×2400: 4.9–5.2 s trước, 4.7–5.1 s sau; grasp 3200×2400 6.0–6.2 → 5.1–6.1 s; background 3200×2400 3.8–4.5 → 2.7–3.9 s) |
| Grasp search | **Xong**. Top-K + bỏ allocation từng candidate đã có từ efcf9de. bcf2571 bỏ bảng summed-area (8 byte mỗi pixel bbox) khi quét biên và kiểm tra độ mở tay kẹp trước force closure. Một mask lớn (ảnh 3200×2400): 105–170 ms / 41 MB → **60–100 ms / 7.7 MB**; 58 mask thật của một ảnh 3200×2400: 1.0–1.3 s / 256 MB → 0.8–1.0 s / 109 MB; mask ở ảnh 640 px tốn ≤ 30 ms. Cả request vẫn do SAM chi phối: `grasp-rfdetr` 3.7–5.8 s → 1.5–1.8 s nhờ 5bea01e |
| Background `method=sam` | **Xong** từ efcf9de (`encoderOnce`). Đếm trên server thật: encoder chạy 6 → **1** lần mỗi request, decoder vẫn 6 lần. Thời gian: 14 s (trước efcf9de) → 2.8–3.9 s → **0.63–0.74 s** với 5bea01e |
| textalign `exact` | **Xong, đã bật tại máy**: role `files.head: head.onnx` (sinh bằng `models/rfdetr-textalign-dec1-siglip/export_head_onnx.py`), Go là fallback khi không khai báo. ORT CPU so với Go: lệch tối đa 1.43e-6 logit. Đo cả request (median, 2026-10-04): CPU 134 ms (Go) → 121 ms với `runtime.threads: {head: 1}` (354 ms nếu để thread mặc định của ORT); GPU 73 → 41 ms. **Đã bật** (`files.head` + `runtime.threads: {head: 1}`) trong 7 manifest dùng phương pháp `exact` có weights tại máy: `rfdetr-textalign-etri`, `-etri-probe`, `-dec1`, `-dec1-probe`, `-dec1-siglip`, `-dec1-siglip-probe`, `-dec1-siglip-prod`. `head.onnx` được sinh tại máy vào `models/<dir>/` (thư mục `-probe` symlink sang `head.onnx` của thư mục gốc, giống `proj.bin`), không commit; bản cho `pull` nằm trên repo HF `mtbui2010/rfdetr-textalign-ONNX` (xem mục 6). Không bật cho `rfdetr-dualhead-dec1` (dùng `method: dual`, head ONNX không chạy trên đường đó) và các hybrid `rfdetr-gdino-*` (không dùng head của textalign; fast-path đã có `head.onnx` riêng). Kiểm tra qua server CPU, ngưỡng 0.001, 2 ảnh × 2 prompt × 7 model: lệch conf tối đa 7.4e-7, cùng tập (class, box); chỉ có một cặp kề nhau đổi chỗ vì hai conf gần bằng nhau (0.04532, chênh 1e-7). GPU (CUDA EP) so với Go trên GPU: lệch ≤ 2.7e-4 |

Các số trên đo trên CPU (`CUDA_VISIBLE_DEVICES=`), chạy qua server, peak RSS lấy từ `VmHWM`. Máy dùng chung 48 luồng và đang tải nặng (load 25–95), nên thời gian ghi dạng khoảng. Output trùng từng byte với bản trước, và golden 43 case trùng từng bit. Phần bộ nhớ automask còn lại so với lúc mới load (~0.3 GB ở 640×480, ~0.45 GB ở 3200×2400) chủ yếu là arena của encoder (activation TinyViT ở 1024 px, ~0.29 GB, giống nhau ở mọi kích thước ảnh) và 4 buffer logit 30 MB trong lượt full-res. Tắt arena riêng cho encoder không giảm được peak.

### Rà lỗi sau refactor

Có 4 agent review độc lập: lõi runtime; preprocess + model thường; pipeline open-vocab;
catalog/SDK/docs. Thêm vào đó là một đợt fuzz Go/Python cho `preprocess.Spec` và một bài so
sánh trực tiếp bản cũ (efcf9de) với bản mới trên weights thật. Lỗi tìm được và đã sửa:

- **Lifecycle**: một Load đến sau khi Unload đã trả lời lại nhập vào lần load đã hủy, nên trả
  500 (e69fda9). `/api/preprocess` trả 500 thay vì 400 khi model từ chối prompt (fd0ea04).
  `/api/preprocess` chưa resolve `template_name` (399dfa4).
- **Engine**: parser header ONNX panic với file hỏng (465c75f).
- **HTTP**: lỗi prompt trên `/api/predict` (thiếu text, `" . "`, phrase quá dài, thiếu
  template) trả 500. Nay `models.BadPrompt` khiến chúng trả 400 (dc0e9d1).
- **Templates**: một request upload có thể decode hàng trăm GB trước khi chạm giới hạn của
  store. Nay giới hạn được kiểm tra từ header ảnh (1349d1a).
- **Preprocess**:
  - `pad: .nan` lọt qua kiểm tra; NaN/Inf giờ bị từ chối ở cả Go lẫn Python (e1ea01d).
  - Python đọc bool YAML, key null và list normalize lạ khác Go (e1ea01d).
  - SAM/PaddleOCR âm thầm bỏ qua block `preprocess:`; nay đó là lỗi khi load (adf4d83).
- **NMS**: panic khi có box không hữu hạn và từ 512 ứng viên trở lên (aff9680).
- **Pipeline**: key của cache embedding giữ nguyên chuỗi từ, nên prompt dài làm RAM tăng không
  giới hạn. Nay key là SHA-256 (5fe743b).
- **Catalog**:
  - Có thể xóa bản cài cũ khi đó là bản duy nhất (dcc32ec).
  - `pull` một tên sai vẫn dọn registry (71c715b).
  - Model cài từ thư mục có quyền 0700 (d78ae2b).
- **SDK / client**:
  - Python mặc định trả `FloatArray` thay vì `list`; nay base64 là opt-in (6519a3d).
  - Client JS mặc định `localhost` (IPv6 trên Node 18) và `tsc` build lỗi (2331191).

### Cổng tương đương (chạy lại sau commit cuối)

- **Golden:** 43 case trên CPU, trùng bản gốc từng bit.
- **Held-out protocol (GPU):** 89.75/49.54, 89.75/62.47, 89.75/48.11, 57.41/61.07, đúng như
  trước, cùng số detection.
- **So sánh trực tiếp bản cũ/bản mới:** khoảng 730 request (model thường + SAM, 22 ảnh có kích
  thước và định dạng lạ) và 40 request open-vocab, trùng từng byte.
- **Test:** `go test -race` (39 package), staticcheck, test Python, test trong image converter,
  và `visionserve convert` e2e (PASS).
- **Image Docker CPU:** smoke test đạt (health, predict rf-detr, prompt sai → 400, model lạ → 404).

### Còn mở

- **Head ONNX trên CPU.** **Đã sửa (701b336).** Head chỉ mất khoảng 1 ms, nhưng một session ORT
  CPU với thread pool mặc định (spinning) làm detector chậm đi khoảng 3 lần khi hai session chạy
  xen kẽ. Manifest giờ có `runtime.threads` (role → số intra-op thread, ghi đè mặc định của
  lifecycle cho riêng role đó). Đo trên CPU (median cả request, các vòng xen kẽ):
  - textalign `exact`: Go 134 ms, ORT mặc định 354 ms, ORT 1 thread 121 ms.
  - fast-path (`rfdetr-gdino-fastpath`, head bật sẵn): 2.30 s → 1.83 s với `head: 1`. Manifest
    này giờ đặt `threads: {head: 1}`. Kết quả không đổi: golden trùng từng bit, held-out trên CPU
    89.70/47.76 (n=6526) ở cả hai cấu hình.
  Binding v1.13 không có `allow_spinning`, nên 1 thread là cách duy nhất hiện có để tắt spinning.
- **Mask MobileSAM trên GPU** lệch 1–43 pixel biên giữa các lần chạy. **Đã sửa (20d939b).**
  - Không phải race trong Go. Với cùng một embedding, 4 session decoder trong pool cho kết quả
    trùng từng bit, dù chạy tuần tự hay song song.
  - Nguồn lỗi là encoder. Dump mọi tensor trung gian thì tensor lệch đầu tiên luôn là
    `neck.3/ReduceMean_1`, tức phép lấy trung bình theo kênh trong LayerNorm2d cuối. Kernel
    reduction của CUDA EP cộng theo thứ tự phụ thuộc vào việc GPU đang chạy gì khác. Embedding
    lệch ở bit cuối của vài trăm giá trị, rồi ngưỡng logit>0 làm lật vài pixel biên.
  - Lỗi cần có việc khác chạy cùng trên GPU (request khác, session khác, process khác). GPU rảnh:
    0/2000 lần lệch. Có thêm một session encoder chạy song song: 26/1500.
  - `cudnn_conv_algo_search=DEFAULT` không giúp: 78/2000 lần lệch và chậm khoảng 4 lần.
  - Cách sửa: `SetDeterministicCompute` cho mọi session không phải CPU, **opt-in** bằng
    `VISIONSERVE_DETERMINISTIC=1` cho cả process. Mặc định tắt, vì các con số mAP ghi trong
    manifest được đo khi chưa bật (xem dưới). Session CPU giữ nguyên.
  - Binding không bọc hàm này (kể cả bản mới nhất v1.36.0), và ORT cũng không có config key cho
    nó. Vì vậy `internal/engine/deterministic_cgo.go` gọi thẳng C API. Đây là đoạn cgo đầu tiên
    trong code của mình. Trên Windows chưa nối.
  - Kết quả trên GPU 3 (RTX A6000, ORT 1.26). Encoder khi có tải: 0/1500 lần lệch. Server, cold
    và song song 8: 0/2176 request lệch, trước đó khoảng 1/400. Độ trễ và throughput không đổi
    trong mức nhiễu: mobile-sam ~61 ms, rf-detr ~23 ms, grounding-dino 122–127 ms, mobile-sam
    song song 8 khoảng 37 req/s ở cả hai chế độ.
  - Kernel tất định làm tròn khác, nên output GPU đổi một lần. Mask mobile-sam lệch 1–16 pixel ở
    26/32 case so với trước, score GroundingDINO lệch tối đa ~0.002. Held-out protocol (GPU):
    49.54→49.59, 62.47→62.43, 48.11 giữ nguyên, gdino-siglip 57.41/61.07→57.19/60.93, đổi theo
    cả hai chiều. Khi tắt (mặc định), bản mới cho đúng số cũ. Khi bật thì snapshot GPU cũ phải chụp lại; CPU
    không đổi.
  - Test opt-in `TestEncoderBitwiseRepeatableOnGPU` tái hiện lỗi khi tắt và chứng minh đã sửa.
- ~~**`Admit`** chưa nhận ctx, và multipart vẫn được parse trước `Admit`.~~ Đã xong:
  `Admit(ctx, name)` từ chối request mà client đã bỏ đi (499). Multipart nay được đọc từng
  part: ngay trước khi đọc part file đầu tiên (`image`/`depth`), server *thử* admit (lấy slot rồi
  trả lại ngay), nên một request bị từ chối (503) không đọc cũng không lưu upload của nó, và trả
  về ngay. Slot thật chỉ được lấy sau khi đọc hết body: bản đầu giữ slot suốt lúc upload, nên
  khoảng 32 client upload chậm đủ làm mọi request khác của model đó nhận 503 tới 2 phút (review
  phát hiện, đã sửa). Khi từ chối giữa chừng, server gửi `Connection: close` nếu phần body chưa
  đọc có thể lớn. Nếu ảnh đến trước field `model`, ảnh
  được giữ với đúng giới hạn cũ (32 MiB trong RAM, phần còn lại ra file tạm) và request được admit
  sau khi đọc hết form. JSON vẫn đọc trọn trước `Admit` (model nằm trong JSON, body đã bị chặn ở
  32 MiB). Hai điểm còn lại (503 thay vì 400 khi quá tải; `PredictPrompt` chờ slot không theo
  ctx) đã xử lý ở mục dưới.
- **`top_left_pad`** (SCRFD): **đã sửa**. Trước đây Meta dùng một `det_scale = new_h / h` cho cả
  hai trục, giống InsightFace gốc. Vì `new_w` và `new_h` bị cắt phần lẻ riêng, trục x lệch tới
  1–2 pixel input ở mép xa với ảnh thường, và lệch hẳn với panorama cực đoan (10000×10 → 640×1:
  x bị chia cho 0,1 thay vì 0,064). Nay Meta ghi đúng tỉ lệ của từng trục (`new_w / w`,
  `new_h / h`), cả Go lẫn Python (`spec.py`, `geometry_sync.json`). Tensor và trục y không đổi.
  Golden: 43 case vẫn trùng từng byte (hai ảnh SCRFD của golden có tỉ lệ đúng bằng 1). Đo thêm
  trên 200 ảnh COCO val có mặt người (40 ảnh × 5 kích thước): 2012 mặt, số mặt và điểm số không
  đổi, x lệch tối đa 3,96 px ảnh gốc (1,38 px input), trung vị 0,18 px.

### Vẫn còn mở sau đợt này (2026-10-04)

- ~~`PredictPrompt` chờ slot session mà không theo ctx; khi quá tải, request multipart sai định
  dạng có thể nhận 503 thay vì 400.~~ **Đã xong (365ad7b).**
  - `PredictPrompt`, `InferTensor`, `Explain`, `Preprocess` và `Load` nhận ctx của request. Mọi
    chỗ chờ trước khi chạy inference đều theo ctx: chờ load model, chờ lock của model
    `Exclusive` (nay là semaphore 1 slot thay cho mutex), chờ session
    (`engine.Runnable.Run(ctx, …)`: chờ worker của session đơn hoặc một session rảnh trong pool).
    Client bỏ đi thì request thôi chờ, không chạy gì, server trả 499.
  - Không ngắt một `Run` đã vào ORT (không ngắt được). Nhưng `Runner` mang ctx của request, nên
    pipeline không chạy stage kế tiếp cho request đã bỏ; Score-CAM dừng giữa các kênh.
  - Load chạy trên goroutine riêng, không thuộc request nào. Một request thôi chờ (kể cả request
    khởi động load) không hủy load của các request khác; load xong thì model vẫn live. Chỉ
    `Unload`/`Close` hủy load.
  - Lease và slot admission được trả trên mọi đường đi. Test: request chờ slot duy nhất (session,
    pool, lock `Exclusive`) bị hủy thì trả về ngay, không chạy, `refs`/admission về 0; load
    singleflight không bị ảnh hưởng khi waiter (và cả leader) hủy; `-race -count=5` sạch. Golden
    43 case trùng từng byte.
  - 503 thay vì 400: **giữ nguyên, có chủ ý** (ghi trong `multipart.go`). Phép thử admission chạy
    trước khi validate form; đa số lỗi định dạng nằm trong hoặc sau part ảnh, muốn thấy thì phải
    đọc đúng phần upload mà phép thử muốn bỏ qua. Kiểm vài field đứng trước ảnh thì phải chép
    validate của từng endpoint vào đó, chỉ để đổi lỗi nào client hỏng thấy trước. Gửi lại sẽ
    nhận 400.
- Kernel GPU tất định đang là opt-in. Bật mặc định thì phải đo lại và cập nhật các con số mAP
  trong manifest. Trên Windows chưa nối. Có thể thay shim cgo bằng một PR nhỏ lên
  `yalue/onnxruntime_go` (thêm `SessionOptions.SetDeterministicCompute`).
- Thread của pool: **đã đo, đã đổi công thức**. Đo end to end trên CPU, giả lập máy 4, 8 (4c/8t
  và 8 core), 16, 24 CPU bằng `taskset`, cộng máy 48 luồng; 3 vòng xen kẽ, lấy trung vị. Mặc
  định của ORT chậm nhất ở mọi cỡ máy, kể cả một request có prompt: máy 4 core, box 1.19 s so
  với 0.47 s, automask 22.1 s so với 6.6 s. Nỗi lo "1 thread/decoder làm request có prompt chậm
  hơn" là sai. Nhưng `NumCPU/(4n)` cho 1 thread tới 31 CPU, nên ở 16 và 24 CPU automask chậm
  1.6–2×. Công thức mới: `NumCPU/(2n)`, kẹp trong [1, 3]. Kết quả: 1 thread tới 15 CPU, 2 ở
  16, 3 từ 24 trở lên (ở 48 vẫn là 3 như cũ). Bảng đo nằm trong comment của
  `poolIntraOpThreads` và trong docs/architecture.md. Output không đổi: golden 43 case trùng từng
  byte với base1; 8 case của các model có pool, chạy ở 16 CPU với 2 thread, 1 thread và mặc định
  ORT, cũng trùng từng byte. Lưu ý khi đo: `runtime.NumCPU()` của Go theo affinity mask, còn
  ORT 1.26 thì không. Dưới `taskset -c 0-3`, mặc định của ORT vẫn tạo 24 thread/session và pin
  ra cả 24 core của máy, nên bảng giả lập mặc định đó bằng 1 thread mỗi core giả lập. Còn mở:
  chưa đo trên board ARM thật và trong container giới hạn bằng cpuset. Sweep `VS_POOL_OVERRIDE` (pool cả detector/encoder) cũng bị
  kẹp ở 3; muốn so với số cũ thì đặt `VISIONSERVE_POOL_THREADS`.
- `head.onnx` của textalign trên catalog HF: **xong** (2026-10-04). Bốn model
  `rfdetr-textalign-dec1`, `-dec1-siglip`, `-dec1-siglip-prod`, `-etri` và tháp text `clip-text`
  (MIT) đã lên repo public `mtbui2010/rfdetr-textalign-ONNX`, mỗi model một thư mục, kèm
  `head.onnx`, `proj.bin`, labels, templates (và `detector-qf.onnx` cho explain của `-etri`).
  `visionserve pull <tên>` tải đủ và kéo theo `clip-text` hoặc `siglip-text`. Manifest do catalog
  sinh ra khớp manifest trong `models/` (`TestTextalignEntriesMatchRepoManifests`); predict trên
  CPU qua server trùng JSON với bản trong `models/` (trừ `duration_ms`), kể cả ở
  `VISIONSERVE_VERIFY=strict`. Registry giờ nhận model "composed một phần" ở verified mode: file
  của chính nó qua source_url + pin, file mượn của model khác qua model sở hữu file đó. Một bản
  checkout vẫn phải tự chạy `export_head_onnx.py` (`*.onnx` không commit), hoặc pull vào một
  thư mục models khác. Còn mở: `conf_threshold` 0.35 của `-dec1` và `-dec1-siglip` được chỉnh
  cho method `gated`, nhưng method mặc định là `exact`, nên request không ghi method trả về rất ít
  hoặc không có detection; chưa đổi vì đổi mặc định là thay đổi output.
- Bộ nhớ automask: đã giảm (xem mục "Điểm nóng"). Việc tắt memory pattern mới áp dụng cho session
  CPU; trên GPU (CUDA/TensorRT) vẫn để mặc định của ORT vì chưa đo VRAM và tốc độ ở đó.
- `internal/vision/preprocess` import `internal/engine` chỉ để dùng kiểu `engine.Tensor`, nên kéo
  theo cgo (binding `yalue/onnxruntime_go`). Vì vậy preprocess và mọi package model không build
  hay test được khi `CGO_ENABLED=0` (ví dụ cross-build mà không có compiler aarch64). Chuyển
  `Tensor` (và các helper như `engine.F32`) sang một package không dùng cgo, `engine` giữ alias,
  thì preprocess và model build/test được mà không cần cgo. Chưa làm.
- Kích thước input trong manifest so với graph: **đã kiểm lúc load** (2026-10-05,
  `checkInputShape`). Còn mở: với PipelineModel chỉ kiểm role `explain` (khi graph của role đó có
  đúng một input); các role khác (encoder SAM, GroundingDINO, tháp text) do `Infer` của model tự
  dựng tensor nên lifecycle không biết tensor nào vào input nào, chưa kiểm. Một mode có kích thước
  đổi theo ảnh (`none`, `long_side`, `keep_aspect`) trên graph có H/W cố định (ngoài
  `keep_aspect`, đã có `checkKeepAspectGraph`) cũng chưa bị từ chối lúc load.
- OWLv2 với template (2026-10-05): decode đúng, khớp HF `Owlv2ForObjectDetection.image_guided_detection`
  (logits lệch tối đa 1.3e-3, box 5e-5; box cuối cùng lệch < 0.1 px, score < 1e-4). Score của
  image query là sigmoid thô, không hiệu chỉnh; ví dụ image-guided của HF dùng `threshold=0.9`,
  còn manifest `owlvit-base` (và `defaultSimThreshold`) dùng 0.1, giá trị của text query, nên ví
  dụ trong docs trả 10 box (4 con chó ≈ 1.0, người 0.90/0.88, dải cỏ 0.30–0.58). Đã sửa:
  `box_threshold` theo request nay có tác dụng với owlvit (trước bị bỏ qua). Còn mở: đổi mặc định
  sang 0.9 là thay đổi output (ví dụ docs: 10 → 5 box), chưa có protocol đo cho template nên chưa
  đổi. Template lỏng (nhiều nền, hoặc hẹp nên bị pad đen) khớp nền khắp ảnh ở score ≈ 1.0 ở cả
  HF lẫn VisionServe; không ngưỡng nào cứu được, docs đã ghi.
