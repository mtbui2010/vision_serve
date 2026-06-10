# Đánh giá khả năng accept tại IEEE Access — VisionServe

**Bài:** *VisionServe: A Lean, License-Safe Inference Server for Computer Vision* (Trung Minh Bui, KETI)
**Nguồn đọc:** `paper/main.pdf` + `paper/supplementary.pdf`
**Ngày đánh giá:** 2026-06-09

---
---

# 🔄 CẬP NHẬT — Re-check sau khi revise (2026-06-09, vòng 2)

> Bản PDF đã được revise. Dưới đây là kết quả phản biện + check AI tone lần 2. Đánh giá vòng 1 (trước revise) được giữ nguyên ở phần sau để đối chiếu.

## A. Verdict sau revise

| | Vòng 1 (trước) | **Vòng 2 (sau revise)** |
|---|---|---|
| Khả năng accept (hiện trạng) | ~35–45% | **~60–70%** |
| Template | ❌ NeurIPS | ✅ **IEEEtran journal** (`\documentclass[journal]{IEEEtran}`, có Index Terms) |
| Lỗi đỏ biên tập | 3 lỗi | ✅ **Đã sửa hết** (xem B) |
| AI tone | 🔴 Rất nặng | ✅ **Sạch về cơ bản** (xem C) |

**Tóm lại:** ba rào cản "cứng" của vòng 1 (sai template, lỗi biên tập, AI-tone) **đều đã được xử lý**. Bài giờ ở trạng thái có thể nộp; rủi ro còn lại chỉ là **vài lỗi nhỏ** (D) và một câu hỏi novelty *mềm* mà IEEE Access vốn dễ tính.

## B. Các lỗi đỏ vòng 1 — đã sửa ✅

1. ✅ **Template** → đã chuyển sang `IEEEtran` 2 cột, có `Index Terms` (Computer vision, inference serving, model licensing, software supply chain, edge computing, ONNX Runtime, open-vocabulary detection). Không còn rủi ro desk-reject vì format.
2. ✅ **Abstract nhân đôi** ("serves 16 / serves 19") → đã gộp thành **một** câu: *"VisionServe serves 19 registered models across seven task types"*.
3. ✅ **Mâu thuẫn số model/task** → nay nhất quán **19 registered models / seven task types** ở tất cả section (kể cả `limitations.tex` đã đổi "all six"→"all seven", "16"→"19").

## C. Check AI tone lần 2 — cải thiện rất mạnh ✅

Đo lại trực tiếp trên `paper/sections/*.tex` (đã loại các dòng comment `% ----`):

| Tín hiệu AI | Vòng 1 | **Vòng 2** | Ghi chú |
|---|---|---|---|
| **Em-dash `—` trong prose** | **163** | **9** | catalog 5 + evaluation 4; chủ yếu là dùng kỹ thuật hợp lệ (mô tả grasp, arrow). Trong PDF render: abstract/intro/related/design/conclusion **0 em-dash tu từ** |
| Italic nhấn giọng `\emph/\textit` | 112 | **15** | giảm 87% |
| Crucially/Notably/Importantly/Critically… | nhiều | **0** | đã bỏ sạch trạng từ mở câu |
| "we do not claim" | 4 | 3 | còn, nhưng hợp lý (định vị scope) |
| "genuinely uncontested" | — | 2 | còn đậm chất quảng cáo nhẹ |
| "to our knowledge" / "orthogonal" | — | 2 / 1 | mức chấp nhận được |

→ Văn nay đọc **như prose học thuật bình thường**: câu được tách bằng dấu phẩy/ngoặc/dấu chấm thay vì em-dash; bỏ meta-hedging rải rác. Tác giả còn thêm `\newunicodechar{—}{---}` trong `main.tex` để map em-dash Unicode về ligature LaTeX (không ảnh hưởng nội dung).

**Khuyến nghị nhỏ còn lại (không bắt buộc):** đổi 2 chỗ *"genuinely uncontested"* → diễn đạt trầm hơn (vd "the one axis without a direct competitor"); rà 9 em-dash còn lại trong catalog/evaluation xem có thể thay bằng dấu phẩy/ngoặc không. Đây là cosmetic, không phải blocker.

## D. Vấn đề CÒN LẠI cần xử lý trước khi nộp

1. **🟠 Mâu thuẫn "grounded-sam đã/​chưa benchmark".** Hai chỗ nói ngược nhau:
   - `catalog.tex:119–120`: *"All four grasp/composite pipelines (grasp, grasp-rfdetr, **grounded-sam**, grasp-gd) are now benchmarked (Section V-B)."*
   - `evaluation.tex:88–89` (caption Table V): *"Four models (rt-detr, depth-anything-v2, nano-sam, **grounded-sam**) are not yet benchmarked and are omitted."*

   Có thể reconcile (Table V là bảng latency single-stream; grounded-sam được benchmark riêng ở V-B), nhưng câu chữ *"not yet benchmarked"* trực tiếp chọi với *"now benchmarked"*. **Sửa:** đổi caption Table V thành "…are omitted from this single-stream latency table" và bỏ "not yet benchmarked" cho grounded-sam, hoặc thay grounded-sam bằng một model thực sự pending.

2. **🟠 Nghi va chạm layout ở trang 14 (Table XII).** Khi render, Table XII (Published accuracy metrics) ở đầu cột phải **trông như đè lên** đoạn văn Applications ("…ployments impose even before any model weights are loaded… 4–8 GB of unified memory"). Có thể là float placement collision của IEEEtran (bảng single-column đặt `[t]`). **Cần mở PDF xem mắt trang 14**; nếu đè thật thì chỉnh vị trí float (`[!t]`/`[h]`) hoặc thu bảng.

3. **🟡 Số model benchmark cần khớp rõ.** Abstract/intro nói *"16 of 19 models benchmarked"*; Table V caption nói *"all 12 models"* + 4 omitted (=16 còn lại gồm 12 ở Table V + 4 grasp/composite). Logic ổn nhưng nên thêm một câu chốt "12 single models in Table V + 4 grasp/composite pipelines in V-B = 16 of 19" để reviewer không phải tự cộng.

## E. Phản biện nội dung (novelty & baseline) — trạng thái sau revise

- **Novelty tự bào mòn:** vẫn còn nhượng bộ ("we concede the No-Python and single-binary axes", "multi-EP is ONNX Runtime's, not ours"), **nhưng đã cân bằng tốt hơn**: abstract/intro nay nói thẳng *"The primary contribution is the license gate; the remaining items are the system and the measurements that support it"*, và đóng khung đóng góp = *"the combination under the license gate"*. Với IEEE Access (không đòi breakthrough) mức này **chấp nhận được**. Không cần sửa thêm; chỉ cần giữ license gate là trục chính như hiện tại.
- **Baseline default-config:** vòng 1 mình lo reviewer đòi tuned Triton. Revise đã **chủ động xử lý** trong Limitations: nêu rõ không claim parity với tuned Triton, và **định lượng** hướng/độ lớn khoảng cách ("dynamic batching reported to lift Triton to ~780 rps at scale [13]", "tuned config plausibly improves by a similar fraction"). Đây là cách trả lời reviewer rất tốt — giữ nguyên.
- **Trung thực + reproducibility:** vẫn là điểm mạnh nhất (provenance sidecar, `errors=0`, vegeta re-measure cho tail, export parity cosine >0.999999). Đúng gu IEEE Access.

## F. Việc cần làm (ưu tiên) sau vòng 2

1. Sửa mâu thuẫn grounded-sam (D1) — **bắt buộc**, reviewer dễ bắt.
2. Kiểm tra mắt + fix va chạm Table XII trang 14 (D2) — **bắt buộc nếu đè thật**.
3. Thêm câu chốt "16 = 12 + 4" (D3) — nên làm.
4. (Cosmetic) hạ 2 chỗ "genuinely uncontested" + rà 9 em-dash còn lại (C).

> **Kết luận vòng 2:** Bài đã vượt qua các rào cản hình thức/biên tập/AI-tone của vòng 1. Sau khi sửa nốt mục D (chủ yếu D1, D2), đây là một bản thảo systems **chắc, trung thực, reproducible** với một đóng góp phân biệt rõ (license-enforcement gate) — **khả năng accept khá (~60–70%)**, dự kiến qua một vòng minor/major revision điển hình của IEEE Access.

---
---

# 📄 Đánh giá VÒNG 1 (trước revise) — giữ để đối chiếu

---

## 0. Verdict nhanh

| Tiêu chí | Đánh giá |
|---|---|
| **Khả năng accept (giữ nguyên hiện trạng)** | **Thấp–trung bình (~35–45%)** |
| **Khả năng accept (sau khi sửa các lỗi bên dưới)** | **Trung bình–khá (~60–70%)** |
| Rủi ro lớn nhất #1 | **Sai template** — đang dùng NeurIPS 2024, không phải IEEE Access (desk-reject) |
| Rủi ro lớn nhất #2 | **Novelty bị chính tác giả "tự bào mòn"** — paper liên tục thừa nhận gần như mọi trục đều không mới |
| Rủi ro lớn nhất #3 | **Văn phong nặng "AI tone"** + lỗi copy-paste lộ liễu trong Abstract |

IEEE Access **không** chấm theo "độ mới đột phá" như hội nghị top; nó chấm **technical soundness + tính hữu dụng + trình bày rõ ràng + reproducibility**. Trên các trục đó bài này thực sự *mạnh*. Nhưng IEEE Access có vài lý do reject "cứng" (sai format, tiếng Anh/biên tập kém, mâu thuẫn nội tại) — và bài đang dính đúng mấy lỗi đó. Đây là tin tốt: phần lớn rủi ro là **sửa được**, không phải lỗi khoa học.

---

## 1. Mức độ phù hợp với tiêu chí IEEE Access

IEEE Access review theo checklist nhị phân (mỗi mục Yes/No), không xếp hạng "significance":

1. **Original & chưa publish** — ✅ Đạt (cần đảm bảo arXiv preprint OK; IEEE Access cho phép preprint).
2. **Technically sound, kết luận có dữ liệu chống lưng** — ✅ **Rất mạnh.** Mọi số đều đo thật, có provenance sidecar, invariant `errors=0`, supplementary tách bạch artefact↔bảng. Đây là điểm bán hàng lớn nhất với reviewer Access.
3. **So sánh prior art đầy đủ** — ✅ Đạt (Related Work dày, Table 1 so sánh hệ thống, threat model định vị rõ với SCA/admission controller/model signing).
4. **Tiếng Anh & trình bày rõ ràng** — ⚠️ **Rủi ro.** Nội dung mạch lạc nhưng văn phong rất nặng "AI tone" (xem §3), có lỗi biên tập lộ liễu (§4).
5. **Hữu dụng cho cộng đồng** — ✅ Đạt. "Ollama for CV", Apache-2.0, edge-first, có client lib — rất hợp gu reader Access (kỹ sư ứng dụng).
6. **Format đúng journal** — ❌ **Không đạt** (NeurIPS template).

→ Điểm yếu duy nhất mang tính "khoa học" là **độ mới**; còn lại đều là vấn đề **trình bày/biên tập sửa được**.

---

## 2. Điểm mạnh (nên giữ và làm nổi bật)

- **Reproducibility xuất sắc.** Sidecar `*.meta.json` (git commit, host, GPU clock, loadgen version, UTC), bảng provenance result→table (S2), lệnh chạy đầy đủ. Reviewer Access rất thích cái này — nó gần như loại bỏ nghi ngờ "số bịa".
- **Tự phê bình trung thực.** Thừa nhận thua ở low-load (57 vs 16 ms), 8× chỉ đúng với single-worker default (multi-process đạt parity), Triton mới là default-config, tail p99.9 là artefact của client Python rồi đo lại bằng `vegeta`. Sự trung thực này làm reviewer tin phần còn lại.
- **Đánh giá đa chiều, đo trên phần cứng thật:** A6000 + Jetson AGX Orin, device×EP matrix, năng lượng mJ/inference, footprint RSS/VRAM, accuracy export parity (cosine >0.999999) + served Top-1 thật.
- **License gate là đóng góp thật sự khác biệt.** "Enforce license policy at model-load time" (refuse-to-load) đối lập với toàn bộ prior art chỉ "analyze-and-warn" (ModelGo, MG Analyzer, LiAgent) hoặc "pay-to-relicense" (Roboflow). Adversarial demo (Table 4 / S3) tái lập được. Đây là phần đáng để làm trục chính.
- **Phạm vi rộng nhưng nhất quán schema:** 7 task, `Result` schema thống nhất, RLE column-major.

---

## 3. Phân tích "AI tone" — phần được yêu cầu trọng tâm

> Câu hỏi: các blogger/researcher hay chỉ ra điểm chung nào của văn AI (ví dụ dấu `—`)? Bài này dính bao nhiêu?

### 3.1 Các dấu hiệu "AI giveaway" mà cộng đồng hay nêu

Tổng hợp các tín hiệu được giới review/blogger (Editors, các thread "how to spot AI writing", Paul Graham về em-dash, các tài liệu hướng dẫn reviewer) nhắc đi nhắc lại:

1. **Lạm dụng em-dash `—`** — dấu hiệu *số một* hiện nay. Văn người ít khi nhồi em-dash mỗi câu.
2. **Cấu trúc "not X but/rather Y", "is precisely/exactly Y"** — đối lập tu từ kiểu máy.
3. **Trạng từ mở câu lặp:** *Crucially, Notably, Importantly, Critically, Counter-intuitively, Concretely.*
4. **Meta-bình luận tự trấn an:** "We are explicit about", "We report this honestly", "We are deliberately self-critical", "We do not claim".
5. **Rule-of-three / liệt kê song song dày đặc:** "no account, no API key, no cloud, no telemetry".
6. **Nhồi italic/emphasis** để nhấn giọng (*not*, *is*, *precisely*, *by construction*).
7. **Từ vựng "AI-favourite":** *lean, minefield, honest reading, distinguishing locus, structurally, by construction, foreclosing, viral(ly), orthogonal and complementary.*
8. **Bullet nào cũng có lead-in in đậm** + câu cân đối hoàn hảo, ít "lỗi người".

### 3.2 Bài này dính bao nhiêu (đo thực tế từ `paper/sections/*.tex`)

| Tín hiệu | Số đếm trong bài | Mức độ |
|---|---|---|
| **Em-dash `---`** | **163 trong sections** (≈255 cả `main.tex`), riêng `design.tex` 56 và `related.tex` 55 | 🔴 **Rất nặng** (~6–7 dấu/trang) |
| `\emph`/`\textit` (italic nhấn giọng) | **112** lần | 🔴 Nặng |
| `precisely` | 5 | 🟠 |
| `we do not claim` | 4 | 🟠 |
| `Crucially` | 3 | 🟠 |
| `We are explicit` / `self-critical` / `honest reading` | 2+1+2 | 🟠 |
| `by construction` / `structurally` | 2+2 | 🟠 |
| `Critically`, `Notably`, `Importantly`, `Counter-intuitively` | mỗi từ 1–2 | 🟠 |

Ví dụ điển hình (trích nguyên văn):
- *"a license minefield"*, *"the distinguishing locus of VisionServe is precisely this"*, *"VisionServe's gate is orthogonal and complementary"*.
- *"not 'compliant vs. negligent' but two opposite **strategies**"* — đúng khuôn "not X but Y".
- *"We are explicit about what it is **not**"*, *"We scope this honestly"*, *"We are deliberately self-critical about the **tail**"*.
- Abstract: *"no account, no API key, no cloud, no telemetry"* (rule-of-four).

### 3.3 Hệ quả & khuyến nghị

- **Không reviewer nào reject vì "nghi AI viết".** Nhưng mật độ em-dash + meta-hedging cao bất thường khiến văn đọc như **mệt, lặp giọng**, và làm reviewer **giảm tin vào tính chín của bản thảo** (một dạng "halo effect" ngược).
- **Việc paper liên tục "thú nhận" (we concede / we do not claim / not novel / inherited)** vừa là AI-tone *vừa* là vấn đề định vị (§5.1): nó tự đánh tụt đóng góp của chính mình.

**Hành động cụ thể (rẻ, hiệu quả cao):**
1. Giảm em-dash xuống còn **~1–2/trang**: đổi phần lớn `---` thành dấu phẩy, ngoặc đơn, hoặc tách câu.
2. Cắt 60–70% italic nhấn giọng; chỉ giữ italic cho thuật ngữ lần đầu.
3. Xóa các trạng từ mở câu thừa (*Crucially/Notably/Importantly*) — gần như luôn bỏ được.
4. Gộp các câu "we do not claim / we concede / honestly" rải rác thành **một** đoạn "Scope & honest limitations" duy nhất, viết giọng trầm tĩnh, thay vì rải khắp bài.
5. Đa dạng độ dài câu; thêm vài câu ngắn, trực tiếp để phá nhịp đều đều.

---

## 4. Lỗi cụ thể PHẢI sửa trước khi nộp (reviewer chắc chắn bắt)

1. **🔴 Lỗi copy-paste lộ liễu trong Abstract.** `sections/abstract.tex` dòng 28–29:
   > "...VisionServe also serves **16** / VisionServe also serves **19** registered models across 7 CV task types..."

   Câu bị nhân đôi *và* mâu thuẫn số (16 vs 19) ngay trong abstract. Đây là loại lỗi làm reviewer mất thiện cảm tức thì.

2. **🔴 Mâu thuẫn số model & số task xuyên suốt bài:**
   - "19 registered models" (introduction, conclusion, catalog) **vs** "16 registered models" (limitations).
   - "seven task types" (intro/design/catalog/conclusion) **vs** "all six task types" (limitations).
   → Thống nhất một con số duy nhất (kèm chú thích bao nhiêu model đã benchmark vs pending).

3. **🔴 Sai template (desk-reject risk).** `main.tex` đang `\usepackage[preprint]{neurips_2024}` + `\documentclass{article}`. IEEE Access **bắt buộc** dùng template `IEEEtran` (`\documentclass[journal]{IEEEtran}`, 2 cột, có Biography, Index Terms thay vì abstract NeurIPS-style). Đây là việc bắt buộc, không phải tùy chọn.

4. **🟠 Title + framing "license-safe".** Đóng góp lõi (allowlist check ở registry parser + ledger) cần được trình bày như **cơ chế enforcement có threat model**, tránh để reviewer quy về "vài chục dòng if-check". Phần adversarial demo + ledger separation-of-authority chính là thứ nâng nó lên mức "research" — hãy đẩy nó lên sớm.

---

## 5. Điểm yếu mang tính khoa học (rủi ro reject thật sự)

### 5.1 Novelty bị tự bào mòn
Bài *liên tục* nhượng bộ: "concede the No-Python and single-binary axes to prior art", "Multi-EP is ONNX Runtime's, not ours", "single-binary is shared with llama.cpp", "8× holds only against single-worker default". Trung thực là tốt, nhưng **liều lượng hiện tại khiến net contribution đọc ra rất mỏng**. Reviewer khó tính có thể kết luận: "đây là kỹ thuật/engineering tốt, không phải đóng góp nghiên cứu."

**Khắc phục:** Định khung lại theo **một** đóng góp khoa học sắc nét = *load-time license-policy enforcement gate + audited provenance ledger với separation-of-authority, kèm threat model & adversarial demo tái lập được*. Các trục còn lại (single-binary, multi-EP, crossover) trình bày như **đánh giá thực nghiệm hỗ trợ**, không phải "contribution" ngang hàng — để không tự mời reviewer bác từng cái một.

### 5.2 Baseline là "default-config"
8× và parity-with-Triton đều so với cấu hình mặc định (1 uvicorn worker; Triton HTTP, no batching). Bài đã thừa nhận, nhưng reviewer sẽ hỏi tuned baseline. **Giữ nguyên sự trung thực**, nhưng nêu rõ trong abstract/intro rằng claim là "một binary không tune sánh với deployment Python 8-process" — đừng để con số 8× đứng trần.

### 5.3 Đóng góp = "combination"
Paper tự gọi đóng góp là "the combination" của 4 thành phần đã có. Với IEEE Access điều này **chấp nhận được** (Access không đòi breakthrough), miễn là (a) chứng minh combination giải quyết vấn đề thực và (b) đo nghiêm túc — bài làm được cả hai. Nhưng cần nói thẳng giá trị: *first Python-free, license-enforcing, multi-task CV server*.

### 5.4 Vài hạn chế kỹ thuật còn mở (đã khai báo)
GroundingDINO pipelines phải serialize (bug `SA_ONSTACK`), grasp automask nặng, single-image batching only, Windows chưa test, ROCm chưa có. Đều đã liệt kê trung thực ở §7 — ổn cho Access, không cần giấu.

---

## 6. Khuyến nghị hành động (ưu tiên giảm dần)

1. **Chuyển sang template IEEEtran (IEEE Access).** Bắt buộc. Thêm Index Terms, đổi cấu trúc abstract, biography.
2. **Sửa 3 lỗi đỏ ở §4** (abstract nhân đôi, số 16/19, six/seven). Rẻ, ảnh hưởng lớn tới ấn tượng đầu.
3. **Edit pass giảm AI-tone (§3.3):** em-dash → ~1–2/trang, cắt italic & trạng từ mở câu, gom phần "honest scope" lại một chỗ.
4. **Định khung lại novelty quanh license gate** (§5.1); hạ các trục "inherited" xuống mục evaluation.
5. **Làm rõ baseline default-config ngay ở abstract/intro** (§5.2).
6. (Tùy chọn, tăng điểm) Thêm một baseline tuned nhẹ hoặc ít nhất thảo luận định lượng "tuned Triton sẽ vượt bao nhiêu" để chặn câu hỏi reviewer.

---

## 7. Kết luận

Về **nội dung**, đây là một bài systems/engineering **chắc tay, đo đạc trung thực, reproducible cao** — đúng tạng IEEE Access ưa thích, và license-enforcement gate là một đóng góp thật sự phân biệt với prior art. Khả năng accept **không** bị chặn bởi chất lượng khoa học.

Nó bị chặn bởi ba thứ **sửa được**: (1) **sai template** (rủi ro desk-reject), (2) **lỗi biên tập + mâu thuẫn số liệu** lộ liễu (abstract nhân đôi, 16 vs 19 model, six vs seven task), và (3) **văn phong AI-tone quá nặng** (≈163–255 em-dash, 112 italic nhấn giọng, nhiều meta-hedging) cộng với việc **tự bào mòn novelty**.

Sửa xong ba nhóm trên, bài có cơ hội accept **khá (~60–70%)**, có thể qua một vòng minor/major revision điển hình của IEEE Access.
