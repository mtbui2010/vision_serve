#!/usr/bin/env bash
# Demo VisionServe: chạy RF-DETR detection trên vài ảnh COCO thật, xuất ảnh có vẽ bbox.
#
# Ưu tiên dùng weights THẬT (models/rf-detr/rf-detr-base-real.onnx) nếu có; nếu không
# có thì rơi về dummy (gen_dummy_onnx.py) — detection sẽ là tổng hợp (car/cat cố định),
# chỉ minh hoạ luồng. KHÔNG đụng tới file dummy mặc định của repo: dùng thư mục models
# riêng (demo/models) trỏ tới weights qua symlink.
#
# Dùng: ORT_DYLIB_PATH=/path/libonnxruntime.so ./scripts/demo.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DEMO="$ROOT/demo"
IMG_DIR="$DEMO/images"
OUT_DIR="$DEMO/out"
DMODELS="$DEMO/models"
SRC_MODEL="$ROOT/models/rf-detr"
BIN="$ROOT/bin/visionserve"

# --- 1. ONNX Runtime shared lib ---
if [ -z "${ORT_DYLIB_PATH:-}" ]; then
  echo "→ ORT_DYLIB_PATH chưa đặt, thử tự dò libonnxruntime.so ..."
  # loại node_modules (có bản arm64 sẽ lỗi trên x86); ưu tiên bản ORT đầy đủ
  # (onnxruntime/capi) và bản có version suffix (.so.X.Y.Z).
  cands="$(find "$HOME" /usr/local/lib /usr/lib -name 'libonnxruntime.so*' 2>/dev/null | grep -v node_modules || true)"
  ORT_DYLIB_PATH="$(printf '%s\n' "$cands" | grep 'onnxruntime/capi.*\.so\.[0-9]' | head -1 || true)"
  [ -z "$ORT_DYLIB_PATH" ] && ORT_DYLIB_PATH="$(printf '%s\n' "$cands" | grep -v '^$' | head -1 || true)"
  if [ -z "$ORT_DYLIB_PATH" ]; then
    echo "✗ Không tìm thấy libonnxruntime.so. Hãy đặt ORT_DYLIB_PATH thủ công." >&2
    exit 1
  fi
  echo "  dùng: $ORT_DYLIB_PATH"
fi
export ORT_DYLIB_PATH

# --- 2. Build binary ---
if [ ! -x "$BIN" ]; then
  echo "→ build binary ..."
  ( command -v make >/dev/null && make build ) || go build -o "$BIN" ./cmd/visionserve
fi

# --- 3. Chuẩn bị thư mục models demo (không clobber dummy mặc định) ---
mkdir -p "$DMODELS/rf-detr" "$IMG_DIR" "$OUT_DIR"
cp -f "$SRC_MODEL/manifest.yaml" "$SRC_MODEL/coco91.txt" "$DMODELS/rf-detr/"
[ -f "$SRC_MODEL/coco.txt" ] && cp -f "$SRC_MODEL/coco.txt" "$DMODELS/rf-detr/" || true

if [ -f "$SRC_MODEL/rf-detr-base-real.onnx" ]; then
  ln -sf "$SRC_MODEL/rf-detr-base-real.onnx" "$DMODELS/rf-detr/rf-detr-base.onnx"
  echo "→ dùng weights THẬT: rf-detr-base-real.onnx"
  REAL=1
else
  echo "→ KHÔNG có weights thật — sinh dummy (detection tổng hợp, chỉ minh hoạ luồng)."
  echo "  (tải weights thật: xem models/rf-detr/README.md)"
  python "$SRC_MODEL/gen_dummy_onnx.py" >/dev/null
  cp -f "$SRC_MODEL/rf-detr-base.onnx" "$DMODELS/rf-detr/rf-detr-base.onnx"
  REAL=0
fi

# --- 4. Tải ảnh COCO mẫu (idempotent) ---
declare -a IMAGES=(
  "000000039769"  # 2 mèo + 2 remote trên sofa
  "000000000139"  # phòng khách: người, tivi, ghế
  "000000001268"  # người + đồ vật
)
for id in "${IMAGES[@]}"; do
  f="$IMG_DIR/$id.jpg"
  if [ ! -s "$f" ]; then
    echo "→ tải ảnh COCO $id ..."
    curl -sL "http://images.cocodataset.org/val2017/$id.jpg" -o "$f" || {
      echo "  ✗ tải $id thất bại (mạng?) — bỏ qua" >&2; rm -f "$f"; }
  fi
done

# --- 5. Chạy detection + xuất ảnh có bbox ---
echo
echo "===================== KẾT QUẢ ====================="
shopt -s nullglob
for f in "$IMG_DIR"/*.jpg; do
  name="$(basename "${f%.jpg}")"
  out_png="$OUT_DIR/$name.png"
  out_json="$OUT_DIR/$name.json"
  "$BIN" run --models "$DMODELS" rf-detr "$f" --out "$out_png" >"$out_json" 2>>"$OUT_DIR/_run.log" || {
    echo "✗ lỗi khi chạy $name (xem $OUT_DIR/_run.log)" >&2; continue; }
  n=$(grep -c '"class"' "$out_json" || true)
  echo "• $name → $out_png   ($n detection, JSON: $out_json)"
done

echo "==================================================="
[ "${REAL}" = "1" ] && echo "Weights: THẬT (detection thực tế)." || echo "Weights: DUMMY (detection tổng hợp)."
echo "Ảnh có bbox nằm trong: $OUT_DIR/"
