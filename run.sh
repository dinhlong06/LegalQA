#!/usr/bin/env bash
# Tự chọn 1 GPU đang trống (trong TẤT CẢ GPU 0-7) rồi chạy `python -m rag_model.run`.
#
# Dùng khoá flock trước khi chiếm GPU — tái dùng nguyên eval/gpu_lib.sh (không viết bộ
# chọn GPU riêng: viết riêng dễ giành GPU với job khác của người khác trên máy, xem
# comment trong file đó — đã có sự cố thật vì lỗi này). gpu_lib.sh mặc định chỉ cho chọn
# GPU 4-7 (quy ước riêng của các job trong eval/); ở đây mở hết ra 0-7 vì chỉ chạy tạm
# thời (predict/eval), không phải job train chiếm GPU nhiều giờ như các job trong eval/.
#
#   ./rag_model/run.sh --mode eval
#   ./rag_model/run.sh --mode full
#   GPU_THRESH_MIB=6000 ./rag_model/run.sh --mode full   # đòi GPU trống nhiều hơn mặc định
set -euo pipefail
cd "$(dirname "$0")/.."   # về gốc repo — rag_model.run bắt buộc chạy bằng -m từ đây
ALLOW_GPUS="${ALLOW_GPUS:-0 1 2 3 4 5 6 7}"
source eval/gpu_lib.sh

_run(){
  echo "==> Dùng GPU $G (CUDA_VISIBLE_DEVICES=$G)"
  CUDA_VISIBLE_DEVICES=$G python -m rag_model.run "$@"
}

# Ngưỡng VRAM trống tối thiểu (MiB) trước khi chiếm GPU. `rag_model.run` giờ chạy 2 PHA
# TÁCH RỜI (xem rag_model/retrieve/candidate.py, rerank.py): pha 1 chỉ nạp 2 encoder
# (bge-m3+e5-large), pha 2 chỉ nạp reranker — không bao giờ giữ cả 3 model cùng lúc như
# bản cũ (lúc đó cần ngưỡng 6000 theo eval/job_bgem3_eval.sh). Đỉnh VRAM giờ là pha 1
# (2 encoder + hoạt động, ước lượng ~4GB) — hạ ngưỡng theo đó, còn dư cho phân mảnh.
# Mỗi pha còn tự gọi reserve_vram() (--reserve-gb) ngay sau khi model lên GPU để giữ
# đúng phần đã cấp, không bị người khác chen vào giữa 2 pha.
THRESH="${GPU_THRESH_MIB:-4500}"

echo "==> Cần GPU trống >= ${THRESH} MiB, đang xét trong: ${ALLOW_GPUS}"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | awk -v allow="$ALLOW_GPUS" '
  BEGIN { n = split(allow, a, " "); for (i = 1; i <= n; i++) ok[a[i]] = 1 }
  { gsub(",", ""); if ($1 in ok) print "    GPU " $1 ": " $2 " MiB free" }'

# Vòng lặp chờ bên trong run_on_gpu (eval/gpu_lib.sh) KHÔNG in gì cả — hợp lý cho job chạy
# nền hàng giờ, nhưng chạy tương tác thế này thì trông như treo. In nhắc mỗi 30s cho biết
# script vẫn sống, không phải bị treo.
( while :; do sleep 30; echo "    ... vẫn đang chờ GPU trống (Ctrl+C để huỷ, hoặc hạ GPU_THRESH_MIB)"; done ) &
WAITER=$!
trap 'kill "$WAITER" 2>/dev/null' EXIT

run_on_gpu "$THRESH" _run "$@"
