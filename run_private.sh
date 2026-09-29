#!/usr/bin/env bash
# Chạy gói trong Docker (đường ngắn cho BTC). Image đã bake sẵn index + trọng số.
#
#   bash run_private.sh /duong/dan/private-official.json /duong/dan/out
#
# Mặc định: arm both (submission.zip + submission_nollm.zip), 1 GPU (đổi GPU_DEVICE nếu cần).
set -euo pipefail
Q="${1:?cách dùng: bash run_private.sh <questions.json> <out_dir>}"
OUT="${2:?cách dùng: bash run_private.sh <questions.json> <out_dir>}"
GPU_DEVICE="${GPU_DEVICE:-0}"
mkdir -p "$OUT"
docker run --rm --gpus "device=${GPU_DEVICE}" \
  -v "$(dirname "$(realpath "$Q")"):/data:ro" \
  -v "$(realpath "$OUT"):/out" \
  uitdsc2026-legalqa:private \
  "/data/$(basename "$Q")" /out --arm both