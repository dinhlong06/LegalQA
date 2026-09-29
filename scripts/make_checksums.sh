#!/usr/bin/env bash
# Sinh SHA256SUMS cho trọng số + index + bài đối chứng (BTC đối chiếu tính toàn vẹn).
#   bash scripts/make_checksums.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # gói legalqa/
CHECK="$HERE/../check"                                      # bộ verify của đội (ngoài gói)
cd "$HERE"

find models index -type f \( -name '*.safetensors' -o -name '*.npy' -o -name '*.json' \
    -o -name '*.json.gz' -o -name '*.model' -o -name '*.txt' -o -name '*.jinja' \) \
    -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
shopt -s nullglob
for f in "$CHECK"/reference_legalqa/*.zip "$CHECK"/reference_legalqa/*.json; do
    sha256sum "../check/reference_legalqa/$(basename "$f")" >> SHA256SUMS
done
echo "Đã ghi SHA256SUMS ($(wc -l < SHA256SUMS) dòng)"