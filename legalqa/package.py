"""Đóng gói answers -> submission.zip đúng hợp đồng của scoring/legalqa/scoring.py:
- submission.json DUY NHẤT trong zip (không thư mục cha).
- Tập key phải khớp KHÍT ground truth — thiếu/thừa khiến scorer raise -> 0 điểm TOÀN BÀI
  (không phải 0 điểm 1 câu), vì `len(ids_preds) != len(ids_truth)` raise ngay từ đầu.
- Mỗi answer là string không rỗng — dict/list bị str() hoá sẽ chấm gần 0.
Xem scoring/SCORING_LegalQA.md.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path


def build_submission(answers: dict, expected_ids: set, out_zip) -> None:
    out_zip = Path(out_zip)
    errors = []
    got = set(answers.keys())
    if got != expected_ids:
        errors.append(f"Key lệch: thiếu {len(expected_ids - got)}, thừa {len(got - expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            errors.append(f"[{qid}] answer rỗng hoặc không phải string")
    if errors:
        raise ValueError("Submission KHÔNG hợp lệ:\n  - " + "\n  - ".join(errors[:20]))

    normalized = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    json_path = out_zip.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")

    with zipfile.ZipFile(out_zip) as zf:
        assert zf.namelist() == ["submission.json"]
        reloaded = json.loads(zf.read("submission.json").decode("utf-8"))
        assert reloaded == normalized
    print(f"  OK — {out_zip} ({len(normalized)} câu trả lời, đã kiểm tra lại từ đĩa)")
