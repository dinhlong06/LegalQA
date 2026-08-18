"""Trích loại văn bản (Luật/Nghị định/...) và số hiệu (vd 90/2017/NĐ-CP) từ passage gốc của
mỗi văn bản trong corpus — dùng làm lead-in khi ghép câu trả lời template extractive.

Metadata này ở CẤP VĂN BẢN (document_id), không phải cấp chunk: chunk trong retrieval/db_fast
được cắt theo độ dài cố định 450 từ (retrieval/prepare_data/build_db_corpus_dsc.py:split_chunks),
không neo theo ranh giới Điều, nên không có cách suy ra số Điều cho từng chunk một cách đáng tin.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)


def extract_vb_info(passage: str) -> tuple:
    """-> (loai_vb, so_hieu), rỗng nếu không trích được."""
    m = SO_HEADER_RE.search(passage[:1500])
    so_hieu = m.group(1).strip("., ") if m else ""
    if not (so_hieu and SO_HIEU_RE.fullmatch(so_hieu)):
        m2 = SO_HIEU_RE.search(passage[:1500])
        so_hieu = m2.group(0) if m2 else ""
    m3 = LOAI_PATTERN.search(passage[:200]) or LOAI_PATTERN.search(passage[:800])
    loai_vb = ""
    if m3:
        low = m3.group(1).lower()
        for canon in LOAI_VB_CANON:
            if canon.lower() == low:
                loai_vb = canon
                break
    return loai_vb, so_hieu


def build_doc_meta(contexts_dir) -> dict:
    """contexts_dir chứa context_<id>.json ({"id","name","link","passage"}).
    -> {document_id (str): {"loai_vb": str, "so_hieu": str}}"""
    contexts_dir = Path(contexts_dir)
    meta = {}
    n_empty = 0
    files = sorted(contexts_dir.glob("context_*.json"))
    for fp in files:
        try:
            with fp.open(encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        passage = doc.get("passage")
        if not passage:
            continue
        loai_vb, so_hieu = extract_vb_info(passage)
        if not loai_vb and not so_hieu:
            n_empty += 1
        meta[str(doc["id"])] = {"loai_vb": loai_vb, "so_hieu": so_hieu}
    if files:
        pct = round(100 * (1 - n_empty / len(files)), 2)
        print(f"  doc_meta: {len(files)} văn bản -> {pct}% trích được loai_vb/so_hieu.")
    return meta


def load_or_build_doc_meta(contexts_dir, cache_path) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)
    meta = build_doc_meta(contexts_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta
