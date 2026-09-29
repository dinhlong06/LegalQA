"""Chunking 2 tầng + nạp corpus (Cell 4 + Cell 5 của notebook v11).

Tầng 1 (`chunk_passage_words`): 450 từ liên tục — dùng cho BM25/dense retrieval để CHỌN VĂN BẢN.
Tầng 2 (`chunk_passage`): cắt theo Điều/Mục/Phụ lục/tiết — chỉ dùng làm ứng viên cuối.

Hai hàm và mọi regex dưới đây chép nguyên từ notebook; đổi bất kỳ chỗ nào là đổi thứ tự
chunk ⇒ phải encode lại toàn bộ corpus.
"""
import json
import re
from collections import Counter
from pathlib import Path

DIEU_RE = re.compile(r"^[ \t]*Điều\s+(\d+)[a-zđA-ZĐ]?[\.\s]", re.MULTILINE)
MUC_RE = re.compile(r"^[ \t]*Mục\s+([0-9]+|[IVXLCDM]+)\s*[\.\s:]", re.MULTILINE)
# v11 (Cell 4): nhận hậu tố chữ sau số Phụ lục ("Phụ lục IIa") — khác v7, đổi chunk tầng 2.
PHU_LUC_RE = re.compile(r"^[ \t]*Phụ\s+lục\s+([0-9IVXLCDM]+[a-zđA-ZĐ]?)\b", re.MULTILINE | re.IGNORECASE)
TIET_RE = re.compile(r"^[ \t]*(\d+\.\d+(?:\.\d+)?)\s*[\.\s]", re.MULTILINE)
MAX_UNIT_WORDS = 2000

_MUC_PREFIX_STRIP_RE = re.compile(r"^\s*Mục\s+[0-9IVXLCDM]+\s*[\.\s:]*\s*", re.IGNORECASE)
_PHU_LUC_PREFIX_STRIP_RE = re.compile(r"^\s*Phụ\s+lục\s+[0-9IVXLCDM]+\s*[\.\s:]*\s*", re.IGNORECASE)
_TIET_PREFIX_STRIP_RE = re.compile(r"^\s*\d+\.\d+(?:\.\d+)?\s*[\.\s]*\s*")
_DIEU_PREFIX_STRIP_RE = re.compile(r"^\s*Điều\s+\d+[a-zđA-ZĐ]?\.?\s*", re.IGNORECASE)
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def extract_vb_info(passage: str):
    """(loai_vb, so_hieu) trích từ NỘI DUNG passage, không phải tên file."""
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


def tokenize_simple(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def _split_words_raw(text: str, n: int) -> list:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def chunk_passage(passage: str, doc_id) -> list:
    """Tầng 2 — tầng bậc Điều → Mục → Phụ lục → tiết; đơn vị vượt MAX_UNIT_WORDS bị cắt 450 từ."""
    for regex, unit_type in ((DIEU_RE, "dieu"), (MUC_RE, "muc"),
                             (PHU_LUC_RE, "phu_luc"), (TIET_RE, "tiet")):
        matches = list(regex.finditer(passage))
        if matches:
            break
    else:
        matches, unit_type = [], None

    if not matches:
        words_chunks = _split_words_raw(passage, 450)
        if not words_chunks:
            return [{"id": f"{doc_id}_0", "dieu_so": "0", "unit_type": "raw", "unit_no": "",
                     "loai_vb": "", "so_hieu": "", "text": passage.strip()}]
        return [{"id": f"{doc_id}_w{i}", "dieu_so": "0", "unit_type": "raw450", "unit_no": "",
                 "loai_vb": "", "so_hieu": "", "text": t}
                for i, t in enumerate(words_chunks)]

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
        unit_no = m.group(1)
        unit_text = passage[start:end].strip()
        if len(unit_text.split()) > MAX_UNIT_WORDS:
            for j, sub in enumerate(_split_words_raw(unit_text, 450)):
                chunks.append({"id": f"{doc_id}_{unit_type}{unit_no}_{i}_{j}",
                               "dieu_so": unit_no if unit_type == "dieu" else "0",
                               "unit_type": f"{unit_type}_capped", "unit_no": unit_no,
                               "loai_vb": "", "so_hieu": "", "text": sub})
        else:
            chunks.append({"id": f"{doc_id}_{unit_type}{unit_no}_{i}",
                           "dieu_so": unit_no if unit_type == "dieu" else "0",
                           "unit_type": unit_type, "unit_no": unit_no,
                           "loai_vb": "", "so_hieu": "", "text": unit_text})
    return chunks


def chunk_passage_words(passage: str, doc_id, n: int = 450) -> list:
    """Tầng 1 — 450 từ liên tục, không quan tâm ranh giới Điều."""
    words = passage.split()
    if not words:
        return [{"id": f"{doc_id}_w0", "text": passage.strip()}]
    return [{"id": f"{doc_id}_w{i // n}", "text": " ".join(words[i:i + n])}
            for i in range(0, len(words), n)]


def load_corpus(contexts_dir):
    """-> (all_chunks [tầng 2], all_chunks_t1 [450 từ], dieu_by_doc).

    Thứ tự file `sorted(context_*.json)` là BẤT BIẾN — index bake sẵn dựa vào thứ tự này.
    """
    contexts_dir = Path(contexts_dir)
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {contexts_dir}")
    files = sorted(contexts_dir.glob("context_*.json"))
    if not files:
        nested = contexts_dir / "selected-contexts"
        if nested.exists():
            files = sorted(nested.glob("context_*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy context_*.json trong {contexts_dir}")

    all_chunks, all_chunks_t1, dieu_by_doc = [], [], {}
    for fp in files:
        try:
            with fp.open(encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        passage = doc.get("passage")
        if not passage:
            continue
        doc_id = doc["id"]
        loai_vb, so_hieu = extract_vb_info(passage)
        chunks = chunk_passage(passage, doc_id)
        for c in chunks:
            c["loai_vb"], c["so_hieu"] = loai_vb, so_hieu
        all_chunks.extend(chunks)
        dieu_by_doc[str(doc_id)] = chunks
        for c in chunk_passage_words(passage, doc_id):
            c["loai_vb"], c["so_hieu"] = loai_vb, so_hieu
            all_chunks_t1.append(c)
    return all_chunks, all_chunks_t1, dieu_by_doc


def unit_type_histogram(all_chunks) -> dict:
    return dict(Counter(c.get("unit_type", "dieu") for c in all_chunks))