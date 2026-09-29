"""Dựng câu trả lời trích xuất — template T3_theo_qd_tai_title + câu kết echo2 (Cell 11/11b).

`render_answer` là arm `base` (và cũng là arm `nollm`): câu dẫn theo loại đơn vị (Điều/Mục/
Phụ lục/tiết) + trích nguyên văn + câu kết lặp lại câu hỏi hai lần. Câu kết là thay đổi rẻ
nhất và đáng giá nhất của pipeline (echo2: +0,0131 METEOR so với echo, split-half cả hai
nửa cùng chọn — xem docstring gốc ở Cell 11).
"""
from .chunking import (
    MUC_RE, PHU_LUC_RE, TIET_RE, _DIEU_PREFIX_STRIP_RE, _MUC_PREFIX_STRIP_RE,
    _PHU_LUC_PREFIX_STRIP_RE, _TIET_PREFIX_STRIP_RE,
)

import re

ANSWER_TEMPLATE = "T3_theo_qd_tai_title"
TRIM_KHOAN_WINDOW = 4   # v11: cửa sổ khoản quanh khoản overlap cao nhất

_V11_KHOAN_RE = re.compile(r"^[ \t]*(\d{1,2})\.[ \t]+", re.MULTILINE)
_V11_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_V11_STOPWORDS = {"của", "và", "các", "được", "cho", "là", "có", "trong", "này", "về",
                  "theo", "với", "để", "khi", "như", "đã", "sẽ", "một", "những", "tại",
                  "từ", "hay", "hoặc", "thì", "nào", "gì", "sao", "không", "phải", "trên",
                  "dưới", "đó"}


def _v11_tokenize(text):
    return [t for t in _V11_TOKEN_RE.findall(text.lower())
            if t not in _V11_STOPWORDS and len(t) > 1]


def trim_body_by_khoan(body, question, window=TRIM_KHOAN_WINDOW):
    """v11: body >=3 khoản đánh số -> chỉ giữ khoản overlap từ vựng cao nhất với câu hỏi
    +/- window. Không có tín hiệu (câu hỏi rỗng sau lọc, overlap=0 mọi khoản) thì giữ nguyên."""
    matches = list(_V11_KHOAN_RE.finditer(body))
    if len(matches) < 3:
        return body
    head = body[:matches[0].start()].strip()
    parts = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        parts.append(body[m.start():end].strip())
    q_tokens = set(_v11_tokenize(question))
    if not q_tokens:
        return body
    scores = [len(q_tokens & set(_v11_tokenize(text))) for text in parts]
    if max(scores) == 0:
        return body
    best_idx = scores.index(max(scores))
    lo, hi = max(0, best_idx - window), min(len(parts), best_idx + window + 1)
    kept = "\n".join(parts[lo:hi])
    return f"{head}\n{kept}" if head else kept
CONCL = "echo2"
_TITLE_MAX_WORDS = 30


def _dieu_title(text: str) -> str:
    body = _DIEU_PREFIX_STRIP_RE.sub("", text, count=1)
    first = body.split("\n", 1)[0].strip().rstrip(".:;").strip()
    if not first or len(first.split()) > _TITLE_MAX_WORDS:
        return ""
    return first[0].lower() + first[1:]


def _dieu_lead(ref: str, text: str, template: str) -> str:
    if template == "T0_current":
        return f"Căn cứ {ref} quy định như sau:"
    title = _dieu_title(text) if template in ("T2_title", "T3_theo_qd_tai_title") else ""
    if template == "T1_theo_qd_tai":
        return f"Căn cứ theo quy định tại {ref} như sau:"
    if template == "T2_title":
        return f"Căn cứ {ref} quy định về {title} như sau:" if title else f"Căn cứ {ref} quy định như sau:"
    if template == "T3_theo_qd_tai_title":
        return (f"Căn cứ theo quy định tại {ref} quy định về {title} như sau:" if title
                else f"Căn cứ theo quy định tại {ref} như sau:")
    raise ValueError(f"template không hợp lệ: {template}")


def render_answer(selected_chunks: list, top_n: int, question: str = "",
                  concl: str = CONCL, template: str = ANSWER_TEMPLATE,
                  trim_khoan: bool = False) -> str:
    parts, seen = [], set()
    for c in selected_chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        loai_vb = c["loai_vb"] or "văn bản"
        so_hieu = c["so_hieu"] or ""
        dieu = c["dieu_so"]
        unit_type = c.get("unit_type", "dieu" if dieu != "0" else "raw")
        unit_no = c.get("unit_no", "")
        base_unit_type = unit_type[:-len("_capped")] if unit_type.endswith("_capped") else unit_type
        if base_unit_type == "dieu":
            lead = _dieu_lead(f"Điều {dieu} {loai_vb} {so_hieu}", c["text"], template)
            body = _DIEU_PREFIX_STRIP_RE.sub("", c["text"], count=1)
            if trim_khoan:  # v11: chỉ bật ở câu trả lời sản xuất cuối
                body = trim_body_by_khoan(body, question)
        elif base_unit_type == "muc" and unit_no:
            lead = f"Căn cứ Mục {unit_no} {loai_vb} {so_hieu} quy định như sau:"
            body = _MUC_PREFIX_STRIP_RE.sub("", c["text"], count=1)
        elif base_unit_type == "phu_luc" and unit_no:
            lead = f"Căn cứ Phụ lục {unit_no} {loai_vb} {so_hieu} quy định như sau:"
            body = _PHU_LUC_PREFIX_STRIP_RE.sub("", c["text"], count=1)
        elif base_unit_type == "tiet" and unit_no:
            lead = f"Căn cứ tiết {unit_no} {loai_vb} {so_hieu} quy định như sau:"
            body = _TIET_PREFIX_STRIP_RE.sub("", c["text"], count=1)
        else:
            lead = f"Căn cứ {loai_vb} {so_hieu} quy định như sau:"
            body = c["text"]
        parts.append(f"{lead}\n{body}")
    ans = "\n\n".join(parts)
    if concl != "none" and question:
        q = question.strip().rstrip("?").strip()
        if q:
            ql = q[0].lower() + q[1:]
            if concl == "echo":
                ans += f"\nNhư vậy, theo quy định nêu trên thì {ql}."
            elif concl == "echo2":
                ans += f"\nTheo đó, {ql}.\nNhư vậy, theo quy định nêu trên thì {ql}."
    return ans


def unit_ref(c):
    """Nhãn đơn vị dùng trong prompt LLM: 'Điều 17 Luật Đất đai 2013'."""
    loai, so = c.get("loai_vb") or "văn bản", c.get("so_hieu") or ""
    ut = c.get("unit_type", "dieu")
    ut = ut[:-len("_capped")] if ut.endswith("_capped") else ut
    names = {"dieu": "Điều", "muc": "Mục", "phu_luc": "Phụ lục", "tiet": "tiết"}
    no = c.get("dieu_so") if ut == "dieu" else c.get("unit_no", "")
    return (f"{names[ut]} {no} {loai} {so}" if ut in names and no else f"{loai} {so}").strip()


def _words(text, n):
    w = str(text).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def article_text(c, max_words=700):
    return _words(c["text"], max_words)