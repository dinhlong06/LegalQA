"""Ghép câu trả lời template extractive: lead-in cấp văn bản
("Theo <loại văn bản> <số hiệu> quy định cụ thể:") + nguyên văn top_n chunk đã xếp hạng cao nhất.

Không dùng LLM sinh câu — khớp với METEOR alpha=0,9 (nặng recall, xem scoring/SCORING_LegalQA.md):
bám sát nguyên văn điều luật lợi hơn diễn giải lại.
"""
from __future__ import annotations

FALLBACK_ANSWER = "Không tìm thấy thông tin pháp lý cho câu hỏi này."


def render_answer(ranked_chunks: list, doc_meta: dict, top_n: int) -> str:
    """ranked_chunks: list các dict {"document_id","chunk_id","score","text"} đã sắp theo thứ
    hạng giảm dần — đúng định dạng LegalRetriever.retrieve() trả về (rag_model/retrieve/).
    LegalRetriever đã max-agg theo document_id nên mỗi document chỉ xuất hiện tối đa 1 lần;
    seen_docs bên dưới chỉ là phòng thủ, không đổi hành vi trong trường hợp bình thường.
    doc_meta: {document_id: {"loai_vb", "so_hieu"}} từ doc_meta.build_doc_meta().
    """
    parts, seen_docs = [], set()
    for c in ranked_chunks:
        if len(parts) >= top_n:
            break
        doc_id = c["document_id"]
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        meta = doc_meta.get(doc_id, {})
        loai_vb = meta.get("loai_vb") or "văn bản"
        so_hieu = meta.get("so_hieu") or ""
        lead = f"Theo {loai_vb} {so_hieu} quy định cụ thể:" if so_hieu else f"Theo {loai_vb} quy định cụ thể:"
        parts.append(f"{lead}\n{c['text']}")
    if not parts:
        return FALLBACK_ANSWER
    return "\n\n".join(parts)
