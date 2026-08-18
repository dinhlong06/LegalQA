"""Sinh câu trả lời template extractive cho toàn bộ câu hỏi trong public-official.json, dùng
CandidateRetriever + ChunkReranker (rag_model/retrieve/candidate.py, rerank.py — 2 pha, xem
docstring 2 file đó) làm nguồn context + render_answer (rag_model/legalqa/render.py) để ghép
câu trả lời."""
from __future__ import annotations

import json
from pathlib import Path

from rag_model.legalqa.render import render_answer


def load_questions(public_path) -> dict:
    with Path(public_path).open(encoding="utf-8") as f:
        return json.load(f)


def predict_all(ranked_by_qid: dict, questions: dict, doc_meta: dict, top_n: int) -> dict:
    """ranked_by_qid: kết quả ChunkReranker.rerank_batch() cho đúng tập qid trong `questions`
    (đã rerank xong ở pha 2) -> {qid: câu trả lời}."""
    return {qid: render_answer(ranked_by_qid[qid], doc_meta, top_n) for qid in questions}
