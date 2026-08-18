"""Pha 2: chỉ nạp FlagReranker (Vietnamese_Reranker) — không BM25/dense/encoder. Nhận
candidate pool đã tìm sẵn ở pha 1 (rag_model/retrieve/candidate.py:CandidateRetriever),
rerank cho NHIỀU câu hỏi cùng lúc bằng 1 lệnh compute_score() duy nhất (nhanh hơn hẳn so
với gọi lặp từng câu — theo đúng cách retrieval/run_pipeline_fast.py:run_step2() làm).
"""
from rag_model.retrieve.component.gpu_reserve import reserve_vram


class ChunkReranker:
    """Nạp reranker một lần, gọi .rerank_batch(pool_by_qid) cho nhiều câu hỏi."""

    def __init__(self, reranker: str = "AITeamVN/Vietnamese_Reranker", reserve_gb: float = 0.0):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(reranker, use_fp16=True)
        if reserve_gb > 0:
            reserve_vram(reserve_gb)

    def rerank_batch(self, pool_by_qid: dict, top_k_final: int = 5, batch_size: int = 32,
                      max_length: int = 512) -> dict:
        """pool_by_qid: {qid: {"question": str, "candidates": [{"document_id","chunk_id",
        "text"}, ...]}} (từ CandidateRetriever.search_batch())
        -> {qid: [{"document_id","chunk_id","score","text"}, ...]} — đã max-agg theo
        document_id, cắt top_k_final, sắp giảm dần theo score — cùng format
        LegalRetriever.retrieve() cũ trả về, để rag_model/legalqa/render.py dùng thẳng
        không phải đổi gì."""
        qids = [qid for qid, entry in pool_by_qid.items() if entry["candidates"]]
        flat, spans = [], []
        for qid in qids:
            entry = pool_by_qid[qid]
            start = len(flat)
            flat.extend([[entry["question"], c["text"]] for c in entry["candidates"]])
            spans.append((start, len(flat)))

        result = {qid: [] for qid in pool_by_qid}
        if not flat:
            return result

        scores = self.model.compute_score(flat, batch_size=batch_size, max_length=max_length,
                                          normalize=True)
        if isinstance(scores, float):
            scores = [scores]

        for qid, (start, end) in zip(qids, spans):
            candidates = pool_by_qid[qid]["candidates"]
            sub_scores = scores[start:end]
            best = {}
            for c, sc in zip(candidates, sub_scores):
                aid = c["document_id"]
                if aid not in best or sc > best[aid][0]:
                    best[aid] = (sc, c)
            ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k_final]
            result[qid] = [{"document_id": aid, "chunk_id": c["chunk_id"], "score": float(sc),
                            "text": c["text"]} for aid, (sc, c) in ranked]
        return result
