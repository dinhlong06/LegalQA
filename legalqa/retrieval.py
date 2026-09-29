"""Retrieval 2 tầng: RRF 3 kênh + gộp điểm CE chọn văn bản + rerank Điều (Cell 11 + 11b).

Cấu hình v7 đã đóng băng là "base": trọng số RRF đều 1.0, không kb/prf/rw — nên ở đây chỉ
giữ đường `base`, tương đương `retrieve_two_tier()` của notebook khi mọi tham số mở rộng
bỏ trống. `rrf_k = 60` giữ đúng hằng số công thức RRF gốc.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .chunking import tokenize_simple
from .rerank import rerank

RRF_K = 60


def rrf_rank_lists(question: str, bm25, dense_channels, top_k: int = 100, q_embs=None):
    """-> [xếp hạng BM25, xếp hạng dense A, xếp hạng dense B] (mỗi phần tử là list index).

    `q_embs` (tuỳ chọn): vector query đã encode sẵn cho từng kênh — dùng khi encoder đã được
    đẩy sang CPU để tiết kiệm VRAM. Bỏ trống thì encode tại chỗ (đúng như notebook).
    """
    lists = [list(bm25.top_k(tokenize_simple(question), top_k))]
    for i, ch in enumerate(dense_channels):
        if q_embs is not None:
            q_emb = q_embs[i]
        else:
            q_text = ch["query_prefix"] + question if ch["query_prefix"] else question
            q_emb = ch["model"].encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = ch["embeddings"] @ q_emb
        lists.append(list(np.argsort(-scores)[:top_k]))
    return lists


def rrf_idx(rank_lists, weights, top_k: int = 100):
    """Trộn RRF theo trọng số từng kênh, trả list index đã sort giảm dần."""
    rank_maps = [{idx: r for r, idx in enumerate(rl)} for rl in rank_lists]
    all_idx = set(rank_lists[0])
    for rl in rank_lists[1:]:
        all_idx |= set(rl)
    rrf = {i: sum(w / (RRF_K + rm.get(i, top_k + 1)) for rm, w in zip(rank_maps, weights))
           for i in all_idx}
    return sorted(rrf, key=rrf.get, reverse=True)


def _score_docs(candidates: list, scores, mode: str = "max", T: float = 1.0) -> dict:
    """{doc_id: điểm gộp}. v7 dùng mode="max" cho tầng 1 (LSE đã đo âm, đóng hướng)."""
    by_doc = {}
    for c, sc in zip(candidates, scores):
        by_doc.setdefault(str(c["id"]).split("_")[0], []).append(float(sc))
    if mode == "max":
        return {d: max(v) for d, v in by_doc.items()}
    doc_score = {}
    for d, v in by_doc.items():
        arr = np.array(v, dtype=np.float64) / max(T, 1e-6)
        doc_score[d] = float(T * (arr.max() + np.log(np.exp(arr - arr.max()).sum())))
    return doc_score


def retrieve_two_tier(question: str, bm25_t1, dense_channels, all_chunks_t1, dieu_by_doc,
                      reranker_model=None, reranker_tokenizer=None, doc_k: int = 5,
                      memo_t1: dict = None, memo_t2: dict = None,
                      top_k_retrieve: int = 100, max_dieu_candidates: int = 100, q_embs=None):
    """Hai tầng (result.md §3): tầng 1 chọn doc_k VĂN BẢN trên corpus 450 từ, tầng 2 rerank Điều.

    Trả (danh sách Điều đã CE-sort, điểm CE) — điểm None nếu không có reranker.
    """
    rank_lists = rrf_rank_lists(question, bm25_t1, dense_channels, top_k_retrieve, q_embs=q_embs)
    t1_ranked = [all_chunks_t1[i] for i in rrf_idx(rank_lists, [1.0] * len(rank_lists),
                                                   top_k_retrieve)]
    if not t1_ranked:
        return [], None
    t1_scores = None
    if reranker_model is not None:
        # Notebook Cell 11/11b: rerank() mặc định max_candidates=TOP_K_RETRIEVE=100, tức tầng 1
        # chỉ CE-chấm top-100 RRF (không phải toàn bộ hợp 3 kênh). Giữ đúng con số này.
        t1_ranked, t1_scores = rerank(question, t1_ranked, reranker_model, reranker_tokenizer,
                                      max_candidates=top_k_retrieve, memo=memo_t1)
    if t1_scores is None:
        top_docs = list(dict.fromkeys(str(c["id"]).split("_")[0] for c in t1_ranked))[:doc_k]
    else:
        doc_score = _score_docs(t1_ranked, t1_scores, "max", 1.0)
        top_docs = sorted(doc_score, key=doc_score.get, reverse=True)[:doc_k]

    cand = []
    for d in top_docs:
        cand.extend(dieu_by_doc.get(d, []))
    if len(cand) > max_dieu_candidates:
        qt = set(tokenize_simple(question))
        cand = sorted(cand, key=lambda c: -len(qt & set(tokenize_simple(c["text"]))))[:max_dieu_candidates]
    if not cand:
        return [], None
    if reranker_model is None:
        return cand, None
    return rerank(question, cand, reranker_model, reranker_tokenizer,
                  max_candidates=len(cand), memo=memo_t2)


def split_evenly(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def parallel_process(ids, worker_fn, devices, label: str = ""):
    """Chia `ids` cho các GPU, chạy worker_fn(chunk, device, worker_idx) song song."""
    import time
    ids = list(ids)
    devices = list(devices) if devices else ["cpu"]
    chunks = split_evenly(ids, len(devices))
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futures = [ex.submit(worker_fn, chunk, dev, i)
                   for i, (chunk, dev) in enumerate(zip(chunks, devices))]
        for f in futures:
            results.update(f.result())
    print(f"    [{label}] {len(ids)} câu / {len(devices)} thiết bị -> {time.time()-t0:.0f}s")
    return results