"""Pha 1: nạp BM25 + sparse matrix + 2 dense encoder (bge-m3, e5-large), tìm candidate pool
cho NHIỀU câu hỏi cùng lúc — KHÔNG nạp reranker. Tách riêng khỏi rag_model/retrieve/__init__.py
(LegalRetriever, vẫn giữ cho debug nhanh 1 câu) để chạy hàng loạt câu hỏi không phải giữ cả
2 encoder + reranker trên GPU cùng lúc — đỉnh VRAM cần chỉ bằng pha đang chạy. Dùng cùng
rag_model/retrieve/rerank.py (pha 2, chỉ nạp reranker) qua rag_model/retrieve/cache_io.py.
"""
import gzip
import json
import os

import numpy as np
import scipy.sparse as sp

from rag_model.retrieve.component.fast_index import BM25Csr, DenseMatrix
from rag_model.retrieve.component.encoder import BGEM3Encoder, E5Encoder
from rag_model.retrieve.component.gpu_reserve import reserve_vram


class CandidateRetriever:
    """Nạp index + 2 encoder một lần, gọi .search_batch(questions) cho nhiều câu hỏi."""

    def __init__(self, index_dir="retrieval/db_fast",
                 text_source="retrieval/db_fast/texts.json.gz",
                 bge_model="BAAI/bge-m3",
                 e5_model="intfloat/multilingual-e5-large",
                 reserve_gb: float = 0.0):
        self.bm25 = BM25Csr(index_dir)
        self.sparse_mat = sp.load_npz(os.path.join(index_dir, "bge_sparse.npz")).T.tocsr()
        with gzip.open(os.path.join(index_dir, "bge_sparse.npz.ids.json.gz"),
                       "rt", encoding="utf-8") as f:
            self.sp_meta = json.load(f)

        self.bge_dm = DenseMatrix(os.path.join(index_dir, "bge_dense"))
        self.e5_dm = DenseMatrix(os.path.join(index_dir, "e5_dense"))
        self.aid_of = dict(zip(self.bge_dm.chunk_ids, self.bge_dm.aids))

        with gzip.open(text_source, "rt", encoding="utf-8") as f:
            self.texts = json.load(f)

        self.bge_enc = BGEM3Encoder(model_name=bge_model, use_fp16=True)
        self.e5_enc = E5Encoder(model_name=e5_model)

        # Giữ chỗ VRAM NGAY SAU KHI 2 encoder đã lên GPU (gọi trước sẽ bị from_pretrained
        # tự xoá sạch qua accelerate — xem docstring reserve_vram()).
        if reserve_gb > 0:
            reserve_vram(reserve_gb)

    def _row(self, cid):
        aid = self.aid_of.get(cid)
        if aid is None or cid not in self.texts:
            return None
        return {"document_id": str(aid), "chunk_id": cid, "text": self.texts[cid]}

    def search_one(self, question: str, top_k_bm25: int = 50, top_k_dense: int = 30) -> list:
        """Hợp 4 kênh (BM25 + bge dense + e5 dense + bge sparse) — đúng thuật toán
        LegalRetriever.retrieve() cũ, chỉ bỏ điểm số thô (nguyên bản cũng discard điểm này
        trước khi đưa qua reranker, không đổi kết quả cuối)."""
        bge_dense_q, sp_idx, sp_val = self.bge_enc.encode(question, type="query")
        e5_q, _, _ = self.e5_enc.encode(question, type="query")

        bge_val, bge_idx = self.bge_dm.search_batch(np.asarray([bge_dense_q]), top_k_dense)
        e5_val, e5_idx = self.e5_dm.search_batch(np.asarray([e5_q]), top_k_dense)

        vocab_n = self.sparse_mat.shape[0]
        cols = [int(a) for a in sp_idx if int(a) < vocab_n]
        vals = [float(v) for a, v in zip(sp_idx, sp_val) if int(a) < vocab_n]
        qvec = sp.csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, vocab_n))
        sparse_scores = (qvec @ self.sparse_mat).toarray()[0]
        k = min(top_k_dense, sparse_scores.size)
        top_sp = np.argpartition(-sparse_scores, k - 1)[:k]
        top_sp = top_sp[np.argsort(-sparse_scores[top_sp])]

        cids = {}
        for _sc, cid in self.bm25.search(question, top_k=top_k_bm25):
            cids[cid] = None
        for i in bge_idx[0]:
            cids[self.bge_dm.chunk_ids[i]] = None
        for i in e5_idx[0]:
            cids[self.e5_dm.chunk_ids[i]] = None
        for i in top_sp:
            if sparse_scores[i] <= 0:
                continue
            cids[self.sp_meta["chunk_ids"][i]] = None

        pool = []
        for cid in cids:
            r = self._row(cid)
            if r:
                pool.append(r)
        return pool

    def search_batch(self, questions: dict, top_k_bm25: int = 50, top_k_dense: int = 30,
                      log_every: int = 100) -> dict:
        """questions: {qid: question_text}
        -> {qid: {"question": str, "candidates": [{"document_id","chunk_id","text"}, ...]}}
        (CHƯA rerank — xem rag_model/retrieve/rerank.py:ChunkReranker.rerank_batch())."""
        out = {}
        n = len(questions)
        for i, (qid, q) in enumerate(questions.items()):
            out[qid] = {"question": q, "candidates": self.search_one(q, top_k_bm25, top_k_dense)}
            if (i + 1) % log_every == 0 or (i + 1) == n:
                print(f"  candidate search: {i + 1}/{n}")
        return out
