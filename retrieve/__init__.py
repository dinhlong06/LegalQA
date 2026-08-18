"""Retrieval cho rag_model — self-contained, không import gì từ package `retrieval`.

Dùng khi cần nộp riêng code (rag_model/) mà không mang theo toàn bộ retrieval/
(search.py, run_pipeline*.py, prepare_data/...). Mọi phần cần thiết đã trích vào
rag_model/retrieve/component/. Vẫn cần data index đã build sẵn ở retrieval/db_fast/
(dữ liệu, không phải code) — trỏ --index_dir sang chỗ khác nếu copy toàn bộ
rag_model/ sang máy/repo khác.

    python -m rag_model.retrieve "Câu hỏi pháp luật ví dụ?"
"""
import gzip
import json
import os

import numpy as np
import scipy.sparse as sp

from rag_model.retrieve.component.fast_index import BM25Csr, DenseMatrix
from rag_model.retrieve.component.encoder import BGEM3Encoder, E5Encoder


class LegalRetriever:
    """Nạp index một lần, gọi .retrieve(question) nhiều lần."""

    def __init__(self, index_dir="retrieval/db_fast",
                 text_source="retrieval/db_fast/texts.json.gz",
                 bge_model="BAAI/bge-m3",
                 e5_model="intfloat/multilingual-e5-large",
                 reranker="AITeamVN/Vietnamese_Reranker"):
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

        from FlagEmbedding import FlagReranker
        self.reranker = FlagReranker(reranker, use_fp16=True)

    def _row(self, cid, score):
        aid = self.aid_of.get(cid)
        return None if aid is None else (cid, str(aid), float(score))

    def retrieve(self, question: str, top_k_bm25=50, top_k_dense=30, top_k_final=5):
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

        rows = {}
        for sc, cid in self.bm25.search(question, top_k=top_k_bm25):
            r = self._row(cid, sc)
            if r:
                rows[r[0]] = r
        for v, i in zip(bge_val[0], bge_idx[0]):
            r = self._row(self.bge_dm.chunk_ids[i], v)
            if r:
                rows[r[0]] = r
        for v, i in zip(e5_val[0], e5_idx[0]):
            r = self._row(self.e5_dm.chunk_ids[i], v)
            if r:
                rows[r[0]] = r
        for i in top_sp:
            if sparse_scores[i] <= 0:
                continue
            r = self._row(self.sp_meta["chunk_ids"][i], sparse_scores[i])
            if r:
                rows[r[0]] = r

        pool = [(cid, aid) for cid, aid, _ in rows.values() if cid in self.texts]
        if not pool:
            return []

        pairs = [[question, self.texts[cid]] for cid, _ in pool]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]

        best = {}
        for (cid, aid), sc in zip(pool, scores):
            if aid not in best or sc > best[aid][0]:
                best[aid] = (sc, cid)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k_final]
        return [{"document_id": aid, "chunk_id": cid, "score": float(sc),
                 "text": self.texts[cid]} for aid, (sc, cid) in ranked]
