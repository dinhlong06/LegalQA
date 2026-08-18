"""Bản trích từ retrieval/fast_index.py — CHỈ phần INFERENCE (đọc index có sẵn).

Bỏ build_bm25_csr/extract_dense (chỉ dùng lúc DỰNG index, kéo theo pickle alias
ViDRILL + qdrant_client — không cần lúc suy luận). Nếu cần dựng lại index từ đầu,
quay về retrieval/fast_index.py bản đầy đủ, không sửa ở đây.
"""
import json
import os

import numpy as np
import scipy.sparse as sp


def _top_k_indices(scores, top_k: int):
    """Top-k chỉ số, trùng khít sorted(range(n), key=..., reverse=True)[:top_k]."""
    scores = np.asarray(scores)
    n = scores.shape[0]
    k = min(top_k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.lexsort((np.arange(n), -scores))

    thr = scores[np.argpartition(-scores, k - 1)[:k]].min()
    strictly = np.flatnonzero(scores > thr)
    ties = np.flatnonzero(scores == thr)
    sel = np.concatenate([strictly, ties[:k - strictly.shape[0]]])
    return sel[np.lexsort((sel, -scores[sel]))]


class BM25Csr:
    """BM25Okapi tương đương, chạy bằng nhân ma trận thưa. Đọc index đã build sẵn."""

    def __init__(self, index_dir: str, chunk_to_aid: dict = None):
        self.W = sp.load_npz(os.path.join(index_dir, "bm25_W.npz")).tocsc()
        self.idf = np.load(os.path.join(index_dir, "bm25_idf.npy"))
        with open(os.path.join(index_dir, "bm25_vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(os.path.join(index_dir, "bm25_ids.json"), encoding="utf-8") as f:
            self.chunk_ids = json.load(f)
        self.chunk_to_aid = chunk_to_aid

    def get_scores(self, tokens):
        score = np.zeros(self.W.shape[0], dtype=np.float64)
        for t in tokens:
            j = self.vocab.get(t)
            if j is None:
                continue
            col = self.W.getcol(j)
            score[col.indices] += self.idf[j] * col.data
        return score

    def search(self, query: str, top_k: int = 5):
        scores = self.get_scores(query.split())
        idx = _top_k_indices(scores, top_k)
        return [(float(scores[i]), self.chunk_ids[i]) for i in idx]


class DenseMatrix:
    """Tìm top-k bằng một phép nhân ma trận, trên GPU nếu có."""

    def __init__(self, prefix: str, device: str = "cuda"):
        import torch
        self.torch = torch
        mat = np.load(prefix + ".npy")
        with open(prefix + ".ids.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunk_ids = meta["chunk_ids"]
        self.aids = meta["aids"]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mat = torch.from_numpy(mat).to(self.device)
        if self.device == "cpu":
            self.mat = self.mat.float()

    def search_batch(self, qvecs: np.ndarray, top_k: int, block: int = 512):
        torch = self.torch
        qs = np.asarray(qvecs, dtype=np.float16)
        k = min(top_k, self.mat.shape[0])
        out_v, out_i = [], []
        for s in range(0, len(qs), block):
            q = torch.from_numpy(qs[s:s + block]).to(self.device)
            if self.device == "cpu":
                q = q.float()
            scores = q @ self.mat.T
            vals, idx = torch.topk(scores.float(), k=k, dim=1)
            out_v.append(vals.cpu().numpy())
            out_i.append(idx.cpu().numpy())
            del q, scores, vals, idx
        return np.concatenate(out_v), np.concatenate(out_i)
