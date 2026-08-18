"""Bản trích từ retrieval/encoder.py — CHỈ 2 encoder dùng ở rag_model/retrieve/__init__.py.

Trích riêng (không import package `retrieval`) để rag_model/ có thể mang đi nộp
độc lập, không kéo theo phần Task 1 (search.py, run_pipeline*.py, prepare_data/...).
Nếu retrieval/encoder.py đổi, phải đồng bộ tay lại 2 class này. Cài phụ thuộc qua
rag_model/requirements.txt.
"""
from typing import List, Tuple, Union

import torch
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEM3Encoder:
    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.use_fp16 = use_fp16

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        is_batch = isinstance(query_text, list)
        emb = self.model.encode(query_text, return_dense=True, return_sparse=True,
                                return_colbert_vecs=False)
        emb_sparse = emb["lexical_weights"]
        dense_vec = emb["dense_vecs"]

        if not is_batch:
            indices = list(emb_sparse.keys())
            values = list(emb_sparse.values())
            return dense_vec, indices, values
        dense_out, indices_out, values_out = [], [], []
        for dv, sparse_vec in zip(dense_vec, emb_sparse):
            dense_out.append(dv)
            indices_out.append(list(sparse_vec.keys()))
            values_out.append(list(sparse_vec.values()))
        return dense_out, indices_out, values_out


class E5Encoder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device=_pick_device())

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        if not isinstance(query_text, str):
            query_text = [f"{type}: " + text for text in query_text]
        else:
            query_text = f"{type}: " + query_text
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec, [], []
