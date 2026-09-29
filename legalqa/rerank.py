"""Cross-encoder reranker zero-shot (Cell 10 + Cell 11 của notebook gốc).

v7 dùng BẢN GỐC `AITeamVN/Vietnamese_Reranker` (không fine-tune — v6_2 đo âm hai nửa dev).
Điểm CE được chấm theo lô nhỏ `RERANK_SUBBATCH`, có memo theo id để không chấm lại.
"""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def is_oom(err: Exception) -> bool:
    """Nhận diện lỗi hết VRAM — torch có nhiều cách báo: 'out of memory' (CUDA OOM) hoặc
    'CUBLAS_STATUS_ALLOC_FAILED'/'CUBLAS_STATUS_NOT_INITIALIZED' khi cấp handle thất bại."""
    s = str(err).lower()
    return ("out of memory" in s or "cublas_status_alloc_failed" in s
            or "cuda error" in s and "memory" in s)


def load_reranker(device: str, source: str, retries: int = 2):
    """Nạp cross-encoder lên `device`; trả (None, None) nếu thất bại (pipeline vẫn chạy RRF thuần)."""
    for attempt in range(retries):
        try:
            tok = AutoTokenizer.from_pretrained(source)
            mdl = AutoModelForSequenceClassification.from_pretrained(source)
            mdl = mdl.to(device)
            if device.startswith("cuda"):
                mdl = mdl.half()
            mdl.eval()
            return mdl, tok
        except Exception as e:
            if attempt == 0:
                import time
                print(f"  [reranker] lỗi lần 1: {e} — thử lại...")
                time.sleep(5)
                continue
            print(f"  [reranker] KHÔNG nạp được trên {device}: {e} -> bỏ qua reranker")
            return None, None


def rerank(question: str, candidates: list, model, tokenizer,
           max_candidates: int = None, max_length: int = 1024,
           sub_batch: int = 64, memo: dict = None):
    """Chấm `candidates[:max_candidates]` bằng CE, trả (danh sách đã sort, điểm).

    `memo` = {id: điểm} cho ĐÚNG model này + ĐÚNG câu hỏi này; id đã có điểm thì bỏ qua.
    Trả (candidates, None) khi không có model — giữ nguyên thứ hạng RRF.
    """
    if model is None or not candidates:
        return candidates, None
    subset = candidates if max_candidates is None else candidates[:max_candidates]
    scores = np.empty(len(subset), dtype=np.float32)
    todo = []
    for j, c in enumerate(subset):
        if memo is not None and c["id"] in memo:
            scores[j] = memo[c["id"]]
        else:
            todo.append(j)
    if todo:
        device = next(model.parameters()).device
        pairs = [[question, subset[j]["text"]] for j in todo]
        out_all = np.empty(len(pairs), dtype=np.float32)
        bs, i = max(1, sub_batch), 0
        while i < len(pairs):
            batch = pairs[i:i + bs]
            try:
                with torch.no_grad():
                    inputs = tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=max_length).to(device)
                    out = model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
                out_all[i:i + len(batch)] = out
                i += bs
            except RuntimeError as e:
                if is_oom(e) and bs > 1:
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    continue
                if is_oom(e):
                    return candidates, None
                raise
        for j, s in zip(todo, out_all):
            scores[j] = s
            if memo is not None:
                memo[subset[j]["id"]] = float(s)
    order = np.argsort(-scores)
    reranked = [subset[i2] for i2 in order]
    sorted_scores = scores[order]
    return reranked + candidates[len(subset):], sorted_scores