"""Nạp hai dense encoder fine-tune (A-ft/B-ft) và encode corpus (Cell 8 + Cell 9).

A-ft = checkpoint fine-tune của `AITeamVN/Vietnamese_Embedding_v2` (pooling CLS),
B-ft = checkpoint fine-tune của `mainguyen9/vietlegal-harrier-0.6b` (pooling last-token).
Cả hai KHÔNG dùng tiền tố query:/passage:.

Encode corpus dùng fp16 CÓ CỔNG: chỉ dùng fp16 khi không NaN và cosine nhỏ nhất so với fp32
trên mẫu >= FP16_MIN_COS; trượt thì tự lùi fp32.
"""
import random
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import (
    DENSE_MAX_SEQ_LEN, ENCODE_BATCH_SIZE, ENCODE_FP16, FP16_GUARD_N, FP16_MIN_COS, SEED,
)


def load_encoder(path: str, device: str, fp16: bool = True) -> SentenceTransformer:
    """Nạp encoder. Trên CUDA mặc định `.half()` — đúng trạng thái model sau cổng fp16 ở
    Cell 9 notebook (corpus encode fp16, và query encode SAU ĐÓ cũng bằng model đã half()).
    Encode query fp32 trong khi corpus là fp16 sẽ lệch hệ toạ độ và đổi thứ hạng retrieval."""
    model = SentenceTransformer(path, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN
    if fp16 and str(device).startswith("cuda"):
        model = model.half()
    return model


def build_channels(model_a, model_b, cfg=None) -> list:
    cfg = cfg or {}
    return [
        {"name": "A", "model": model_a, "embeddings": None,
         "query_prefix": cfg.get("QUERY_PREFIX_A", ""), "passage_prefix": ""},
        {"name": "B", "model": model_b, "embeddings": None,
         "query_prefix": cfg.get("QUERY_PREFIX_B", ""),
         "passage_prefix": cfg.get("PASSAGE_PREFIX_B", "")},
    ]


def _fp16_guard(model, sample_texts, device):
    dtype0 = next(model.parameters()).dtype
    if dtype0 == torch.float16:
        return True, {"note": "checkpoint đã là fp16 sẵn"}
    model.to(device)
    ref = model.encode(sample_texts, batch_size=32, convert_to_numpy=True,
                       normalize_embeddings=True, device=device)
    model.half()
    try:
        emb = model.encode(sample_texts, batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True, device=device)
    except Exception as e:
        model.float()
        return False, {"error": f"{type(e).__name__}: {e}"}
    finite = bool(np.isfinite(emb).all())
    cos = (ref.astype(np.float64) * emb.astype(np.float64)).sum(1) if finite else np.array([0.0])
    info = {"finite": finite, "min_cos": round(float(cos.min()), 5),
            "mean_cos": round(float(cos.mean()), 5), "n_sample": len(sample_texts)}
    ok = finite and float(cos.min()) >= FP16_MIN_COS
    if not ok:
        model.float()
    return ok, info


def encode_corpus(channels: list, texts_raw: list, device: str, encode_batch: int = ENCODE_BATCH_SIZE):
    """Encode toàn bộ chunk tầng 1 cho từng kênh; điền `embeddings` (fp32 numpy) vào channel."""
    rng = random.Random(SEED)
    by_len = sorted(range(len(texts_raw)), key=lambda i: -len(texts_raw[i]))
    n_guard = min(FP16_GUARD_N, len(texts_raw))
    guard_idx = by_len[: n_guard // 2] + rng.sample(range(len(texts_raw)), n_guard - n_guard // 2)
    info = {}
    for ch in channels:
        t0 = time.time()
        texts = [ch["passage_prefix"] + t for t in texts_raw] if ch["passage_prefix"] else texts_raw
        model = ch["model"]
        use_fp16, ginfo = False, {"skipped": "ENCODE_FP16=False hoặc không có CUDA"}
        if ENCODE_FP16 and device.startswith("cuda"):
            use_fp16, ginfo = _fp16_guard(model, [texts[i] for i in guard_idx], device)
        info[ch["name"]] = {"fp16": use_fp16, **ginfo}
        print(f"  [{ch['name']}] cổng fp16: {'QUA' if use_fp16 else 'TRƯỢT -> fp32'} {ginfo}", flush=True)
        model = model.to(device)
        if device.startswith("cuda") and use_fp16:
            model = model.half()
        bs = encode_batch
        while True:
            try:
                emb = model.encode(texts, batch_size=bs, convert_to_numpy=True,
                                   show_progress_bar=True, normalize_embeddings=True, device=device)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    print(f"    [CUDA OOM] batch={bs} -> {bs // 2}", flush=True)
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    continue
                raise
        if not np.isfinite(emb).all():
            raise SystemExit(f"[{ch['name']}] embedding corpus có NaN/inf — dừng.")
        ch["embeddings"] = emb.astype(np.float32)
        info[ch["name"]]["encode_min"] = round((time.time() - t0) / 60, 1)
        print(f"    -> {emb.shape} trong {time.time()-t0:.0f}s", flush=True)
    return info


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())