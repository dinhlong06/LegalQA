"""Đóng gói / nạp index bake sẵn cho pipeline inference.

Index gồm:
    chunks_t1.json.gz   chunk 450 từ (tầng 1) — thứ tự bất biến, dùng cho BM25 + dense
    chunks_t2.json.gz   chunk theo Điều/Mục/... (tầng 2) — ứng viên cuối
    dense/A_ft.npy      ma trận dense fp32 của A-ft (N × 1024)
    dense/B_ft.npy      ma trận dense fp32 của B-ft (N × 1024)
    manifest.json       metadata: số chunk, chiều, thời điểm, phiên bản encoder

BM25 KHÔNG bake thành object pickle — rebuild từ `chunks_t1` lúc nạp (vài phút, tránh phụ
thuộc phiên bản numpy/pickle giữa máy build và máy BTC).
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np


def save_index(out_dir, all_chunks, all_chunks_t1, channels, encode_info=None, extra=None):
    out = Path(out_dir)
    (out / "dense").mkdir(parents=True, exist_ok=True)

    def _dump(name, obj):
        with gzip.open(out / name, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    _dump("chunks_t1.json.gz", all_chunks_t1)
    _dump("chunks_t2.json.gz", all_chunks)

    for ch in channels:
        emb = np.asarray(ch["embeddings"], dtype=np.float32)
        np.save(out / "dense" / f"{ch['name']}_ft.npy", emb)

    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_chunks_t1": len(all_chunks_t1),
        "n_chunks_t2": len(all_chunks),
        "dense_dim": int(np.asarray(channels[0]["embeddings"]).shape[1]),
        "channels": [{"name": ch["name"], "shape": list(np.asarray(ch["embeddings"]).shape)}
                     for ch in channels],
        "encode_info": encode_info or {},
    }
    if extra:
        manifest.update(extra)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return manifest


def load_index(index_dir, devices, cfg=None):
    """-> (all_chunks, all_chunks_t1, dieu_by_doc, channels) — channels đã có embeddings.

    A-ft nạp lên thẻ ĐẦU, B-ft lên thẻ CUỐI (đúng Cell 8/9 notebook) — chia VRAM khi có ≥2 GPU.
    """
    devices = list(devices) if isinstance(devices, (list, tuple)) else [devices]
    dev_a, dev_b = devices[0], devices[-1]
    index_dir = Path(index_dir)
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Thiếu index tại {index_dir} — chạy scripts/build_index.py trước.")
    with gzip.open(index_dir / "chunks_t1.json.gz", "rt", encoding="utf-8") as f:
        all_chunks_t1 = json.load(f)
    with gzip.open(index_dir / "chunks_t2.json.gz", "rt", encoding="utf-8") as f:
        all_chunks = json.load(f)

    dieu_by_doc = {}
    for c in all_chunks:
        dieu_by_doc.setdefault(str(c["id"]).split("_")[0], []).append(c)

    from .encoders import build_channels, load_encoder
    from .config import resolve_path
    a_path = resolve_path(cfg or {}, "ENCODER_A_PATH", "models/A-ft")
    b_path = resolve_path(cfg or {}, "ENCODER_B_PATH", "models/B-ft")
    m_a = load_encoder(str(a_path), dev_a)
    m_b = load_encoder(str(b_path), dev_b)
    channels = build_channels(m_a, m_b, cfg)
    for ch in channels:
        emb = np.load(index_dir / "dense" / f"{ch['name']}_ft.npy")
        ch["embeddings"] = emb
    return all_chunks, all_chunks_t1, dieu_by_doc, channels