#!/usr/bin/env python
"""Dựng index bake sẵn: chunk corpus + encode bằng A-ft/B-ft.

    python scripts/build_index.py --corpus /data/selected-contexts \
        --a-ft models/A-ft --b-ft models/B-ft --out index --device cuda:0

BM25 KHÔNG bake (rebuild lúc nạp) — chỉ bake chunk + ma trận dense.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legalqa.chunking import load_corpus, unit_type_histogram  # noqa: E402
from legalqa.encoders import build_channels, encode_corpus, load_encoder  # noqa: E402
from legalqa.index import save_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--a-ft", default="models/A-ft")
    ap.add_argument("--b-ft", default="models/B-ft")
    ap.add_argument("--out", default="index")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== Nạp corpus {args.corpus} ===", flush=True)
    all_chunks, all_chunks_t1, _ = load_corpus(args.corpus)
    print(f"  {len(all_chunks)} chunk tầng 2 · {len(all_chunks_t1)} chunk tầng 1", flush=True)
    print(f"  unit_type: {unit_type_histogram(all_chunks)}", flush=True)

    print("=== Nạp encoder ===", flush=True)
    m_a = load_encoder(args.a_ft, args.device)
    m_b = load_encoder(args.b_ft, args.device)
    channels = build_channels(m_a, m_b)

    print(f"=== Encode corpus trên {args.device} ===", flush=True)
    texts = [c["text"] for c in all_chunks_t1]
    info = encode_corpus(channels, texts, args.device, encode_batch=args.batch)

    manifest = save_index(args.out, all_chunks, all_chunks_t1, channels, info,
                          extra={"corpus_dir": str(Path(args.corpus).resolve())})
    print(f"=== Xong trong {(time.time()-t0)/60:.1f} phút ===", flush=True)
    print(manifest, flush=True)


if __name__ == "__main__":
    main()