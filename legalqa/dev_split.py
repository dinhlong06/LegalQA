"""Cắt 1 tập dev CỐ ĐỊNH (seed cố định) từ train.json của LegalQA (7000 câu có answer thật) để
đo METEOR/ROUGE-L offline trước khi tốn lượt submit — public test chỉ có 3 lượt/ngày (rule.md),
và mọi cấu hình phải được so sánh trên cùng 1 tập, không được trôi nổi giữa các lần chạy
(CLAUDE.md §4 'Đánh giá': không dò tham số qua leaderboard).
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def build_dev_split(train_path, n: int = 300, seed: int = 42) -> list:
    with Path(train_path).open(encoding="utf-8") as f:
        train_data = json.load(f)
    qids = sorted(train_data.keys())
    rng = random.Random(seed)
    n = min(n, len(qids))
    return rng.sample(qids, n)


def load_or_build_dev_split(train_path, cache_path, n: int = 300, seed: int = 42) -> list:
    cache_path = Path(cache_path)
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)
    qids = build_dev_split(train_path, n=n, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(qids, f, ensure_ascii=False)
    return qids
