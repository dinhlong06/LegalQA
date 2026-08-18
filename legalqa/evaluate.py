"""Đo METEOR/ROUGE-L offline theo ĐÚNG cách scoring/legalqa/scoring.py tính: tokenize bằng
str.split() thô (không tách từ tiếng Việt), nltk meteor_score mặc định (alpha=0.9, nặng recall),
rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False) — dùng chính bản vendor trong
scoring/legalqa/rouge_score/ để khỏi lệch version pip. Xem scoring/SCORING_LegalQA.md: ROUGE-L
bị hỏng với tiếng Việt (chỉ để tham khảo), METEOR mới là độ đo quyết định.

Cần package `nltk` (tải wordnet/omw-1.4 nếu chưa có) — chỉ dùng lúc dev-eval, KHÔNG cần lúc
predict/đóng gói submission (rag_model/legalqa/predict.py, package.py không import module này).
"""
from __future__ import annotations

import sys
from pathlib import Path

from rag_model.legalqa.render import render_answer

REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_ROUGE_DIR = REPO_ROOT / "scoring" / "legalqa"

_scorers_cache = {}


def _load_scorers():
    """Import rouge_score từ bản vendor của BTC (scoring/legalqa/rouge_score/) thay vì gói pip,
    để hành vi khớp chính xác với scorer thật dùng lúc chấm."""
    if "rouge" in _scorers_cache:
        return _scorers_cache["rouge"], _scorers_cache["meteor"]

    import nltk
    if str(_VENDOR_ROUGE_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_ROUGE_DIR))
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    _scorers_cache["rouge"] = rouge
    _scorers_cache["meteor"] = meteor_score
    return rouge, meteor_score


def score_one(rouge, meteor_fn, ref: str, pred: str) -> tuple:
    """-> (meteor, rougeL), đúng công thức eval_qa() trong scoring/legalqa/scoring.py."""
    r = rouge.score(str(ref), str(pred))["rougeL"].fmeasure
    m = meteor_fn([str(ref).split()], str(pred).split())
    return m, r


def sweep(dev_qids: list, train_data: dict, pool_by_qid: dict, reranker, doc_meta: dict,
          top_k_final_options=(5, 10, 15), top_n_options=(1, 3, 5, 7)) -> list:
    """pool_by_qid: candidate pool CHƯA rerank của dev_qids (từ
    rag_model/retrieve/candidate.py:CandidateRetriever.search_batch() — pha 1, đã chạy 1 lần
    trước khi gọi sweep(), KHÔNG chạy lại ở đây). reranker: đã nạp sẵn
    (rag_model/retrieve/rerank.py:ChunkReranker — pha 2). Với mỗi top_k_final, rerank_batch()
    chạy 1 lệnh cho cả dev_qids (rẻ, không phải encode lại câu hỏi), rồi thử nhiều top_n bằng
    cách CẮT list đã xếp hạng qua render_answer (rẻ hơn nữa).
    -> list dict {top_k_final, top_n, meteor, rougeL, n}, sắp giảm dần theo meteor."""
    rouge, meteor_fn = _load_scorers()
    results = []
    for top_k_final in top_k_final_options:
        print(f"  --- top_k_final={top_k_final} (rerank cho {len(dev_qids)} câu) ---")
        ranked_cache = reranker.rerank_batch(pool_by_qid, top_k_final=top_k_final)
        for top_n in top_n_options:
            if top_n > top_k_final:
                continue
            ms, rs = [], []
            for qid in dev_qids:
                pred = render_answer(ranked_cache[qid], doc_meta, top_n)
                ref = train_data[qid]["answer"]
                m, r = score_one(rouge, meteor_fn, ref, pred)
                ms.append(m)
                rs.append(r)
            n = len(dev_qids)
            entry = {"top_k_final": top_k_final, "top_n": top_n,
                      "meteor": sum(ms) / n, "rougeL": sum(rs) / n, "n": n}
            print(f"    top_n={top_n:<3} METEOR={entry['meteor']:.4f}  ROUGE-L={entry['rougeL']:.4f}")
            results.append(entry)
    results.sort(key=lambda r: r["meteor"], reverse=True)
    return results
