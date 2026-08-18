"""CLI điều phối Task 2 (LegalQA), chạy theo 2 PHA TÁCH RỜI để không bao giờ giữ đồng thời 2
encoder truy hồi (bge-m3, e5-large) VÀ reranker trên GPU cùng lúc — xem
rag_model/retrieve/candidate.py, rag_model/retrieve/rerank.py để biết lý do (đỡ OOM trên máy
dùng chung, quan sát thật khi giữ cả 3 model cùng lúc):

    Pha 1 (candidate): CandidateRetriever nạp BM25+sparse+2 encoder, tìm candidate pool cho
                       TẤT CẢ câu hỏi cần dùng (dev + public), rồi GIẢI PHÓNG.
    Pha 2 (rerank):    ChunkReranker chỉ nạp reranker, rerank candidate pool đã có sẵn —
                       dùng cho sweep chọn top_k_final/top_n (dev) rồi predict (public).

Mọi file sinh ra (cache, báo cáo eval, submission) đi vào --out-dir (mặc định
rag_model/output/), tách khỏi code trong rag_model/legalqa/.

    python -m rag_model.run --mode eval           # chỉ chạy sweep, chọn cấu hình, không predict
    python -m rag_model.run --mode full            # eval (nếu chưa có cache) -> predict -> zip
    python -m rag_model.run --mode predict --skip-eval   # dùng lại cache/best_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_model.legalqa.doc_meta import load_or_build_doc_meta
from rag_model.legalqa.dev_split import load_or_build_dev_split
from rag_model.legalqa.evaluate import sweep
from rag_model.legalqa.predict import load_questions, predict_all
from rag_model.legalqa.package import build_submission
from rag_model.retrieve.cache_io import save_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "LegalQA_Public_Test"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["eval", "predict", "full"], default="full")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "rag_model" / "output"))
    ap.add_argument("--skip-eval", action="store_true",
                     help="dùng cache/best_config.json thay vì chạy lại sweep (lỗi nếu chưa có)")
    ap.add_argument("--dev-size", type=int, default=300)
    ap.add_argument("--top-k-final-options", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--top-n-options", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--reserve-gb", type=float, default=1.0,
                     help="VRAM giữ chỗ sau khi mỗi pha nạp xong model (xem gpu_reserve.py)")
    return ap.parse_args()


def _empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=== doc_meta ===")
    doc_meta = load_or_build_doc_meta(DATA_DIR / "selected-contexts", cache_dir / "doc_meta.json")

    best_config_path = cache_dir / "best_config.json"
    have_cached_config = args.skip_eval and best_config_path.exists()
    need_eval = args.mode in ("eval", "full") and not have_cached_config
    need_predict = args.mode in ("predict", "full")

    train_data = {}
    dev_qids = []
    if need_eval:
        with (DATA_DIR / "train.json").open(encoding="utf-8") as f:
            train_data = json.load(f)
        dev_qids = load_or_build_dev_split(DATA_DIR / "train.json", cache_dir / "dev_qids.json",
                                            n=args.dev_size)

    questions = {}
    if need_predict:
        questions = load_questions(DATA_DIR / "public-official.json")

    # ---- Pha 1: candidate (BM25 + 2 encoder, KHÔNG reranker) ----
    pool_by_qid = {}
    if need_eval or need_predict:
        print("=== Pha 1/2 — CandidateRetriever (BM25 + bge-m3 + e5-large, không reranker) ===")
        from rag_model.retrieve.candidate import CandidateRetriever
        all_questions = {}
        if need_eval:
            all_questions.update({qid: train_data[qid]["question"] for qid in dev_qids})
        if need_predict:
            all_questions.update({qid: item["question"] for qid, item in questions.items()})

        candidate_retriever = CandidateRetriever(reserve_gb=args.reserve_gb)
        pool_by_qid = candidate_retriever.search_batch(all_questions)
        save_candidates(pool_by_qid, cache_dir / "candidates.json.gz")
        del candidate_retriever
        _empty_cuda_cache()

    # ---- Pha 2: rerank (chỉ reranker) ----
    reranker = None
    if need_eval or need_predict:
        print("=== Pha 2/2 — ChunkReranker (Vietnamese_Reranker, đã fine-tune) ===")
        from rag_model.retrieve.rerank import ChunkReranker
        reranker = ChunkReranker(reserve_gb=args.reserve_gb)

    if have_cached_config:
        with best_config_path.open(encoding="utf-8") as f:
            best_config = json.load(f)
        print(f"=== Dùng lại config đã cache: {best_config} ===")
    elif need_eval:
        print("=== Sweep top_k_final x top_n trên dev split ===")
        pool_dev = {qid: pool_by_qid[qid] for qid in dev_qids}
        results = sweep(dev_qids, train_data, pool_dev, reranker, doc_meta,
                         top_k_final_options=tuple(args.top_k_final_options),
                         top_n_options=tuple(args.top_n_options))
        with (out_dir / "eval_report.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        best_config = {"top_k_final": results[0]["top_k_final"], "top_n": results[0]["top_n"]}
        with best_config_path.open("w", encoding="utf-8") as f:
            json.dump(best_config, f, ensure_ascii=False)
        print(f"=> chọn {best_config} (METEOR={results[0]['meteor']:.4f})")
    else:
        if not best_config_path.exists():
            raise SystemExit(
                "Không có cache/best_config.json — chạy `--mode eval` trước, hoặc bỏ --skip-eval."
            )
        with best_config_path.open(encoding="utf-8") as f:
            best_config = json.load(f)

    if args.mode == "eval":
        return

    print("=== Predict trên public-official.json ===")
    pool_public = {qid: pool_by_qid[qid] for qid in questions}
    ranked_public = reranker.rerank_batch(pool_public, top_k_final=best_config["top_k_final"])
    answers = predict_all(ranked_public, questions, doc_meta, top_n=best_config["top_n"])
    with (out_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False)

    if args.mode == "predict":
        return

    print("=== Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), out_dir / "submission.zip")


if __name__ == "__main__":
    main()
