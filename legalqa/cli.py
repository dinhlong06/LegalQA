"""Entrypoint console script `legalqa`.

    legalqa <questions.json> <out_dir> [--arm full|nollm|both] [--config configs/v11.yaml]
            [--devices cuda:0,cuda:1] [--no-llm] [--corpus DIR] [--index DIR]

Sinh:
    <out_dir>/submission.zip        arm full (LLM) — PRIMARY
    <out_dir>/submission_nollm.zip  arm nollm (trích xuất thuần)
    <out_dir>/run_log.json          log lượt chạy (mốc thời gian, cấu hình, số câu)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from .build_submission import build_submission
from .config import default_config_path, load_config
from .pipeline import LegalQAPipeline


def _questions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"⛔ {path} không phải dict {{qid: {{question: ...}}}}")
    for qid, item in data.items():
        if not isinstance(item, dict) or "question" not in item:
            raise SystemExit(f"⛔ {qid} thiếu trường 'question'")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(prog="legalqa", description="LegalQA v11 — tái lập bài nộp")
    ap.add_argument("questions", help="questions.json {qid: {question}}")
    ap.add_argument("out_dir")
    ap.add_argument("--arm", choices=["full", "nollm", "both"], default="both")
    ap.add_argument("--config", default=str(default_config_path()))
    ap.add_argument("--devices", default=None, help="v.d. 'cuda:0' hoặc 'cuda:0,cuda:1'")
    ap.add_argument("--no-llm", action="store_true", help="tắt LLM (chỉ arm nollm)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--cache", default=None,
                    help="thư mục cache theo pha (mặc định <out_dir>/.cache); 'off' để tắt")
    args = ap.parse_args(argv)

    cfg = load_config(args.config if Path(args.config).exists() else None)
    if args.index:
        cfg["INDEX_DIR"] = args.index
    devices = args.devices.split(",") if args.devices else None
    arms = ("nollm",) if (args.no_llm or args.arm == "nollm") else \
           (("full", "nollm") if args.arm == "both" else ("full",))
    use_llm = (not args.no_llm) and ("full" in arms)

    questions = _questions(args.questions)
    print(f"=== LegalQA v11 · {len(questions)} câu · arm={arms} ===", flush=True)

    t0 = time.time()
    cache_dir = None if args.cache == "off" else (args.cache or str(Path(args.out_dir) / ".cache"))
    pipe = LegalQAPipeline(cfg, devices=devices, corpus_dir=args.corpus, use_llm=use_llm,
                           cache_dir=cache_dir)
    answers = pipe.predict(questions, arms=arms)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    expected = set(questions)
    written = {}
    if "nollm" in answers:
        build_submission(answers["nollm"], expected, out / "submission_nollm.zip")
        written["nollm"] = "submission_nollm.zip"
    if "full" in answers:
        build_submission(answers["full"], expected, out / "submission.zip")
        written["full"] = "submission.zip"

    log = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v11",
        "n_questions": len(questions),
        "arms": list(answers),
        "written": written,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "config": {k: cfg[k] for k in ("ANSWER_TEMPLATE", "CONCL", "DOC_K", "TOP_K_RERANK",
                                       "RERANKER_BASE", "LLM_MODEL")
                   if k in cfg},
        "devices": pipe.devices,
        "llm_used": bool(pipe.llm and pipe.llm.available),
    }
    (out / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())