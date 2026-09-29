# UIT DSC 2026 · Task 2 LegalQA — Private Test Reproduction Package (Team Artiz)

Self-contained package: MIT-licensed inference pipeline (chunk corpus → two-tier retrieval →
cross-encoder rerank → article selection → template answer + LLM opening), fine-tuned checkpoints,
pre-built indexes, and training logs.

**Reproduced submission: `submission_v11_private.zip` — private METEOR 0,6047 / ROUGE-L 0,5167,
#11 on the private leaderboard** (1,918 questions).

## Result

Task 2 is scored by **METEOR (`alpha = 0.9`, recall-heavy)** with ROUGE-L as the secondary metric.

| split | system | METEOR | ROUGE-L |
|---|---|---:|---:|
| public test | v7, arm `full` (same architecture) | 0,5932 | 0,4732 |
| private test | **this package (v11), arm `full`** — **#11 on the official leaderboard** | **0,6047** | 0,5167 |

- Two arms share the same retrieval/reranker/template and differ only in one place: arm `full` lets
  Qwen3-1.7B write an opening paragraph (`free_plus_verbatim`) before the template body. METEOR is
  the primary metric, so `full` is the submitted arm; arm `nollm` (extract-only) is kept as backup
  and is the team's highest-ROUGE-L configuration.
- Answers are built extractively (template + verbatim quotations); METEOR's recall bias means long
  faithful extraction beats short generation — the pipeline follows the length/shape statistics of
  the `train.json` reference answers.

**Model weights (Google Drive):** https://drive.google.com/file/d/1q8oKzywcJQbNIK4e0b7P9-a0EFJj131U/view?usp=sharing
(`legalqa_v11_models.zip`, ~8.6 GB — the 4 checkpoints + pre-built indexes of the layout table in
§5, matching `models/` + `index/` byte for byte; verify against [`SHA256SUMS`](SHA256SUMS)).

---

## 1. Architecture

```
question ─┬─ BM25 (tier 1, 450-word chunks)              top-100 ─┐
          ├─ dense A-ft (Vietnamese_Embedding_v2 FT)      top-100 ┼─► RRF over 3 channels
          └─ dense B-ft (vietlegal-harrier-0.6b FT)       top-100 ┘        │
                                                                           ▼
              cross-encoder Vietnamese_Reranker (zero-shot) scores (question, chunk)
              → aggregate chunks → documents (max) → keep DOC_K = 5 documents
                                                                           │
              tier 2: collect Articles (Điều) inside the 5 documents → CE rescore → pick 1
                                                                           │
              render_answer template T3_theo_qd_tai_title + echo2 closing line   (arm nollm)
              LLM Qwen3-1.7B (greedy) writes opening + verbatim excerpt          (arm full)
                                                                           ▼
                                    submission.zip {qid: {"answer": ...}}
```

- Corpus: 8,512 documents → **166,078 tier-1 chunks** (450 words) + **235,204 tier-2 chunks**
  (Article/Mục granularity).
- **Parameter budget** (counted from weight files; rule: < 4 B for the whole system):

| component | checkpoint | parameters |
|---|---|---:|
| dense channel A | `models/A-ft` (FT `AITeamVN/Vietnamese_Embedding_v2`) | 566,705,152 |
| dense channel B | `models/B-ft` (FT `mainguyen9/vietlegal-harrier-0.6b`) | 596,049,920 |
| cross-encoder | `models/Vietnamese_Reranker` (zero-shot `AITeamVN/Vietnamese_Reranker`) | 567,755,777 |
| LLM | `models/Qwen3-1.7B` (base `Qwen/Qwen3-1.7B`) | 1,720,574,976 |
| **total, arm full** | | **3,451,085,825 = 3.45 B / 4 B** |

BM25 has no neural parameters. Runtime runs with `HF_HUB_OFFLINE=1`; no external API is called.

## 2. Usage

### Docker (recommended)

Requirements: 1 NVIDIA GPU ≥ 11 GB (peak ~7 GB on arm `full`), NVIDIA Container Toolkit, ~20 GB
disk for the image (`uitdsc2026-legalqa:private` ≈ 19.5 GB). All weights + indexes are baked into
the image — **nothing is downloaded at inference time**.

```bash
docker build -t uitdsc2026-legalqa:private .

bash run_private.sh /path/to/private-official.json /path/to/out
# equivalently:
docker run --rm --gpus '"device=0"' \
  -v /path/to/questions_dir:/data:ro -v /path/to/out:/out \
  uitdsc2026-legalqa:private /data/private-official.json /out --arm both
# -> /path/to/out/submission.zip and submission_nollm.zip
```

`questions.json` is JSON `{qid: {"question": "..."}}` in the organizer's format. The submission is
a zip containing **exactly one** `submission.json` of `{qid: {"answer": ...}}`.

### Without Docker

```bash
python -m venv --system-site-packages /opt/venv   # needs torch 2.6.0+cu124 available
/opt/venv/bin/pip install -r requirements.txt
/opt/venv/bin/pip install --no-deps -e .
/opt/venv/bin/legalqa <questions.json> <out_dir> --arm both
```

Reference runtime (1 GPU, 1,000 questions): article selection ~40–80 min + LLM generation ~40 min.
Two GPUs (`--devices cuda:0,cuda:1`) are faster.

## 3. Verification gates

- **Gate 1 — absolute score on 300 dev questions** (gold from `train.json`), official scorer:
  arm `nollm` METEOR 0,5751 / ROUGE-L 0,5624 · arm `full` METEOR 0,5968 / ROUGE-L 0,4747.
  Verified identical inside the Docker image (nollm exact; full within 0,0002 — fp16 rounding in
  greedy LLM decode across GPU topologies).
- **Gate 2 — parity on public test** vs the scored v7 reference: **1000/1000 articles** selected
  identically (both arms); `full` text METEOR 0,9969 vs the reference (87 questions differ by a few
  tokens only, from LLM decoding).
- **v11 vs Kaggle run** (`llm_batch_parity_v11.py` logic): the 67 divergent answers were traced to
  Kaggle's 2×T4 batch splitting of the LLM stage, not to any pipeline difference; regenerated
  batches with the same batching reproduce the reference.

Arm `full` is therefore verified at article-choice parity (100%), not byte-identity of LLM text.

## 4. Retraining (for audit — production uses the shipped checkpoints)

`scripts/train_encoders.py` is a **verbatim copy** of the training worker from the original
notebook (seed 42; 3,436 citation-label pairs from Task 2 data; optimizer/mini-batch as logged).
The pair-generation recipe and run logs: `training_logs/` (`original_notebook.ipynb`,
`A-ft_meta.json`, `B-ft_meta.json`, `v8_decisions.json`, `experiment_log.jsonl`).

## 5. Repo layout

```
LegalQA/
├── README.md  LICENSE  pyproject.toml  requirements.txt  Dockerfile  run_private.sh
├── configs/v11.yaml          single configuration of the submitted arm
├── legalqa/                  the package: cli · config · data · encoders · index · bm25 ·
│                             retrieval · rerank · pipeline · answer · generate · phase_cache ·
│                             chunking · build_submission
├── scripts/                  build_index.py · train_encoders.py · make_checksums.sh
├── training_logs/            training logs (seed 42), metadata, original notebook, decisions
├── SHA256SUMS                pins index/ + models/ contents (shipped via Drive/Docker, not git)
├── models/                   4 checkpoints (~4.4 GB, not in git)
└── index/                    tier-1/tier-2 chunks + dense matrices (~1.5 GB, not in git)
```

## 6. Licenses

Code written by the team is **MIT** (see `LICENSE`). Third-party libraries and weights keep their
original licenses:

| component | source | license |
|---|---|---|
| `AITeamVN/Vietnamese_Embedding_v2` | HuggingFace | per original model page |
| `mainguyen9/vietlegal-harrier-0.6b` | HuggingFace | per original model page |
| `AITeamVN/Vietnamese_Reranker` | HuggingFace | per original model page |
| `Qwen/Qwen3-1.7B` | HuggingFace | Apache-2.0 |
| transformers · sentence-transformers · numpy | PyPI | Apache-2.0 / MIT / BSD |

All 4 models are on the organizer's official approved-model list (checked automatically by the
team's `check_model_compliance.py` against the published list).
