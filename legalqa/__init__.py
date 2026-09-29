"""LegalQA v7 — gói tái lập bài nộp public 0,5932 METEOR / 0,4732 ROUGE-L.

Pipeline (cấu hình đã đóng băng theo training_logs/v7_decisions.json):
    câu hỏi
      -> chunk corpus 2 tầng (Cell 5)
      -> tầng 1: BM25 + dense A-ft + dense B-ft, RRF 3 kênh, CE zero-shot gộp `max`
                 -> chọn DOC_K = 5 văn bản
      -> tầng 2: gom Điều trong 5 văn bản, CE zero-shot -> chọn 1 Điều
      -> render_answer template T3_theo_qd_tai_title + câu kết echo2  (arm nollm)
      -> LLM Qwen3-1.7B greedy sinh free_plus_verbatim (arm full)
      -> submission.json {qid: {"answer": ...}}

Mã nguồn này port từ legalqa/legalqa_kaggle_v7.ipynb, giữ nguyên phép toán.
"""

__version__ = "11.0"