"""Cấu hình đóng băng của v11 (Cell 2 + v8_decisions.json của lượt Kaggle 19/09).

v11 = v7 (retrieval base · t2 base · gen free_plus_verbatim · T3 · echo2) + encoder train lại
+ LLM_MAX_NEW_FREE 768 + trim khoản + cap gen 170 từ. Cổng CE legal-VN trượt (Δ −0,041)
=> vẫn Vietnamese_Reranker zero-shot.


Các giá trị ở đây là những gì `v7_decisions.json` đã chốt sau các cổng split-half trên dev:
    retrieval = "base"   (RRF 3 kênh bm25 + A + B, trọng số 1.0, không kb/prf/rw)
    t2        = "base"   (chỉ điểm CE zero-shot, không breadcrumb/LLM listwise)
    gen       = "free_plus_verbatim"  (LLM Qwen3-1.7B viết đoạn mở đầu + trích nguyên văn)
    template  = T3_theo_qd_tai_title · concl = echo2 · LTR off · warmup 0 câu
"""
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # yaml không bắt buộc — config có thể truyền bằng dict
    yaml = None

# --- tham số mô hình / retrieval (Cell 2) -----------------------------------
TOP_K_RETRIEVE = 100          # số ứng viên sau RRF mỗi kênh
TOP_K_RERANK = 5              # trần số Điều đưa vào câu trả lời
DOC_K = 5                     # số văn bản mở ra ở tầng 2
MAX_DIEU_CANDIDATES = 100     # trần số Điều đưa vào rerank tầng 2
DENSE_MAX_SEQ_LEN = 256
RERANK_SUBBATCH = 64
ENCODE_BATCH_SIZE = 256
ENCODE_FP16 = True
FP16_GUARD_N = 256
FP16_MIN_COS = 0.995
SEED = 42

# --- encoder (Cell 2) --------------------------------------------------------
BASE_DENSE_MODEL_A = "AITeamVN/Vietnamese_Embedding_v2"
BASE_DENSE_MODEL_B = "mainguyen9/vietlegal-harrier-0.6b"
QUERY_PREFIX_A = ""
QUERY_PREFIX_B = ""
PASSAGE_PREFIX_B = ""

# --- reranker / generation (Cell 2) -----------------------------------------
RERANKER_BASE = "AITeamVN/Vietnamese_Reranker"
USE_RERANKER_FINETUNE = False        # v7: zero-shot (v6_2 đo âm hai nửa)
CONCL = "echo2"
ANSWER_TEMPLATE = "T3_theo_qd_tai_title"
USE_LLM = True
LLM_MODEL = "Qwen/Qwen3-1.7B"
LLM_GEN_BATCH = 8
LLM_ARTICLE_WORDS = 700
LLM_MAX_NEW_FREE = 768   # v9+: 768 (v7 là 512)
RRF_K = 60
PARAM_BUDGET = 4_000_000_000

# Danh sách model BTC chấp nhận (rule.md §2.3) — cổng chặn cuối.
REGISTERED_MODELS = {
    "AITeamVN/Vietnamese_Embedding_v2", "mainguyen9/vietlegal-harrier-0.6b",
    "BAAI/bge-m3", "intfloat/multilingual-e5-large",
    "AITeamVN/Vietnamese_Reranker", "Qwen/Qwen3-Reranker-0.6B",
}

# Config RRF/RRF base — trùng BASE_CFG ở Cell 11b.
BASE_CFG = {"w": {}, "kb": None, "prf": None, "rw": None}

_DEFAULTS = {k: v for k, v in globals().items()
             if k.isupper() and not k.startswith("_")}


def load_config(path=None, **overrides):
    """Trả về dict cấu hình = mặc định v7 <- YAML (nếu có) <- overrides."""
    cfg = dict(_DEFAULTS)
    if path:
        if yaml is None:
            raise RuntimeError("cần PyYAML để đọc config .yaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data.pop("reference", None)
        cfg.update({k.upper(): v for k, v in data.items()})
    cfg.update(overrides)
    # Cổng cứng cho encoder + reranker (như Cell 2 notebook). LLM chỉ cảnh báo: Qwen3-1.7B
    # nằm trong danh sách chính thức BTC phát (xem README · check_model_compliance.py) nhưng
    # tập REGISTERED_MODELS nội bộ của notebook không khai nó, nên giữ đúng hành vi gốc.
    unknown = [m for m in (cfg["BASE_DENSE_MODEL_A"], cfg["BASE_DENSE_MODEL_B"],
                           cfg["RERANKER_BASE"]) if m not in cfg["REGISTERED_MODELS"]]
    if unknown:
        raise SystemExit(f"⛔ model ngoài danh sách BTC: {unknown} (rule.md §2.3)")
    return cfg


def package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    return package_root() / "configs" / "v11.yaml"


def resolve_path(cfg, key, default_rel):
    """Đường dẫn tài nguyên: ưu tiên cfg[key], rồi biến môi trường, rồi mặc định trong gói."""
    val = cfg.get(key) or os.environ.get(key)
    if val:
        return Path(val)
    return package_root() / default_rel