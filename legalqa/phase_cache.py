"""Cache theo pha cho một lượt chạy: chọn Điều (`picked`) và câu LLM sinh (`llm_gen`).

Ghi nguyên tử (tmp rồi os.replace), đọc khoan dung (file hỏng hoặc khoá lệch = coi như chưa có,
tính lại — không làm chết tiến trình). Khoá = hash(câu hỏi + cấu hình + nguồn model), nên đổi
bất kỳ thứ nào trong đó thì cache cũ tự vô hiệu, không bao giờ dùng nhầm.
"""
import hashlib
import json
import os
from pathlib import Path


def make_key(questions: dict, cfg: dict, keys, extra=()) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([[q, questions[q]["question"]] for q in sorted(questions)],
                        ensure_ascii=False).encode())
    h.update(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True,
                        default=str).encode())
    h.update(json.dumps(list(extra), default=str).encode())
    return h.hexdigest()[:16]


class PhaseCache:
    def __init__(self, cache_dir, key: str):
        self.dir = Path(cache_dir) if cache_dir else None
        self.key = key
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def load(self, name: str) -> dict:
        if not self.dir:
            return {}
        p = self.dir / f"{name}.json"
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if obj.get("key") == self.key:
                return obj["data"]
            print(f"  [cache] {name}: khoá lệch (đổi câu hỏi/cấu hình) -> bỏ", flush=True)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  [cache] {name} hỏng ({type(e).__name__}) -> bỏ, tính lại", flush=True)
        return {}

    def save(self, name: str, data: dict) -> None:
        if not self.dir:
            return
        p = self.dir / f"{name}.json"
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"key": self.key, "data": data}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, p)
