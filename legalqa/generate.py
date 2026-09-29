"""Sinh câu trả lời bằng LLM Qwen3-1.7B — arm `free_plus_verbatim` (Cell 10b của notebook gốc).

Giải mã GREEDY (`do_sample=False`) nên với cùng checkpoint + cùng bản transformers, kết quả
là tất định. Prompt FREE_USER buộc LLM viết đoạn mở đầu + trích nguyên văn + câu kết; nếu
câu sinh ra không đạt `valid_free` (>= 20 từ và có nhắc "Điều"/số hiệu) thì pipeline lùi về
`render_answer` (arm base) cho câu đó.
"""
import threading

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .answer import article_text, unit_ref

_print_lock = threading.Lock()
GEN_CAP_WORDS = 170   # v11: cap cố định cho `gen` (không theo gold_len)
_THINK_RE = __import__("re").compile(r"<think>.*?</think>", __import__("re").DOTALL)

FREE_USER = (
    "Câu hỏi: {q}\n\nĐiều luật liên quan ({ref}):\n{article}\n\n"
    "Viết câu trả lời cho câu hỏi, dựa hoàn toàn vào điều luật trên, theo đúng khuôn:\n"
    "1. Mở đầu: \"Căn cứ {ref} quy định như sau:\"\n"
    "2. Trích nguyên văn các khoản, điểm liên quan đến câu hỏi.\n"
    "3. Kết luận bắt đầu bằng \"Như vậy,\" trả lời trực tiếp câu hỏi.\n"
    "Không thêm thông tin ngoài điều luật.")


def _chat(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return (tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                + "<think>\n\n</think>\n\n")


def load_llm(device: str, model_name: str = "Qwen/Qwen3-1.7B"):
    """Nạp LLM fp16 trên `device`; tự lùi fp32 nếu logit không hữu hạn (Qwen dễ tràn fp16)."""
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    if device.startswith("cuda"):
        mdl = mdl.half()
    mdl = mdl.to(device).eval()
    enc = tok([_chat(tok, "Điều 1. Phạm vi điều chỉnh. Luật này quy định về hợp đồng lao động.")],
              return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            logits = mdl(**enc, logits_to_keep=1).logits[:, -1, :].float()
        except TypeError:
            logits = mdl(**enc).logits[:, -1, :].float()
    if not bool(torch.isfinite(logits).all()):
        print("  [LLM] logit không hữu hạn ở fp16 -> lùi fp32")
        mdl = mdl.float()
    return mdl, tok


def trim_incomplete(text):
    """Văn bản dừng vì chạm max_new_tokens -> bỏ câu dở dang cuối.
    Chỉ cắt khi còn giữ được ít nhất nửa văn bản; không thì để nguyên."""
    import re
    t = str(text).rstrip()
    if not t or t[-1] in ".!?:;)\"”»":
        return t
    ends = [m.end() for m in re.finditer(r"[.!?;](?=\s|$)", t)]
    if ends and ends[-1] >= 0.5 * len(t):
        return t[:ends[-1]]
    nl = t.rfind("\n")
    if nl >= 0.5 * len(t):
        return t[:nl].rstrip()
    return t


def cap_gen(gen, cap_words=GEN_CAP_WORDS):
    """v11: cắt `gen` về cap_words từ rồi bỏ câu dở dang."""
    if not gen:
        return gen
    words = gen.split()
    if len(words) <= cap_words:
        return gen
    return trim_incomplete(" ".join(words[:cap_words]))


def generate(prompts, model, tokenizer, device, max_new_tokens: int, batch_size: int = 8):
    """Greedy generation theo lô, tự giảm batch khi OOM. Câu chạm max_new_tokens mà chưa gặp
    token dừng thì bỏ câu dở dang cuối (llm_generate của notebook v8+)."""
    stop_ids = {i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id,
                            tokenizer.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int)}
    out, i, bs = [], 0, batch_size
    while i < len(prompts):
        chunk = prompts[i:i + bs]
        try:
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     temperature=None, top_p=None, top_k=None,
                                     repetition_penalty=1.05, pad_token_id=tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(new, skip_special_tokens=True)
            for row, t in zip(new.tolist(), texts):
                t = _THINK_RE.sub("", t).strip()
                cut = len(row) >= max_new_tokens and row[-1] not in stop_ids
                out.append(trim_incomplete(t) if cut else t)
            i += len(chunk)
        except RuntimeError as e:
            from .rerank import is_oom
            if is_oom(e) and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            raise
    return out


def build_free_prompt(question: str, chunk) -> str:
    return FREE_USER.format(q=question, ref=unit_ref(chunk), article=article_text(chunk))


def valid_free(text, c) -> bool:
    """Cổng chấp nhận câu LLM sinh: >= 20 từ và có nhắc "Điều"/số hiệu văn bản."""
    t = str(text)
    return len(t.split()) >= 20 and ("Điều" in t or bool(c.get("so_hieu") and c["so_hieu"] in t))


def compose_answer(chunk, question, gen) -> str:
    """arm `free_plus_verbatim`: gen + "\\n\\n" + base, lùi về base nếu gen không hợp lệ.
    v11: gen cap 170 từ, base trim khoản (compose_answer của notebook v11)."""
    from .answer import render_answer
    gen = cap_gen(gen)
    base = render_answer([chunk], 1, question, trim_khoan=True)
    if not gen or not valid_free(gen, chunk):
        return base
    return gen + "\n\n" + base


class LLMGenerator:
    """Bọc 1 hoặc nhiều bản LLM (mỗi GPU một bản) để sinh song song."""

    def __init__(self, cfg, devices):
        self.models, self.tokens, self.devices = {}, {}, []
        for dev in devices:
            try:
                m, t = load_llm(dev, cfg["LLM_MODEL"])
                self.models[dev], self.tokens[dev] = m, t
                self.devices.append(dev)
                print(f"  [LLM] {cfg['LLM_MODEL']} trên {dev}")
            except Exception as e:
                print(f"  [LLM] không nạp được trên {dev}: {e}")
        self.batch = cfg.get("LLM_GEN_BATCH", 8)

    @property
    def available(self) -> bool:
        return bool(self.models)

    def n_params(self) -> int:
        if not self.models:
            return 0
        return sum(p.numel() for p in next(iter(self.models.values())).parameters())

    def generate_all(self, items: dict) -> dict:
        """items {qid: (question, chunk)} -> {qid: text}. Chia đều các GPU có LLM."""
        if not items:
            return {}
        from .retrieval import split_evenly
        keys = list(items)
        chunks = split_evenly(keys, len(self.devices))
        result = {}

        def _worker(subset, dev, widx):
            tok = self.tokens[dev]
            subset = sorted(subset, key=lambda k: len(build_free_prompt(*items[k])))
            res = {}
            for b in range(0, len(subset), self.batch * 4):
                part = subset[b:b + self.batch * 4]
                prompts = [_chat(tok, build_free_prompt(*items[k])) for k in part]
                outs = generate(prompts, self.models[dev], tok, dev,
                                max_new_tokens=self.max_new if hasattr(self, "max_new") else 512,
                                batch_size=self.batch)
                res.update(zip(part, outs))
            return res

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.devices)) as ex:
            futs = [ex.submit(_worker, chunk, dev, i)
                    for i, (chunk, dev) in enumerate(zip(chunks, self.devices))]
            for f in futs:
                result.update(f.result())
        return result