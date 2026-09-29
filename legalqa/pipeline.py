"""Đường suy luận production của v7 — ghép các mảnh thành pipeline hoàn chỉnh.

Cấu hình đã đóng băng (v7_decisions.json): retrieval "base", t2 "base", gen
"free_plus_verbatim", template T3, concl echo2. Không chạy lại các cng dev — quyết định đã
chốt và được ghi trong training_logs/.

Quản lý VRAM (để chạy được cả trên 1 GPU 11 GB):
    1. nạp index (chunk + ma trận dense) — BM25 ở RAM
    2. encode TOÀN BỘ query một lần bằng 2 encoder (mỗi câu encode riêng, đúng như notebook
       gọi `encode([q_text])`; batch không đổi kết quả)
    3. ĐẨY 2 encoder sang CPU rồi mới nạp reranker + LLM — đỉnh VRAM chỉ còn reranker + LLM
    (giữ encoder trên GPU như notebook cần 2×T4 16 GB; BTC chưa công bố phần cứng nên lùi được)

    pipe = LegalQAPipeline(cfg)
    answers = pipe.predict({qid: {"question": ...}}, arms=("full", "nollm"))
"""
import time

from .answer import render_answer
from .bm25 import BM25
from .chunking import load_corpus, tokenize_simple
from .config import resolve_path
from .retrieval import parallel_process, retrieve_two_tier
from .rerank import load_reranker
from .phase_cache import PhaseCache, make_key


class LegalQAPipeline:
    FALLBACK_ANSWER = "Không tìm thấy thông tin pháp lý cho câu hỏi này."

    def __init__(self, cfg, devices=None, index_dir=None, corpus_dir=None, use_llm=None,
                 cache_dir=None):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.devices = devices or self._detect_devices()
        self.index_dir = resolve_path(cfg, "INDEX_DIR", "index")
        self.corpus_dir = corpus_dir or cfg.get("CORPUS_DIR") or \
            resolve_path(cfg, "CORPUS_DIR", "selected-contexts")
        self.use_llm = cfg.get("USE_LLM", True) if use_llm is None else use_llm

        t0 = time.time()
        all_chunks, all_chunks_t1, dieu_by_doc, channels = self._load_index_or_build()
        self.all_chunks = all_chunks
        self.all_chunks_t1 = all_chunks_t1
        self.dieu_by_doc = dieu_by_doc
        self.channels = channels
        self.chunk_by_id = {c["id"]: c for c in all_chunks}

        print(f"[pipeline] BM25 tầng 1 ({len(all_chunks_t1)} chunk)...", flush=True)
        tokenized_t1 = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks_t1]
        self.bm25_t1 = BM25(tokenized_t1)
        self._prepared = False
        self.llm = None
        print(f"[pipeline] sẵn sàng cấu trúc sau {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _detect_devices():
        import torch
        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)] if n else ["cpu"]

    def _load_index_or_build(self):
        from .index import load_index
        if (self.index_dir / "manifest.json").exists():
            print(f"[pipeline] nạp index bake {self.index_dir}", flush=True)
            return load_index(self.index_dir, self.devices, self.cfg)
        print(f"[pipeline] không có index bake -> dựng từ corpus {self.corpus_dir}", flush=True)
        from .encoders import build_channels, encode_corpus, load_encoder
        all_chunks, all_chunks_t1, dieu_by_doc = load_corpus(self.corpus_dir)
        m_a = load_encoder(str(resolve_path(self.cfg, "ENCODER_A_PATH", "models/A-ft")),
                           self.devices[0])
        m_b = load_encoder(str(resolve_path(self.cfg, "ENCODER_B_PATH", "models/B-ft")),
                           self.devices[-1])
        channels = build_channels(m_a, m_b, self.cfg)
        encode_corpus(channels, [c["text"] for c in all_chunks_t1], self.devices[0])
        return all_chunks, all_chunks_t1, dieu_by_doc, channels

    def _reranker_source(self):
        local = resolve_path(self.cfg, "RERANKER_PATH", "models/Vietnamese_Reranker")
        return str(local) if local.exists() else self.cfg["RERANKER_BASE"]

    def _llm_source(self):
        local = resolve_path(self.cfg, "LLM_PATH", "models/Qwen3-1.7B")
        return str(local) if local.exists() else self.cfg["LLM_MODEL"]

    def _encode_queries(self, questions):
        """Encode từng câu riêng cho từng kênh (giống notebook) rồi lưu lại."""
        import numpy as np
        qids = list(questions)
        emb = {}
        for i, ch in enumerate(self.channels):
            for qid in qids:
                q = questions[qid]["question"]
                q_text = ch["query_prefix"] + q if ch["query_prefix"] else q
                emb.setdefault(qid, []).append(
                    ch["model"].encode([q_text], convert_to_numpy=True,
                                       normalize_embeddings=True)[0])
            print(f"  [encode query] kênh {ch['name']} xong {len(qids)} câu", flush=True)
        return emb

    def _offload_encoders(self):
        import torch
        for ch in self.channels:
            try:
                ch["model"].to("cpu")
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare(self, questions):
        """Encode query -> đẩy encoder sang CPU -> nạp reranker + LLM. Idempotent."""
        if self._prepared:
            return
        print(f"[pipeline] encode {len(questions)} query...", flush=True)
        self.q_emb = self._encode_queries(questions)
        self._offload_encoders()
        print("[pipeline] encoder -> CPU, nạp reranker zero-shot + LLM...", flush=True)

        self.reranker_models, self.reranker_tokens = {}, {}
        reranker_source = self._reranker_source()
        print(f"  [reranker] nguồn: {reranker_source}", flush=True)
        for dev in self.devices:
            m, t = load_reranker(dev, reranker_source)
            if m is not None:
                self.reranker_models[dev], self.reranker_tokens[dev] = m, t
        if not self.reranker_models:
            print("  ⚠️ KHÔNG có reranker -> pipeline chạy RRF thuần, điểm sẽ thấp hơn hẳn")

        # LLM nạp LAZY sau khi chọn xong Điều (xem _load_llm): hai giai đoạn không dùng chung VRAM.
        self.llm = None
        self._prepared = True

    def _load_llm(self):
        """Giải phóng reranker rồi mới nạp LLM — chọn Điều xong thì reranker không còn dùng."""
        import gc, torch
        devs = list(self.reranker_models) or self.devices
        self.reranker_models, self.reranker_tokens = {}, {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.use_llm:
            from .generate import LLMGenerator
            llm_cfg = dict(self.cfg)
            llm_cfg["LLM_MODEL"] = self._llm_source()
            print(f"  [LLM] nguồn: {llm_cfg['LLM_MODEL']}", flush=True)
            self.llm = LLMGenerator(llm_cfg, devs)

    # ------------------------------------------------------------------ predict
    def _pick_one(self, qid, question, dev, memo1, memo2):
        ranked, scores = retrieve_two_tier(
            question, self.bm25_t1, self.channels, self.all_chunks_t1, self.dieu_by_doc,
            reranker_model=self.reranker_models.get(dev),
            reranker_tokenizer=self.reranker_tokens.get(dev),
            doc_k=self.cfg["DOC_K"], memo_t1=memo1, memo_t2=memo2,
            top_k_retrieve=self.cfg["TOP_K_RETRIEVE"],
            max_dieu_candidates=self.cfg["MAX_DIEU_CANDIDATES"],
            q_embs=self.q_emb[qid])
        top = (ranked or [])[:self.cfg["TOP_K_RERANK"]]
        return top[0]["id"] if top else None

    _KEY_CFG = ("DOC_K", "TOP_K_RETRIEVE", "TOP_K_RERANK", "MAX_DIEU_CANDIDATES",
                "RERANKER_BASE", "LLM_MODEL", "LLM_MAX_NEW_FREE")

    def predict(self, questions: dict, arms=("full", "nollm")):
        """questions: {qid: {"question": str}} -> {arm: {qid: answer}}.

        Có cache_dir: `picked` và `llm_gen` được ghi theo pha; chạy lại cùng câu hỏi + cấu hình
        thì bỏ qua pha đã xong (kể cả không nạp encoder/reranker nếu `picked` đã đủ)."""
        qids = list(questions)
        key = make_key(questions, self.cfg, self._KEY_CFG,
                       extra=(str(self._reranker_source()), str(self._llm_source())))
        cache = PhaseCache(self.cache_dir, key)

        picked = cache.load("picked")
        if all(q in picked for q in qids):
            print(f"[predict] cache: dùng lại 'chọn Điều' của {len(qids)} câu", flush=True)
        else:
            self.prepare(questions)
            rerank_devs = list(self.reranker_models) or self.devices

            def _worker(chunk_ids, dev, widx):
                out = {}
                for qid in chunk_ids:
                    # memo PHẢI mới cho từng câu: khoá memo là chunk id, nhưng điểm CE phụ thuộc
                    # (câu hỏi, văn bản). Dùng chung memo qua nhiều câu = lấy điểm của câu khác.
                    out[qid] = self._pick_one(qid, questions[qid]["question"], dev, {}, {})
                return out

            print(f"[predict] chọn Điều cho {len(qids)} câu trên {rerank_devs}", flush=True)
            picked = parallel_process(qids, _worker, rerank_devs, label="chọn Điều")
            cache.save("picked", picked)

        answers = {}
        if "nollm" in arms:
            answers["nollm"] = {q: self._render(q, picked[q], questions) for q in qids}
        if "full" in arms:
            full_gen = cache.load("llm_gen")
            todo = [q for q in qids if picked[q] and q not in full_gen]
            if todo:
                if not hasattr(self, "reranker_models"):  # pha chọn Điều đã bỏ qua nhờ cache
                    self.reranker_models = {}
                self._load_llm()
                if self.llm and self.llm.available:
                    self.llm.max_new = self.cfg.get("LLM_MAX_NEW_FREE", 512)
                    print(f"[predict] LLM sinh {len(todo)}/{len(qids)} câu "
                          f"(còn lại lấy từ cache)...", flush=True)
                    step = 200  # ghi cache mỗi 200 câu: crash giữa chừng chỉ mất một đợt
                    for i in range(0, len(todo), step):
                        items = {q: (questions[q]["question"], self.chunk_by_id[picked[q]])
                                 for q in todo[i:i + step]}
                        full_gen.update(self.llm.generate_all(items))
                        cache.save("llm_gen", full_gen)
            else:
                print("[predict] cache: dùng lại câu LLM sinh", flush=True)
            answers["full"] = {q: self._compose(q, picked[q], questions, full_gen.get(q))
                               for q in qids}
        return answers

    def _render(self, qid, cid, questions):
        if not cid:
            return self.FALLBACK_ANSWER
        return render_answer([self.chunk_by_id[cid]], 1, questions[qid]["question"],
                             trim_khoan=True)  # v11

    def _compose(self, qid, cid, questions, gen):
        if not cid:
            return self.FALLBACK_ANSWER
        from .generate import compose_answer
        return compose_answer(self.chunk_by_id[cid], questions[qid]["question"], gen)