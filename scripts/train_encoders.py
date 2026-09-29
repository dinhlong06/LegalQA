#!/usr/bin/env python
"""Worker fine-tune MỘT dense encoder trên MỘT GPU (Bước 4, Cell 8 của notebook gốc).

Chép NGUYÊN VĂN từ notebook v7 — không sửa một dòng nào, để log huấn luyện trong
training_logs/ tái lập được đúng thuật toán đã chạy:

    python scripts/train_encoders.py --base-model AITeamVN/Vietnamese_Embedding_v2 \
        --gpu-index 0 --rows-path pairs.json --output-dir models/A-ft \
        --max-seq-len 256 --batch-size 64 --mini-batch-size 16 --time-budget-sec 6000 --seed 42

Dữ liệu `rows.json` (cặp anchor/positive/negative) sinh từ nhãn citation train.json —
xem training_logs/TRAINING.md và original_notebook.ipynb Cell 7/8.
"""

import argparse, json, os, sys, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--gpu-index", required=True)
    p.add_argument("--rows-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mini-batch-size", type=int, default=16)
    p.add_argument("--time-budget-sec", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--query-prefix", default="")
    p.add_argument("--passage-prefix", default="")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--use-8bit-optim", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        from datasets import Dataset
        from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                            SentenceTransformerTrainingArguments)
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
        import sentence_transformers as _st
        import accelerate as _acc
    except ImportError as e:
        print(f"[LOI IMPORT] {e}", flush=True)
        print(f"[LOI IMPORT] Ban co the dang dung sentence-transformers/accelerate qua cu -- "
              f"CachedMultipleNegativesRankingLoss can sentence-transformers >= 3.0, "
              f"accelerate >= 1.1.0. Chay:", flush=True)
        print(f'    pip install -U "sentence-transformers>=3.0" "accelerate>=1.1.0"', flush=True)
        sys.exit(1)
    st_ver = tuple(int(x) for x in _st.__version__.split(".")[:2] if x.isdigit())
    if st_ver < (3, 0):
        print(f"[PHIEN BAN CU] sentence-transformers={_st.__version__} (can >= 3.0). Chay: "
              f'pip install -U "sentence-transformers>=3.0"', flush=True)
        sys.exit(1)

    # BAN SUA (log loi that: torch.AcceleratorError OOM ngay o optimizer.step() DAU TIEN --
    # AdamW full fine-tune cho model ~568M can ~6-7GB CHI RIENG optimizer state, khong phu
    # thuoc batch size -- khong batch nao du tren GPU 4GB). LoRA: chi train 1 phan rat nho
    # tham so (dong bang phan con lai) -> optimizer state nho lai theo dung ty le do, GIAI
    # QUYET DUOC loai OOM nay ma batch-backoff khong the giai quyet.
    optim_name = "adamw_torch"
    if args.use_8bit_optim:
        try:
            import bitsandbytes  # noqa: F401
            optim_name = "adamw_bnb_8bit"
            print(f"[{args.base_model}] Dung optimizer AdamW 8-bit (bitsandbytes).", flush=True)
        except ImportError:
            print(f"[{args.base_model}] bitsandbytes khong cai duoc -> dung AdamW thuong.", flush=True)

    with open(args.rows_path, encoding="utf-8") as f:
        rows = json.load(f)
    if args.query_prefix or args.passage_prefix:
        fixed = []
        for r in rows:
            r2 = dict(r)
            r2["anchor"] = args.query_prefix + r["anchor"]
            for k in r:
                if k.startswith("positive") or k.startswith("negative"):
                    r2[k] = args.passage_prefix + r[k]
            fixed.append(r2)
        rows = fixed
    dataset = Dataset.from_list(rows)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_len

    lora_used = False
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                   lora_dropout=args.lora_dropout, bias="none",
                                   target_modules="all-linear")
            model[0].auto_model = get_peft_model(model[0].auto_model, lora_cfg)
            lora_used = True
            n_trainable = sum(pp.numel() for pp in model[0].auto_model.parameters() if pp.requires_grad)
            n_total = sum(pp.numel() for pp in model[0].auto_model.parameters())
            print(f"[{args.base_model}] LoRA bat: {n_trainable}/{n_total} tham so co the train "
                  f"({100*n_trainable/max(n_total,1):.2f}%)", flush=True)
        except Exception as e:
            print(f"[{args.base_model}] LoRA loi ({e}) -> full fine-tune (can nhieu VRAM hon).", flush=True)

    batch_size, mini_batch_size = args.batch_size, args.mini_batch_size
    max_steps, calib_time = 0, None
    t0 = time.time()
    for attempt in range(4):
        try:
            loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
            calib_steps = min(10, max(1, len(dataset) // batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir=args.output_dir + "_tmp", max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                save_strategy="no", report_to=[], disable_tqdm=True, fp16=(device == "cuda:0"),
                optim=optim_name)
            c0 = time.time()
            print(f"[{args.base_model}] calib training (batch={batch_size}, mini_batch={mini_batch_size})...", flush=True)
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - c0) / calib_steps

            budget_left = args.time_budget_sec - (time.time() - t0) - 60
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset) // batch_size) * 8)
            print(f"[{args.base_model}] calib {calib_time:.2f}s/step, ngan sach con "
                  f"{budget_left/60:.1f} phut -> {max_steps} step", flush=True)

            if max_steps > 0:
                targs = SentenceTransformerTrainingArguments(
                    output_dir=args.output_dir + "_tmp", max_steps=max_steps,
                    per_device_train_batch_size=batch_size, learning_rate=2e-5,
                    warmup_steps=0.05, lr_scheduler_type="cosine",
                    logging_steps=max(1, max_steps // 20), save_strategy="no", report_to=[],
                    fp16=(device == "cuda:0"), optim=optim_name)
                SentenceTransformerTrainer(model=model, args=targs, train_dataset=dataset, loss=loss).train()
            break
        except Exception as e:
            if "out of memory" in str(e).lower() and mini_batch_size > 1:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # cache da can kiet toi muc khong con gi de don -- bo qua, cu lui batch
                mini_batch_size = max(1, mini_batch_size // 2)
                print(f"[{args.base_model}] OOM -> mini_batch_size={mini_batch_size}", flush=True)
                continue
            raise

    if lora_used:
        try:
            model[0].auto_model = model[0].auto_model.merge_and_unload()
            print(f"[{args.base_model}] Da merge LoRA vao model goc.", flush=True)
        except Exception as e:
            print(f"[{args.base_model}] Merge LoRA loi ({e}) -> luu adapter rieng.", flush=True)

    model.save_pretrained(args.output_dir)
    meta = {"max_steps": max_steps, "mini_batch_final": mini_batch_size,
            "calib_time_s": calib_time, "elapsed_s": time.time() - t0,
            "lora_used": lora_used, "optim": optim_name}
    with open(args.output_dir + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[{args.base_model}] DONE -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
