import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


FALLBACK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"></svg>'


def extract_svg(text: str) -> str:
    m = re.search(r"<svg[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return FALLBACK_SVG


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate submission.csv from test.jsonl using a LoRA adapter.")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_dir", type=Path, default=Path("runs/svg_qwen_full_len4096/final_adapter"))
    ap.add_argument("--test_jsonl", type=Path, default=Path("sft_sft_data/test.jsonl"))
    ap.add_argument("--out_csv", type=Path, default=Path("runs/submission.csv"))
    ap.add_argument("--log_path", type=Path, default=Path("runs/test_infer.log"))

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1200)

    # If temperature == 0 -> deterministic (no sampling)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--limit", type=int, default=0, help="Optional: only run first N examples.")
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with args.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    if args.log_path.exists():
        args.log_path.unlink()

    log("=== start ===")
    log(f"base_model={args.base_model}")
    log(f"adapter_dir={args.adapter_dir} exists={args.adapter_dir.exists()}")
    log(f"test_jsonl={args.test_jsonl} exists={args.test_jsonl.exists()}")
    log(f"out_csv={args.out_csv}")

    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter_dir: {args.adapter_dir}")
    if not args.test_jsonl.exists():
        raise FileNotFoundError(f"Missing test_jsonl: {args.test_jsonl}")

    log(f"torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
        log(f"bf16_supported={torch.cuda.is_bf16_supported()}")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    log(f"dtype={dtype}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    log(f"tokenizer_loaded_sec={time.time() - t0:.1f}")

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    model.eval()
    dev = next(model.parameters()).device
    log(f"model_loaded_sec={time.time() - t0:.1f} device={dev}")

    do_sample = args.temperature > 0.0
    log(f"batch_size={args.batch_size} max_new_tokens={args.max_new_tokens} do_sample={do_sample} temp={args.temperature} top_p={args.top_p}")

    total = 0
    wrote = 0
    fallback = 0
    start = time.time()

    with args.test_jsonl.open("r", encoding="utf-8") as f_in, args.out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["id", "svg"])
        writer.writeheader()
        f_out.flush()

        batch = []
        for line in f_in:
            if args.limit and total >= args.limit:
                break
            row = json.loads(line)
            batch.append(row)
            total += 1

            if len(batch) < args.batch_size:
                continue

            prompts = [r["prompt"] for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(dev) for k, v in enc.items()}

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            in_len = enc["input_ids"].shape[1]
            comps = tok.batch_decode(gen[:, in_len:], skip_special_tokens=True)

            out_rows = []
            for r, c in zip(batch, comps):
                svg = extract_svg(c)
                if svg == FALLBACK_SVG:
                    fallback += 1
                out_rows.append({"id": r["id"], "svg": svg})

            writer.writerows(out_rows)
            wrote += len(out_rows)
            f_out.flush()
            batch = []

            if wrote % 1000 == 0:
                elapsed = time.time() - start
                rps = wrote / max(elapsed, 1e-6)
                log(f"progress wrote={wrote} fallback={fallback} rows_per_sec={rps:.2f}")

        # remainder
        if batch:
            prompts = [r["prompt"] for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(dev) for k, v in enc.items()}

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            in_len = enc["input_ids"].shape[1]
            comps = tok.batch_decode(gen[:, in_len:], skip_special_tokens=True)
            out_rows = []
            for r, c in zip(batch, comps):
                svg = extract_svg(c)
                if svg == FALLBACK_SVG:
                    fallback += 1
                out_rows.append({"id": r["id"], "svg": svg})
            writer.writerows(out_rows)
            wrote += len(out_rows)
            f_out.flush()

    elapsed = time.time() - start
    log("=== done ===")
    log(f"read_total={total} wrote={wrote} fallback={fallback} sec={elapsed:.1f} rows_per_sec={wrote / max(elapsed, 1e-6):.2f}")
    log(f"out_csv={args.out_csv}")
    log(f"log_path={args.log_path}")


if __name__ == "__main__":
    main()
