"""
Inference from test.jsonl (prompts must match SFT: same PROMPT_TEMPLATE ending with SVG:).

Training used max_length=4096 for prompt+completion *together*. At inference, `max_new_tokens`
is only the *new* completion; you can set it high (e.g. 2048) if SVGs are long.

- num_candidates=1: greedy by default → writes submission CSV (one SVG per id).
- num_candidates>1: writes JSONL only (id, candidate_idx, svg, reason). No selection here —
  pick / canonicalize / validate later. Stochastic sampling is auto-enabled so candidates differ.
"""
from __future__ import annotations

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


def postprocess_svg(raw: str, canonicalize: bool) -> tuple[str, str]:
    svg = extract_svg(raw)
    if svg == FALLBACK_SVG:
        return FALLBACK_SVG, "extract_fallback"

    if not canonicalize:
        return svg, "ok"

    try:
        from svg_canonicalizer import canonicalize_row_svg

        out = canonicalize_row_svg(svg)
        if not out or not out.startswith("<svg"):
            return FALLBACK_SVG, "canon_fallback"
        return out, "ok"
    except Exception:
        return FALLBACK_SVG, "canon_fallback"


def run_generate(
    model,
    tok,
    dev,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(dev) for k, v in enc.items()}

    gen_kwargs = {
        **enc,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        gen = model.generate(**gen_kwargs)

    in_len = enc["input_ids"].shape[1]
    return tok.batch_decode(gen[:, in_len:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate submission CSV and/or candidate JSONL from test.jsonl."
    )
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_dir", type=Path, default=Path("runs/svg_qwen_full_len4096/final_adapter"))
    ap.add_argument("--test_jsonl", type=Path, default=Path("sft_sft_data/test.jsonl"))
    ap.add_argument("--out_csv", type=Path, default=Path("runs/submission.csv"))
    ap.add_argument("--log_path", type=Path, default=Path("runs/test_infer.log"))

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max *new* tokens per completion. Training max_length=4096 was prompt+completion; "
        "2048 is a strong default for long SVGs (tune 1500–3000 if needed).",
    )
    ap.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="K>1: save K samples per id to --candidates_out (no CSV pick). Greedy repeats are identical, "
        "so sampling is auto-enabled for K>1.",
    )
    ap.add_argument(
        "--candidates_out",
        type=Path,
        default=None,
        help="JSONL path. Default: runs/candidates.jsonl when num_candidates>1. "
        "Also written for num_candidates==1 if you set this path.",
    )
    ap.add_argument(
        "--do_sample",
        action="store_true",
        help="Stochastic decoding. Default False when num_candidates=1 (greedy).",
    )
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument(
        "--canonicalize",
        action="store_true",
        help="Run svg_canonicalizer after extract (optional; usually do this in post-process).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Only first N examples.")
    args = ap.parse_args()

    if args.num_candidates < 1:
        raise ValueError("--num_candidates must be >= 1")

    candidates_path = args.candidates_out
    if args.num_candidates > 1 and candidates_path is None:
        candidates_path = Path("runs/candidates.jsonl")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    if candidates_path is not None:
        candidates_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with args.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    if args.log_path.exists():
        args.log_path.unlink()

    do_sample = args.do_sample
    temperature = args.temperature
    top_p = args.top_p

    if args.num_candidates > 1:
        if not do_sample:
            log("num_candidates>1: enabling do_sample=True (greedy would duplicate K outputs).")
            do_sample = True
        if temperature <= 0:
            temperature = 0.8
            log(f"Using temperature={temperature} for diverse candidates.")

    log("=== start ===")
    log("Prompts in test.jsonl must match training (instruction + 'SVG:' tail).")
    log(f"num_candidates={args.num_candidates} max_new_tokens={args.max_new_tokens} do_sample={do_sample}")
    log(f"adapter_dir={args.adapter_dir} test_jsonl={args.test_jsonl}")
    if candidates_path:
        log(f"candidates_out={candidates_path}")
    if args.num_candidates == 1:
        log(f"out_csv={args.out_csv} (single candidate → submission CSV)")
    else:
        log("No submission CSV this run; use candidates JSONL only (pick later).")

    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter_dir: {args.adapter_dir}")
    if not args.test_jsonl.exists():
        raise FileNotFoundError(f"Missing test_jsonl: {args.test_jsonl}")

    log(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    model.eval()
    dev = next(model.parameters()).device
    log(f"model device={dev}")

    total = 0
    wrote_csv = 0
    wrote_cand = 0
    n_extract_fallback = 0
    n_canon_fallback = 0
    start = time.time()

    csv_file = None
    csv_writer = None
    if args.num_candidates == 1:
        csv_file = args.out_csv.open("w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["id", "svg"])
        csv_writer.writeheader()
        csv_file.flush()

    cand_file = None
    if candidates_path is not None:
        cand_file = candidates_path.open("w", encoding="utf-8")

    try:
        f_in = args.test_jsonl.open("r", encoding="utf-8")
        batch: list[dict] = []

        def flush_batch(rows: list[dict]) -> None:
            nonlocal wrote_csv, wrote_cand, n_extract_fallback, n_canon_fallback
            if not rows:
                return
            prompts = [r["prompt"] for r in rows]
            for k in range(args.num_candidates):
                comps = run_generate(
                    model,
                    tok,
                    dev,
                    prompts,
                    args.max_new_tokens,
                    do_sample,
                    temperature,
                    top_p,
                )
                for r, c in zip(rows, comps):
                    svg, reason = postprocess_svg(c, args.canonicalize)
                    if reason == "extract_fallback":
                        n_extract_fallback += 1
                    elif reason == "canon_fallback":
                        n_canon_fallback += 1

                    if cand_file is not None:
                        cand_file.write(
                            json.dumps(
                                {
                                    "id": r["id"],
                                    "candidate_idx": k,
                                    "svg": svg,
                                    "reason": reason,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        wrote_cand += 1

                    if args.num_candidates == 1 and csv_writer is not None and k == 0:
                        csv_writer.writerow({"id": r["id"], "svg": svg})
                        wrote_csv += 1

            if csv_file:
                csv_file.flush()
            if cand_file:
                cand_file.flush()

        for line in f_in:
            if args.limit and total >= args.limit:
                break
            batch.append(json.loads(line))
            total += 1

            if len(batch) >= args.batch_size:
                flush_batch(batch)
                batch = []
                if wrote_csv and wrote_csv % 1000 == 0 and args.num_candidates == 1:
                    log(f"csv progress wrote_csv={wrote_csv} rows/s={wrote_csv / max(time.time() - start, 1e-6):.2f}")
                if wrote_cand and wrote_cand % 5000 == 0 and cand_file:
                    log(f"cand progress lines={wrote_cand} ...")

        flush_batch(batch)
        f_in.close()

    finally:
        if csv_file:
            csv_file.close()
        if cand_file:
            cand_file.close()

    elapsed = time.time() - start
    log("=== done ===")
    log(
        f"examples_read={total} csv_rows={wrote_csv} candidate_lines={wrote_cand} "
        f"extract_fallback={n_extract_fallback} canon_fallback={n_canon_fallback} "
        f"sec={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
