"""
Inference from test.jsonl (prompt text must match training: template vs raw vs chat-wrapped).

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


def postprocess_svg(raw: str, canonicalize: bool, notebook_svg_post: bool = False) -> tuple[str, str]:
    svg = extract_svg(raw)
    if svg == FALLBACK_SVG:
        return FALLBACK_SVG, "extract_fallback"

    if notebook_svg_post:
        from util.notebook_svg_constraints import (
            NOTEBOOK_PLACEHOLDER_SVG,
            enforce_svg_constraints,
            is_valid_svg,
        )

        svg = enforce_svg_constraints(svg)
        if not is_valid_svg(svg):
            return NOTEBOOK_PLACEHOLDER_SVG, "notebook_placeholder"
        return svg, "ok"

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
    ap.add_argument(
        "--legacy_plain_prompt",
        action="store_true",
        help="Tokenize prompts as raw strings (matches old train.py before chat template). "
        "Omit this for Qwen-Instruct models trained with current train.py (chat formatting).",
    )
    ap.add_argument(
        "--notebook_svg_post",
        action="store_true",
        help="After <svg> extract: apply highestnotebook-style enforce (256/8k/paths); "
        "if invalid, use the notebook's green-circle placeholder SVG.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Only first N examples.")
    ap.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split test.jsonl by line index i: keep lines where i %% num_shards == shard_index. "
        "Run 4 jobs with shard_index 0..3 and different CUDA_VISIBLE_DEVICES to use all GPUs.",
    )
    ap.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="This worker's shard (0 .. num_shards-1).",
    )
    ap.add_argument(
        "--log_every_candidates",
        type=int,
        default=250,
        help="When writing candidates JSONL, log progress every N lines (0 = only per-batch + final).",
    )
    args = ap.parse_args()

    if args.num_candidates < 1:
        raise ValueError("--num_candidates must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

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
    if args.legacy_plain_prompt:
        log("Prompts: legacy plain strings (must match how the adapter was trained).")
    else:
        log("Prompts: Qwen chat template (user message only); match train.py + jsonl text.")
    log(f"num_candidates={args.num_candidates} max_new_tokens={args.max_new_tokens} do_sample={do_sample}")
    if args.notebook_svg_post:
        log("notebook_svg_post=True (256/8k/path rules + notebook placeholder on invalid).")
    if args.canonicalize and args.notebook_svg_post:
        log("note: --notebook_svg_post runs instead of --canonicalize per row.")
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

    with args.test_jsonl.open("r", encoding="utf-8") as _cf:
        n_lines_in_file = sum(1 for _ in _cf)

    cap = min(n_lines_in_file, args.limit) if args.limit else n_lines_in_file
    if args.num_shards > 1:
        n_prompts_target = sum(1 for i in range(cap) if i % args.num_shards == args.shard_index)
        log(
            f"sharding: num_shards={args.num_shards} shard_index={args.shard_index} "
            f"cap_lines={cap} this_shard_prompts={n_prompts_target}"
        )
    else:
        n_prompts_target = cap

    if args.num_candidates > 1:
        expected_cand_lines = n_prompts_target * args.num_candidates
        log(
            f"dataset: prompts_in_file={n_lines_in_file} cap={cap} will_process_this_run={n_prompts_target} "
            f"num_candidates={args.num_candidates} expected_candidate_lines={expected_cand_lines}"
        )
    else:
        log(
            f"dataset: prompts_in_file={n_lines_in_file} cap={cap} will_process_this_run={n_prompts_target} "
            f"(submission csv rows this run={n_prompts_target})"
        )

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

    def format_prompts_for_inference(raw_prompts: list[str]) -> list[str]:
        if args.legacy_plain_prompt:
            return raw_prompts
        out = []
        for p in raw_prompts:
            out.append(
                tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return out

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    model.eval()
    dev = next(model.parameters()).device
    log(f"model device={dev}")
    log(
        "Inference starting: first [submission] log line appears after the first generate() batch "
        f"(batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}; often a few minutes)."
    )

    total = 0
    wrote_csv = 0
    wrote_cand = 0
    prompts_flushed = 0
    n_extract_fallback = 0
    n_canon_fallback = 0
    n_notebook_placeholder = 0
    start = time.time()
    last_cand_log_milestone = 0

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
            nonlocal wrote_csv, wrote_cand, prompts_flushed
            nonlocal n_extract_fallback, n_canon_fallback, n_notebook_placeholder
            nonlocal last_cand_log_milestone
            if not rows:
                return
            raw_prompts = [r["prompt"] for r in rows]
            prompts = format_prompts_for_inference(raw_prompts)
            batch_t0 = time.time()
            for k in range(args.num_candidates):
                k_t0 = time.time()
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
                    svg, reason = postprocess_svg(c, args.canonicalize, args.notebook_svg_post)
                    if reason == "extract_fallback":
                        n_extract_fallback += 1
                    elif reason == "canon_fallback":
                        n_canon_fallback += 1
                    elif reason == "notebook_placeholder":
                        n_notebook_placeholder += 1

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
                        if args.log_every_candidates > 0 and wrote_cand - last_cand_log_milestone >= args.log_every_candidates:
                            last_cand_log_milestone = wrote_cand
                            elapsed = time.time() - start
                            exp = n_prompts_target * args.num_candidates
                            pct = 100.0 * wrote_cand / max(exp, 1)
                            log(
                                f"[candidates] lines={wrote_cand}/{exp} ({pct:.1f}%) "
                                f"lines/s={wrote_cand / max(elapsed, 1e-6):.2f} "
                                f"prompts_done~={wrote_cand // max(args.num_candidates, 1)}/{n_prompts_target} "
                                f"extract_fb={n_extract_fallback} canon_fb={n_canon_fallback} "
                                f"nb_placeholder={n_notebook_placeholder}"
                            )

                    if args.num_candidates == 1 and csv_writer is not None and k == 0:
                        csv_writer.writerow({"id": r["id"], "svg": svg})
                        wrote_csv += 1

                if args.num_candidates > 1:
                    log(
                        f"[candidates] finished candidate_idx={k + 1}/{args.num_candidates} "
                        f"for_batch={len(rows)} prompts gen_step_sec={time.time() - k_t0:.2f}"
                    )

            prompts_flushed += len(rows)
            if csv_file:
                csv_file.flush()
            if cand_file:
                cand_file.flush()

            if args.num_candidates == 1 and csv_writer is not None:
                elapsed = time.time() - start
                small_run = n_prompts_target <= 2500
                if small_run or wrote_csv % 1000 == 0 or wrote_csv >= n_prompts_target:
                    log(
                        f"[submission] csv_rows={wrote_csv}/{n_prompts_target} "
                        f"rows/s={wrote_csv / max(elapsed, 1e-6):.2f} "
                        f"last_batch_size={len(rows)} last_batch_sec={time.time() - batch_t0:.2f}"
                    )

            if args.num_candidates > 1 and cand_file is not None:
                elapsed = time.time() - start
                exp = n_prompts_target * args.num_candidates
                log(
                    f"[candidates] batch_done prompts_total={prompts_flushed}/{n_prompts_target} "
                    f"lines_total={wrote_cand}/{exp} batch_wall_sec={time.time() - batch_t0:.2f} "
                    f"overall_lines/s={wrote_cand / max(elapsed, 1e-6):.2f}"
                )

        line_idx = 0
        for line in f_in:
            if args.limit and line_idx >= args.limit:
                break
            row_obj = json.loads(line)
            if args.num_shards > 1 and line_idx % args.num_shards != args.shard_index:
                line_idx += 1
                continue
            batch.append(row_obj)
            line_idx += 1
            total += 1

            if len(batch) >= args.batch_size:
                flush_batch(batch)
                batch = []

        flush_batch(batch)
        f_in.close()

    finally:
        if csv_file:
            csv_file.close()
        if cand_file:
            cand_file.close()

    elapsed = time.time() - start
    log("=== done ===")
    exp_cand = n_prompts_target * args.num_candidates if args.num_candidates > 1 else n_prompts_target
    rate = (wrote_cand if args.num_candidates > 1 else wrote_csv) / max(elapsed, 1e-6)
    log(
        f"examples_read={total} prompts_target={n_prompts_target} "
        f"csv_rows={wrote_csv} candidate_lines={wrote_cand} expected_lines={exp_cand} "
        f"extract_fallback={n_extract_fallback} canon_fallback={n_canon_fallback} "
        f"notebook_placeholder={n_notebook_placeholder} "
        f"sec={elapsed:.1f} throughput_lines_or_rows_per_sec={rate:.2f}"
    )


if __name__ == "__main__":
    main()
