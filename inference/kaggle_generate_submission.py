import argparse
import csv
import re
import tarfile
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """Generate a valid SVG for the following description.
Return only SVG markup with no explanation.

Description: {description}

SVG:
"""


def maybe_extract_adapter(adapter_tar: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(adapter_tar, "r:gz") as tf:
        tf.extractall(extract_to)

    # Common output shape: extract_to/checkpoint-XXXX
    candidates = [p for p in extract_to.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return extract_to


def extract_svg(text: str) -> str:
    m = re.search(r"<svg[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"></svg>'


def read_test_rows(test_csv: Path) -> List[dict]:
    rows = []
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError("test.csv must have columns: id,prompt")
        for row in reader:
            rows.append({"id": row["id"], "prompt": row["prompt"]})
    return rows


def write_submission(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "svg"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission from LoRA adapter.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_dir", type=Path, default=None, help="Path to extracted adapter/checkpoint")
    parser.add_argument("--adapter_tar", type=Path, default=None, help="Optional checkpoint tar.gz to extract")
    parser.add_argument("--extract_to", type=Path, default=Path("/kaggle/working/adapter_extracted"))
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path("/kaggle/input/dl-spring-2026-svg-generation/test.csv"),
    )
    parser.add_argument("--output_csv", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    if args.adapter_dir is None and args.adapter_tar is None:
        raise ValueError("Provide either --adapter_dir or --adapter_tar")
    if args.adapter_tar is not None:
        args.adapter_dir = maybe_extract_adapter(args.adapter_tar, args.extract_to)
        print(f"Using extracted adapter dir: {args.adapter_dir}")

    if args.adapter_dir is None or not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {args.adapter_dir}")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.eval()

    rows = read_test_rows(args.test_csv)
    outputs = []

    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        prompts = [PROMPT_TEMPLATE.format(description=r["prompt"]) for r in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            gen = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature if args.temperature > 0.0 else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = encoded["input_ids"].shape[1]
        completions = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)

        for row, comp in zip(chunk, completions):
            outputs.append({"id": row["id"], "svg": extract_svg(comp)})

        if (i // args.batch_size) % 20 == 0:
            print(f"Processed {min(i + args.batch_size, len(rows))}/{len(rows)}")

    write_submission(args.output_csv, outputs)
    print(f"Wrote submission: {args.output_csv}")


if __name__ == "__main__":
    main()
