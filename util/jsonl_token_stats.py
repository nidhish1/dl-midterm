import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def pctl(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    lengths = []
    with args.jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            row = json.loads(line)
            text = row["prompt"] + row["completion"]
            toks = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            lengths.append(len(toks))

    lengths.sort()

    if not lengths:
        raise ValueError("No examples found.")

    print(f"count   : {len(lengths)}")
    print(f"min     : {lengths[0]}")
    print(f"mean    : {statistics.mean(lengths):.1f}")
    print(f"median  : {statistics.median(lengths):.1f}")
    print(f"p90     : {pctl(lengths, 0.90)}")
    print(f"p95     : {pctl(lengths, 0.95)}")
    print(f"p99     : {pctl(lengths, 0.99)}")
    print(f"max     : {lengths[-1]}")


if __name__ == "__main__":
    main()
