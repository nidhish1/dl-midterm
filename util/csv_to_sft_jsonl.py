import argparse
import csv
import json
import random
from pathlib import Path

PROMPT_TEMPLATE = """Generate a valid SVG for the following description.
Return only SVG markup with no explanation.

Description: {description}

SVG:
"""

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT_DIR = _REPO_ROOT / "sft_sft_data"


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"prompt", "svg"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            svg = (row.get("svg") or "").strip()
            if not prompt or not svg:
                continue

            rows.append(
                {
                    "prompt": PROMPT_TEMPLATE.format(description=prompt),
                    "completion": svg,
                }
            )
    return rows


def load_test_prompts(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "prompt"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            rid = (row.get("id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not rid or not prompt:
                continue
            rows.append(
                {
                    "id": rid,
                    "prompt": PROMPT_TEMPLATE.format(description=prompt),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_val", "test"],
        default="train_val",
        help="train_val: write train.jsonl + val.jsonl. test: write single test.jsonl.",
    )
    parser.add_argument("--val_frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    if args.mode == "test":
        rows = load_test_prompts(args.input_csv)
        if not rows:
            raise ValueError("No valid rows found.")
        write_jsonl(out_dir / "test.jsonl", rows)
        print(f"Total rows: {len(rows)}")
        print(f"Wrote: {out_dir / 'test.jsonl'}")
        return

    rows = load_rows(args.input_csv)
    if not rows:
        raise ValueError("No valid rows found.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)

    print(f"Total rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows:   {len(val_rows)}")
    print(f"Wrote: {out_dir / 'train.jsonl'}")
    print(f"Wrote: {out_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
