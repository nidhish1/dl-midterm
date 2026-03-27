#!/usr/bin/env python3
"""
Run the same svg_canonicalizer pipeline as training (canonicalize_row_svg per row).

Typical flow: submission.csv → post_processing → submission_post.csv → this → submission_final.csv

Requires columns at least: id, svg (preserves all input columns; rewrites svg).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svg_canonicalizer import process_train_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonicalize SVG column to match training targets.")
    ap.add_argument("--in_csv", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--decimals", type=int, default=2, help="Numeric rounding (same as training canonicalizer).")
    args = ap.parse_args()

    if not args.in_csv.exists():
        raise SystemExit(f"Missing {args.in_csv}")

    process_train_csv(args.in_csv, args.out_csv, decimals=args.decimals)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
