#!/usr/bin/env python3
"""Apply highestnotebook-style enforce_svg_constraints to an id,svg submission CSV (no training needed)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from util.notebook_svg_constraints import enforce_svg_constraints, is_valid_svg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    with args.in_csv.open(newline="", encoding="utf-8") as f_in:
        r = csv.DictReader(f_in)
        if r.fieldnames is None or "svg" not in r.fieldnames:
            raise SystemExit("CSV needs an svg column")
        fieldnames = list(r.fieldnames)
        rows = list(r)

    n_swap = 0
    for row in rows:
        raw = row.get("svg") or ""
        cand = enforce_svg_constraints(raw)
        if is_valid_svg(cand):
            if cand != raw:
                n_swap += 1
            row["svg"] = cand

    with args.out_csv.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out_csv}  rows={len(rows)}  svg_rewrites={n_swap}")


if __name__ == "__main__":
    main()
