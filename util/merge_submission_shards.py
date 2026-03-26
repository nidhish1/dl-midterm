#!/usr/bin/env python3
"""Merge shard CSVs (id,svg) into one submission; dedupe by id, last file wins if overlap."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, nargs="+", required=True, help="submission_shard*.csv files")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_id: dict[str, str] = {}
    for p in args.shards:
        with p.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                by_id[row["id"]] = row["svg"]

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "svg"])
        w.writeheader()
        for rid in sorted(by_id.keys()):
            w.writerow({"id": rid, "svg": by_id[rid]})

    print(f"Merged {len(by_id)} rows -> {args.out}")


if __name__ == "__main__":
    main()
