#!/usr/bin/env python3
"""Pick random rows from train.csv vs train_canonicsed.csv and write an HTML preview."""

from __future__ import annotations

import argparse
import csv
import html
import random
from difflib import unified_diff
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parent
DATA = REPO / "dl-spring-2026-svg-generation"


def load_svg_by_id(csv_path: Path) -> Dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: (row.get("svg") or "") for row in csv.DictReader(f) if row.get("id")}


def build_html(rows: list[tuple[str, str, str, str]], title: str) -> str:
    chunks = []
    for sample_i, (rid, prompt, raw, canon) in enumerate(rows):
        same = raw == canon
        diff_lines = list(
            unified_diff(
                raw.splitlines(),
                canon.splitlines(),
                fromfile="train",
                tofile="canonical",
                lineterm="",
            )
        )
        diff_preview = html.escape("\n".join(diff_lines[:120]))
        if len(diff_lines) > 120:
            diff_preview += "\n…"

        pre_max = 12_000
        raw_pre = html.escape(raw[:pre_max] + ("…" if len(raw) > pre_max else ""))
        canon_pre = html.escape(canon[:pre_max] + ("…" if len(canon) > pre_max else ""))

        chunks.append(
            f"""
<section class="sample">
  <h2>Sample {sample_i + 1} <code>{html.escape(rid)}</code></h2>
  <p class="prompt">{html.escape(prompt[:500])}{"…" if len(prompt) > 500 else ""}</p>
  <p class="meta">
    Identical: <strong>{"yes" if same else "no"}</strong> ·
    len train={len(raw)} · len canonical={len(canon)}
  </p>
  <div class="row">
    <div class="cell">
      <h3>Train (rendered)</h3>
      <div class="host">{raw}</div>
    </div>
    <div class="cell">
      <h3>Canonical (rendered)</h3>
      <div class="host">{canon}</div>
    </div>
  </div>
  <details>
    <summary>Unified diff (first lines)</summary>
    <pre class="diff">{diff_preview}</pre>
  </details>
  <details>
    <summary>Raw SVG text (truncated)</summary>
    <div class="pregrid">
      <pre class="code">{raw_pre}</pre>
      <pre class="code">{canon_pre}</pre>
    </div>
  </details>
</section>
"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 1200px; }}
    h1 {{ font-size: 1.25rem; }}
    .sample {{ border-top: 1px solid #ccc; padding: 1.5rem 0; }}
    .prompt {{ color: #444; font-size: 0.9rem; }}
    .meta {{ font-size: 0.85rem; color: #333; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0; }}
    .cell {{ flex: 1; min-width: 260px; }}
    .cell h3 {{ margin: 0 0 8px; font-size: 0.95rem; }}
    .host {{
      width: 280px; height: 280px; border: 1px solid #bbb;
      display: flex; align-items: center; justify-content: center;
      background: #fafafa; overflow: hidden;
    }}
    .host svg {{ max-width: 100%; max-height: 100%; }}
    details {{ margin-top: 10px; }}
    pre.diff, pre.code {{
      background: #f4f4f4; padding: 10px; overflow: auto;
      font-size: 11px; line-height: 1.35; border: 1px solid #ddd;
    }}
    .pregrid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    @media (max-width: 800px) {{ .pregrid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Open this file in a browser. Each pair renders the SVG as the browser would draw it.</p>
  {"".join(chunks)}
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare random train vs canonicalised SVG rows.")
    ap.add_argument("--train", type=Path, default=DATA / "train.csv")
    ap.add_argument("--canonical", type=Path, default=DATA / "train_canonicsed.csv")
    ap.add_argument("-o", "--output", type=Path, default=REPO / "svg_sample_compare.html")
    ap.add_argument("-n", "--count", type=int, default=6, help="Number of random samples.")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    args = ap.parse_args()

    train_path: Path = args.train
    canon_path: Path = args.canonical
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}")
    if not canon_path.is_file():
        raise SystemExit(f"Missing {canon_path}")

    train_map = load_svg_by_id(train_path)
    canon_map = load_svg_by_id(canon_path)
    common = sorted(set(train_map) & set(canon_map))
    if not common:
        raise SystemExit("No overlapping ids between the two CSVs.")

    rng = random.Random(args.seed)
    k = min(args.count, len(common))
    picked = rng.sample(common, k)

    rows: list[tuple[str, str, str, str]] = []
    with train_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        id_to_prompt = {row["id"]: (row.get("prompt") or "") for row in reader if row.get("id")}
    for rid in picked:
        rows.append(
            (
                rid,
                id_to_prompt.get(rid, ""),
                train_map[rid],
                canon_map[rid],
            )
        )

    title = f"SVG compare ({k} samples, seed={args.seed})"
    html_out = build_html(rows, title)
    args.output.write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
