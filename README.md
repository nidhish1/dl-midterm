# SVG canonicalizer

Canonicalizes the `svg` column in a CSV. Requires Python 3.

**Defaults** (from repo root): reads `dl-spring-2026-svg-generation/train.csv`, writes `dl-spring-2026-svg-generation/train_canonicsed.csv`.

```bash
cd /path/to/dl-midterm-2
python3 svg_canonicalizer.py dl-spring-2026-svg-generation/train.csv dl-spring-2026-svg-generation/train_canonicsed.csv
```

Custom input/output:

```bash
python3 svg_canonicalizer.py path/to/in.csv path/to/out.csv
```

## SFT JSONL (from canonicalised CSV)

Writes `sft_sft_data/train.jsonl` and `sft_sft_data/val.jsonl` by default:

```bash
cd /Users/mudrex/Desktop/dl-midterm-2
python3 util/csv_to_sft_jsonl.py --input_csv dl-spring-2026-svg-generation/train_canonicsed.csv
```

## Compare random samples (train vs canonicalised)

Writes `svg_sample_compare.html` in the repo root (open in a browser):

```bash
cd /path/to/dl-midterm-2
python3 compare_svg_samples.py --seed 42 -n 10
```

