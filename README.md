# dl-midterm-2 — SVG generation (SFT, inference, submission)

## New server setup

From an empty GPU box (CUDA 12.x assumed for the PyTorch index below). Adjust paths and CUDA wheel if needed.

```bash
git clone <your-repo-url> dl-midterm
cd dl-midterm

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# PyTorch (CUDA 12.1 wheels) + training stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft trl accelerate kaggle sentencepiece protobuf
```

Kaggle CLI (competition data download):

```bash
mkdir -p ~/.kaggle
# Place kaggle.json here, or create from env:
# echo '{"username":"...","key":"..."}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Optional: one-shot bootstrap (venv, deps, download, canonicalize, JSONL, train smoke or full) — edit variables at top of the script first:

```bash
chmod +x bootstrap_h100.sh
# Smoke: TRAIN_MODE=smoke ./bootstrap_h100.sh
# Full:  TRAIN_MODE=full  ./bootstrap_h100.sh
```

---

## Before your next training run (fix data — no GPU time)

High-scoring **`highestnotebook.ipynb`** uses **short prompts** (`test.csv` / `train.csv` text only) and **notebook-style SVG targets** (`viewBox="0 0 256 256"`, **8k** char cap, strip script). The default repo pipeline used a **long prompt template** and often **full `svg_canonicalizer`**, which is a different distribution.

1. **Rebuild JSONL** with aligned prompts and (optionally) notebook-style completions:

```bash
# Train on raw Kaggle prompts + notebook-cleaned SVGs (use *uncanonicalised* train.csv if you want 256/8k targets like the notebook)
python3 util/csv_to_sft_jsonl.py \
  --input_csv dl-spring-2026-svg-generation/train.csv \
  --out_dir sft_sft_data \
  --prompt_format raw \
  --svg_target_clean notebook

# Same prompt style for test inference (required if you trained with --prompt_format raw)
python3 util/csv_to_sft_jsonl.py \
  --mode test \
  --input_csv dl-spring-2026-svg-generation/test.csv \
  --out_dir sft_sft_data \
  --prompt_format raw
```

`--prompt_format template` (default) keeps the old long instruction block. `--svg_target_clean none` (default) leaves SVG strings unchanged.

2. **Optional — no retrain:** patch an existing submission CSV toward the same 256/8k rules:

```bash
python3 util/notebook_enforce_submission_csv.py \
  --in_csv runs/submission.csv \
  --out_csv runs/submission_notebook_rules.csv
```

---

## Data prep (train / val JSONL)

Download and unzip the competition files (from repo root), then canonicalize **train** and build JSONL:

```bash
mkdir -p dl-spring-2026-svg-generation
kaggle competitions download -c dl-spring-2026-svg-generation -p dl-spring-2026-svg-generation
unzip -o dl-spring-2026-svg-generation/*.zip -d dl-spring-2026-svg-generation

python3 svg_canonicalizer.py \
  dl-spring-2026-svg-generation/train.csv \
  dl-spring-2026-svg-generation/train_canonicsed.csv

python3 util/csv_to_sft_jsonl.py \
  --input_csv dl-spring-2026-svg-generation/train_canonicsed.csv \
  --out_dir sft_sft_data
```

**Test prompts** for inference (`test.jsonl`, prompt-only) from `test.csv`:

```bash
python3 util/csv_to_sft_jsonl.py \
  --mode test \
  --input_csv dl-spring-2026-svg-generation/test.csv \
  --out_dir sft_sft_data
```

---

## Training

**Smoke** (small subset, fast):

```bash
source .venv/bin/activate
python3 train.py \
  --train_jsonl sft_sft_data/train.jsonl \
  --val_jsonl sft_sft_data/val.jsonl \
  --output_dir runs/svg_qwen_smoke \
  --max_length 2048 \
  --max_train_samples 512 \
  --max_eval_samples 128 \
  --num_train_epochs 1 \
  --eval_steps 50 \
  --save_steps 50
```

**Full** LoRA SFT (example matching a long-context run: `max_length` 4096, adapter under `runs/svg_qwen_full_len4096/final_adapter`):

```bash
source .venv/bin/activate
python3 train.py \
  --train_jsonl sft_sft_data/train.jsonl \
  --val_jsonl sft_sft_data/val.jsonl \
  --output_dir runs/svg_qwen_full_len4096 \
  --max_length 4096 \
  --load_best_model_at_end
```

Artifacts: checkpoints and logs under `--output_dir`; final adapter at `runs/svg_qwen_full_len4096/final_adapter` (adjust if you change `output_dir`). See `train.py` for batch size, LR, LoRA ranks, `eval_steps`, etc.

---

## Inference (multi-GPU, 4 shards)

Run from repo root with the same flags on four processes (one GPU each). Merge, then optionally post-process.

```bash
cd /path/to/dl-midterm

CUDA_VISIBLE_DEVICES=0 python3 test.py --batch_size 32 --num_shards 4 --shard_index 0 \
  --out_csv runs/submission_shard0.csv --log_path runs/test_infer_shard0.log &
CUDA_VISIBLE_DEVICES=1 python3 test.py --batch_size 32 --num_shards 4 --shard_index 1 \
  --out_csv runs/submission_shard1.csv --log_path runs/test_infer_shard1.log &
CUDA_VISIBLE_DEVICES=2 python3 test.py --batch_size 32 --num_shards 4 --shard_index 2 \
  --out_csv runs/submission_shard2.csv --log_path runs/test_infer_shard2.log &
CUDA_VISIBLE_DEVICES=3 python3 test.py --batch_size 32 --num_shards 4 --shard_index 3 \
  --out_csv runs/submission_shard3.csv --log_path runs/test_infer_shard3.log &
wait
```

Add if needed (defaults in `test.py` may already match):

`--base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_dir runs/svg_qwen_full_len4096/final_adapter --test_jsonl sft_sft_data/test.jsonl`

**Merge** shard CSVs:

```bash
python3 util/merge_submission_shards.py \
  --shards runs/submission_shard0.csv runs/submission_shard1.csv runs/submission_shard2.csv runs/submission_shard3.csv \
  --out runs/submission.csv
wc -l runs/submission.csv   # expect 1001 = header + 1000 rows
```

Single-GPU shortcut: omit sharding and use `--out_csv runs/submission.csv`.

---

## Post-processing (hard validation + safe repairs)

Runs strict checks on each `svg`: first `<svg>…</svg>` block, valid XML, root `<svg>`, length ≤ 16000, path count ≤ 256, allowed tag set (plus common SVG filter primitives). Safe repairs only (trim junk, collapse inter-tag whitespace, drop script/metadata/foreignObject/SMIL, unwrap other disallowed wrappers, missing `xmlns`). Does **not** rewrite path `d`, `viewBox`, or geometry like full `svg_canonicalizer.py` canonicalization.

```bash
cd /path/to/dl-midterm
python3 post_processing/post_process_submission.py \
  --in_csv runs/submission.csv \
  --out_csv runs/submission_post.csv \
  --report_json runs/post_process_report.json
```

**Submit this** if you stop here, or run the next step to match **training** targets.

Alternate entry: `python3 -m post_processing.hard_validate --in_csv … --out_csv …`

### Final step: full canonicalization (same as training `train_canonicsed.csv`)

Hard post-processing does not round paths or strip attrs like `svg_canonicalizer.py`. For distribution alignment with SFT labels, run:

```bash
python3 util/canonicalize_submission_csv.py \
  --in_csv runs/submission_post.csv \
  --out_csv runs/submission_final.csv
wc -l runs/submission_final.csv
```

Upload **`runs/submission_final.csv`** to Kaggle. Rows that fail canonicalization stay as in the input (see stderr warning count).

---

## SVG canonicalizer (train CSV column)

Requires Python 3.

```bash
cd /path/to/dl-midterm-2
python3 svg_canonicalizer.py dl-spring-2026-svg-generation/train.csv dl-spring-2026-svg-generation/train_canonicsed.csv
```

Custom I/O:

```bash
python3 svg_canonicalizer.py path/to/in.csv path/to/out.csv
```

---

## Compare random samples (train vs canonicalised)

Writes `svg_sample_compare.html` in the repo root:

```bash
cd /path/to/dl-midterm-2
python3 compare_svg_samples.py --seed 42 -n 10
```
