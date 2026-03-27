# Notebook-aligned experiment (`highestnotebook.ipynb` vs this repo)

## What the Kaggle notebook actually does

| Stage | Notebook behavior |
|--------|-------------------|
| **Data** | `train.csv` / `test.csv` from competition. **No** HF tokenizer, **no** Qwen. |
| **Targets** | `enforce_svg_constraints`: force `viewBox="0 0 256 256"`, `width/height` 256, `xmlns`, strip `<script>`, **truncate 8000** chars + append `</svg>` if needed. |
| **Filter** | `is_valid_svg`: `<svg` prefix, len ≤ 8000, `<path` count ≤ 256. |
| **Train pool** | Only valid rows; **random sample `sample_size` (3000)** with `random_state=42`. |
| **LM** | **~2.66M param** char-level GPT (`d_model=256`, 3 layers, seq **512**), vocab ~**180** chars. |
| **Train objective** | Batches provide `input_ids` = encode(`<SOS>{prompt}<EOS>`) and `target_ids` = encode(SVG), both padded to 512. Loss is `CrossEntropy(logits(model(input_ids)), target_ids)`. **Position-wise supervision between two unrelated padded sequences is not standard next-token prediction**; treat notebook training as a loose baseline, not a spec to copy literally. |
| **Optim** | AdamW lr **1e-3**, cosine schedule, **3 epochs**, batch **4**, CPU in template. |
| **Inference** | `generate_svg`: **sampling** with **temperature=0.7**, up to **250 new characters**; then same `enforce_svg_constraints` + validity; else **fixed placeholder** SVG (gray rect + green circle + “Generated”). |
| **Submission** | `submission.csv`; rows re-checked with `is_valid_svg`, invalid rows replaced with simple gray rect SVG. |

## What this repo does (Qwen + LoRA)

| Stage | Repo behavior |
|--------|----------------|
| **Data** | `util/csv_to_sft_jsonl.py` → `prompt` + `completion` in JSONL. |
| **Prompts** | `template` (long instruction) or **`raw`** (Kaggle prompt only, notebook-like). |
| **Targets** | **`none`** (raw SVG), **`notebook`** (same constraints as notebook when result stays valid), or canonicalize via separate `svg_canonicalizer.py` CSV. |
| **Train** | **Qwen2.5-Instruct** + PEFT LoRA; examples → **chat** string via `apply_chat_template` → column **`text`**; **`completion_only_loss` off** by default (avoids TRL/tokenizer mismatch). |
| **Inference** | `test.py`: chat-wrapped prompts (unless `--legacy_plain_prompt`), `generate`, extract `<svg>...</svg>`, optional **`--notebook_svg_post`** (enforce + notebook placeholder), or `--canonicalize`. |

## Recommended “redo everything” run (notebook-aligned *distribution*, correct LM)

Do these in order on a machine with data paths set (e.g. `../dl-spring-2026-svg-generation/train.csv`).

### 1) JSONL (raw prompts + notebook-clean targets)

```bash
cd ~/dl-midterm   # or dl-midterm-2

python3 util/csv_to_sft_jsonl.py \
  --input_csv ../dl-spring-2026-svg-generation/train.csv \
  --out_dir sft_sft_data \
  --prompt_format raw \
  --svg_target_clean notebook

python3 util/csv_to_sft_jsonl.py \
  --mode test \
  --input_csv ../dl-spring-2026-svg-generation/test.csv \
  --out_dir sft_sft_data \
  --prompt_format raw
```

### 2) Train (full data or ablate like notebook)

Full run:

```bash
python3 train.py \
  --train_jsonl sft_sft_data/train.jsonl \
  --val_jsonl sft_sft_data/val.jsonl \
  --output_dir runs/exp_notebook_align_full \
  --max_length 2048
```

Notebook-sized ablation (~3k examples):

```bash
python3 train.py \
  --train_jsonl sft_sft_data/train.jsonl \
  --val_jsonl sft_sft_data/val.jsonl \
  --output_dir runs/exp_notebook_align_3k \
  --max_length 2048 \
  --max_train_samples 3000 \
  --num_train_epochs 3 \
  --eval_steps 100 \
  --save_steps 100
```

Ensure `--save_steps` is a multiple of `--eval_steps` if you use `--load_best_model_at_end`.

### 3) Inference (notebook-like decode + post)

Notebook used **sampling** and **~250 characters** of completion; subword models need more **tokens**. Try:

```bash
python3 test.py \
  --adapter_dir runs/exp_notebook_align_full/final_adapter \
  --test_jsonl sft_sft_data/test.jsonl \
  --out_csv runs/submission.csv \
  --do_sample --temperature 0.7 --top_p 0.95 \
  --max_new_tokens 512 \
  --notebook_svg_post
```

Greedy A/B: omit `--do_sample` and keep `--max_new_tokens 1024` or `2048`.

### 4) Optional extra CSV steps

- Strict XML / 16k / tags: `post_processing/post_process_submission.py`
- Full `svg_canonicalizer` on CSV: `util/canonicalize_submission_csv.py` (training-distribution match if you trained on canonicalized data—**not** this notebook-target run)

### 5) What we are **not** reproducing

- Char-vocab tiny GPT (different inductive bias).
- Notebook’s exact **broken-looking** supervised loss pairing (we use standard causal LM on chat `text`).
- CPU-only 512-token cap (Qwen run uses `max_length` on **BPE** pieces).

## Files touched for parity helpers

- `util/notebook_svg_constraints.py` — `enforce_svg_constraints`, `is_valid_svg`, `NOTEBOOK_PLACEHOLDER_SVG`
- `test.py` — `--notebook_svg_post`
- `util/csv_to_sft_jsonl.py` — `--prompt_format raw`, `--svg_target_clean notebook`
- `train.py` — chat `text` column + `dataset_text_field`

## Success criteria

- No TRL “tokenized prompt mismatch” during training.
- LB improvement vs previous: compare greedy vs `--do_sample --temperature 0.7` and with/without `--notebook_svg_post` on a small shard first.
