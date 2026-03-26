#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
REPO_DIR="${REPO_DIR:-$PWD}"
COMPETITION="${COMPETITION:-dl-spring-2026-svg-generation}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

# train mode: "smoke" or "full"
TRAIN_MODE="${TRAIN_MODE:-smoke}"

# Kaggle auth:
# Option A (recommended): put kaggle.json at ~/.kaggle/kaggle.json before running.
# Option B: export KAGGLE_USERNAME and KAGGLE_KEY env vars before running.
# ---------------------------

cd "$REPO_DIR"

echo "==> Creating venv"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing Python packages"
# PyTorch (CUDA build) + training stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft trl accelerate kaggle sentencepiece protobuf

echo "==> Setting up Kaggle credentials"
mkdir -p ~/.kaggle
if [[ -f ~/.kaggle/kaggle.json ]]; then
  chmod 600 ~/.kaggle/kaggle.json
elif [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  cat > ~/.kaggle/kaggle.json <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
  chmod 600 ~/.kaggle/kaggle.json
else
  echo "ERROR: Kaggle credentials not found."
  echo "Either place ~/.kaggle/kaggle.json or export KAGGLE_USERNAME and KAGGLE_KEY."
  exit 1
fi

echo "==> Downloading Kaggle competition data"
mkdir -p dl-spring-2026-svg-generation
kaggle competitions download -c "$COMPETITION" -p dl-spring-2026-svg-generation

echo "==> Unzipping data"
unzip -o dl-spring-2026-svg-generation/*.zip -d dl-spring-2026-svg-generation

echo "==> Canonicalizing train.csv"
python svg_canonicalizer.py \
  dl-spring-2026-svg-generation/train.csv \
  dl-spring-2026-svg-generation/train_canonicsed.csv

echo "==> Building SFT JSONL"
python util/csv_to_sft_jsonl.py \
  --input_csv dl-spring-2026-svg-generation/train_canonicsed.csv \
  --out_dir sft_sft_data

echo "==> Token length stats"
python util/jsonl_token_stats.py --jsonl sft_sft_data/train.jsonl --model_name "$MODEL_NAME"

echo "==> Launching training (${TRAIN_MODE})"
if [[ "$TRAIN_MODE" == "smoke" ]]; then
  python train.py \
    --train_jsonl sft_sft_data/train.jsonl \
    --val_jsonl sft_sft_data/val.jsonl \
    --output_dir runs/svg_qwen_smoke \
    --max_length "$MAX_LENGTH" \
    --max_train_samples 512 \
    --max_eval_samples 128 \
    --num_train_epochs 1 \
    --eval_steps 50 \
    --save_steps 50
else
  python train.py \
    --train_jsonl sft_sft_data/train.jsonl \
    --val_jsonl sft_sft_data/val.jsonl \
    --output_dir runs/svg_qwen_full \
    --max_length "$MAX_LENGTH"
fi

echo "==> Done."
