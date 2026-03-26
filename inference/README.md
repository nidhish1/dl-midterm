# Inference (Kaggle)

Use this folder to run inference from a LoRA adapter/checkpoint and create `submission.csv`.

## 1) Optional: extract checkpoint tar

```bash
python inference/extract_adapter.py \
  --tar_path /kaggle/input/your-adapter-dataset/checkpoint-2200.tar.gz \
  --out_dir /kaggle/working/adapter_extracted
```

## 2) Generate submission

If using extracted folder:

```bash
python inference/kaggle_generate_submission.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir /kaggle/working/adapter_extracted/checkpoint-2200 \
  --test_csv /kaggle/input/dl-spring-2026-svg-generation/test.csv \
  --output_csv /kaggle/working/submission.csv \
  --batch_size 8 \
  --max_new_tokens 1200
```

Or directly from tar:

```bash
python inference/kaggle_generate_submission.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_tar /kaggle/input/your-adapter-dataset/checkpoint-2200.tar.gz \
  --extract_to /kaggle/working/adapter_extracted \
  --test_csv /kaggle/input/dl-spring-2026-svg-generation/test.csv \
  --output_csv /kaggle/working/submission.csv
```
