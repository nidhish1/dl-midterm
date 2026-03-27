import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def pick_dtype() -> tuple[torch.dtype, bool, bool]:
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def resolve_target_modules(model) -> list[str]:
    """
    Pick common projection layers for Qwen-like decoder models.
    Falls back gracefully if some names are missing.
    """
    wanted_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    }

    found = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in wanted_suffixes:
            found.add(leaf)

    if not found:
        raise ValueError("Could not find common LoRA target modules in the model.")

    ordered = [m for m in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"] if m in found]
    return ordered


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_sft_config_kwargs(args, logging_dir: Path, use_bf16: bool, use_fp16: bool) -> dict:
    """Build kwargs compatible with the installed TRL version."""
    kwargs = {
        "output_dir": str(args.output_dir),
        "run_name": args.run_name,
        "logging_dir": str(logging_dir),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": args.seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": True,
        "max_length": args.max_length,
        "packing": False,
        # Must be False for Qwen chat_template + prompt/completion JSONL: TRL's completion-only
        # path tokenizes "prompt" as plain text and disagrees with chat-formatted full sequence.
        "completion_only_loss": args.completion_only_loss,
        "report_to": args.report_to,
        "remove_unused_columns": True,
        "save_strategy": "steps",
    }

    # TRL/Transformers changed this name across versions.
    sig = inspect.signature(SFTConfig.__init__).parameters
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    # Optional across versions.
    if "save_safetensors" in sig:
        kwargs["save_safetensors"] = True

    # Keep only kwargs supported by this installed TRL version.
    return {k: v for k, v in kwargs.items() if k in sig}


def dataset_add_chat_text_column(ds, tokenizer, num_proc: int | None = None):
    """Single column `text` = apply_chat_template(user, assistant). Drops all original columns."""

    def to_text(batch: dict) -> dict:
        texts = []
        for prompt, completion in zip(batch["prompt"], batch["completion"], strict=True):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    map_kwargs: dict = {
        "batched": True,
        "remove_columns": list(ds.column_names),
        "desc": "chat_template text",
    }
    if num_proc is not None and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    return ds.map(to_text, **map_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--val_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_length", type=int, required=True)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--run_name", type=str, default="svg_sft")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--load_best_model_at_end", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument(
        "--completion_only_loss",
        action="store_true",
        help="Mask loss to assistant only (often breaks with Qwen+JSONL in TRL). Default: full-sequence loss on chat-formatted text.",
    )
    parser.add_argument(
        "--dataset_map_num_proc",
        type=int,
        default=1,
        help="Parallel workers for chat preprocessing map. 0 or 1 = sequential (safest); >1 enables multiprocessing.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = args.output_dir / "logs"

    if args.load_best_model_at_end and (args.save_steps % args.eval_steps != 0):
        raise ValueError("--save_steps must be a multiple of --eval_steps when --load_best_model_at_end is enabled.")

    dtype, use_bf16, use_fp16 = pick_dtype()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    target_modules = resolve_target_modules(model)
    print("LoRA target modules:", target_modules)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )

    data_files = {
        "train": str(args.train_jsonl),
        "validation": str(args.val_jsonl),
    }
    dataset = load_dataset("json", data_files=data_files)

    train_ds = dataset["train"]
    eval_ds = dataset["validation"]

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    map_mp = args.dataset_map_num_proc if args.dataset_map_num_proc > 1 else None
    train_ds = dataset_add_chat_text_column(train_ds, tokenizer, num_proc=map_mp)
    eval_ds = dataset_add_chat_text_column(eval_ds, tokenizer, num_proc=map_mp)

    sft_trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    sft_config_sig = inspect.signature(SFTConfig.__init__).parameters
    training_kwargs = build_sft_config_kwargs(args, logging_dir, use_bf16, use_fp16)
    if args.completion_only_loss:
        print(
            "WARNING: --completion_only_loss can trigger TRL token mismatch on Qwen; "
            "prefer default (full-sequence loss on chat text).",
            flush=True,
        )

    if "dataset_text_field" in sft_config_sig:
        training_kwargs["dataset_text_field"] = "text"
    training_args = SFTConfig(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "peft_config": peft_config,
    }
    if "dataset_text_field" in sft_trainer_sig:
        trainer_kwargs["dataset_text_field"] = "text"

    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # Older TRL uses tokenizer=... instead of processing_class=...
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    print(f"train rows: {len(train_ds)}")
    print(f"eval rows : {len(eval_ds)}")
    print(f"train_jsonl rows on disk: {count_jsonl_rows(args.train_jsonl)}")
    print(f"val_jsonl rows on disk  : {count_jsonl_rows(args.val_jsonl)}")
    print(f"checkpoints dir: {args.output_dir}")
    print(f"logging dir    : {logging_dir}")
    print(f"load best model: {args.load_best_model_at_end}")

    trainer.train()
    final_eval_metrics = trainer.evaluate()
    print("final eval metrics:")
    for k, v in sorted(final_eval_metrics.items()):
        print(f"  {k}: {v}")

    metrics_path = args.output_dir / "final_eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_eval_metrics, f, indent=2, sort_keys=True)
    print(f"saved final eval metrics to {metrics_path}")

    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "final_adapter"))

    print(f"Saved adapter to {args.output_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()


