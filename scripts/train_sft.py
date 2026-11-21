"""Train Qwen3.5-2B with LoRA on frozen Spider train and export it for evaluate.py."""

import argparse
import importlib.metadata
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comet_tracking import start_experiment  # noqa: E402
from data import DATA_DIR  # noqa: E402
from prompt import render_prompt  # noqa: E402
from splits import DEFAULT_MANIFEST, load_split, sha256_file  # noqa: E402

IGNORE_INDEX = -100
GIB = 1024**3
BATCH_PROBE_OOM_EXIT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--num-train-examples", type=int, help="Seeded subset for trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--fp16-init-scale", type=float, default=128.0,
                        help="Initial dynamic loss scale for FP16 on T4")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-strategy", choices=("steps", "no"), default="steps",
                        help="Use no for short pilots to save disk space")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--epochs", type=float, default=1.0)
    duration.add_argument("--max-steps", type=int)
    duration.add_argument("--smoke", action="store_true", help="8 steps, 64 train examples, batch 1 x 4")
    parser.add_argument("--batch-probe", action="store_true", help="Measure training memory without exporting a model")
    parser.add_argument("--comet", action="store_true", help="Track this run using COMET_API_KEY")
    args = parser.parse_args()
    if args.batch_probe and args.max_steps is None:
        parser.error("--batch-probe requires an explicit --max-steps")
    if args.smoke:
        args.max_steps = 8
        args.num_train_examples = 64
        args.micro_batch_size = 1
        args.grad_accum = 4
        args.logging_steps = 1
        args.warmup_ratio = 0.0
    for name in ("num_train_examples", "lora_r", "lora_alpha", "micro_batch_size", "grad_accum",
                 "max_length", "logging_steps", "save_steps", "max_steps", "epochs", "learning_rate",
                 "fp16_init_scale"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.out_dir.exists():
        parser.error(f"Output already exists: {args.out_dir}. Pick a new directory for each trial.")
    return args


def encode_example(tokenizer, example: dict, max_length: int) -> dict:
    prompt, _ = render_prompt(tokenizer, example, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer = f"<answer>{example['gold_sql'].strip()}</answer>"
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        raise ValueError(f"{example['id']} has {len(input_ids)} tokens, above --max-length={max_length}")
    return {"input_ids": input_ids, "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids}


def collate_batch(tokenizer, examples: list[dict]) -> dict:
    import torch

    batch = tokenizer.pad([{"input_ids": ex["input_ids"]} for ex in examples], return_tensors="pt")
    width = batch["input_ids"].shape[1]
    batch["labels"] = torch.tensor(
        [ex["labels"] + [IGNORE_INDEX] * (width - len(ex["labels"])) for ex in examples],
        dtype=torch.long,
    )
    return batch


def train(args: argparse.Namespace, experiment) -> dict:
    from functools import partial

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Trainer,
        TrainerCallback, TrainingArguments, set_seed,
    )

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible GPU. Set CUDA_VISIBLE_DEVICES=0 on Kaggle.")
    set_seed(args.seed)
    examples = load_split("train", DATA_DIR, args.manifest)
    if args.num_train_examples is not None:
        if args.num_train_examples > len(examples):
            raise ValueError(f"Requested {args.num_train_examples} train examples, have {len(examples)}")
        examples = random.Random(args.seed).sample(examples, args.num_train_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = "right"
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must have an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = [encode_example(tokenizer, ex, args.max_length) for ex in examples]

    # Keep the checkpoint architecture used by vLLM for Qwen3.5. Only text LoRA is trainable.
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float16, attn_implementation="sdpa",
    )
    revision = getattr(base.config, "_commit_hash", None) or args.revision
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
        exclude_modules=r".*(visual|vision).*",
        task_type="CAUSAL_LM",
        revision=revision,
    ))
    trainable = [param for param in model.parameters() if param.requires_grad]
    if {param.dtype for param in trainable} != {torch.float32}:
        raise RuntimeError("FP16 training requires FP32 LoRA parameters")

    class OptimizerStats(TrainerCallback):
        def __init__(self):
            self.skipped = 0
            self.step_started = None
            self.probe_steps = []

        def on_step_begin(self, args, state, control, **kwargs):
            if probe:
                torch.cuda.reset_peak_memory_stats()
                self.step_started = time.perf_counter()

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.skipped += int(trainer.accelerator.optimizer_step_was_skipped)

        def on_step_end(self, args, state, control, **kwargs):
            if probe:
                torch.cuda.synchronize()
                self.record_probe_step(state.global_step, oom=False)

        def record_probe_step(self, step, oom):
            metrics = {
                "step_peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                "step_peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                "cuda_oom": int(oom),
            }
            if self.step_started is not None:
                metrics["step_seconds"] = time.perf_counter() - self.step_started
            self.probe_steps.append({"step": step, **metrics})
            self.step_started = None
            if experiment is not None:
                experiment.log_metrics(metrics, step=step)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if experiment is not None and logs:
                metrics = {name: value for name, value in logs.items()
                           if isinstance(value, (int, float)) and math.isfinite(value)}
                metrics["optimizer_steps_skipped"] = self.skipped
                metrics["fp16_loss_scale"] = trainer.accelerator.scaler.get_scale()
                if not probe:
                    metrics["peak_allocated_gib_so_far"] = torch.cuda.max_memory_allocated() / GIB
                    metrics["peak_reserved_gib_so_far"] = torch.cuda.max_memory_reserved() / GIB
                experiment.log_metrics(metrics, step=state.global_step)

    probe = args.batch_probe
    config = TrainingArguments(
        output_dir=str(args.out_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="no" if probe else args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        label_names=["labels"],
        report_to="none",
    )
    optimizer_stats = OptimizerStats()
    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=dataset,
        data_collator=partial(collate_batch, tokenizer),
        processing_class=tokenizer,
        callbacks=[optimizer_stats],
    )
    scaler = trainer.accelerator.scaler
    if scaler is None:
        raise RuntimeError("Expected an active FP16 gradient scaler")
    # Configure before the first backward pass. Keep dynamic overflow handling enabled.
    scale_state = scaler.state_dict()
    scale_state["scale"] = args.fp16_init_scale
    scaler.load_state_dict(scale_state)
    report = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "base_revision": revision,
        "manifest_sha256": sha256_file(args.manifest),
        "train_example_ids": [ex["id"] for ex in examples],
        "num_train_examples": len(examples),
        "effective_batch_size": args.micro_batch_size * args.grad_accum,
        "trainable_params": sum(param.numel() for param in trainable),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
        "versions": {pkg: importlib.metadata.version(pkg) for pkg in ("torch", "transformers", "accelerate", "peft")},
        "comet_experiment_key": experiment.get_key() if experiment is not None else None,
    }
    if experiment is not None:
        experiment.log_parameters(report["args"] | {
            key: report[key] for key in ("base_revision", "manifest_sha256", "num_train_examples",
                                         "effective_batch_size", "trainable_params", "gpu")
        })

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = trainer.train()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as error:
        if not probe:
            raise
        result = None
        failed_at_step = trainer.state.global_step + 1
        optimizer_stats.record_probe_step(failed_at_step, oom=True)
        report.update(status="cuda_oom", failed_at_step=failed_at_step, error=str(error))
    seconds = time.perf_counter() - started
    applied = trainer.state.global_step - optimizer_stats.skipped
    report.update(
        global_step=trainer.state.global_step,
        optimizer_steps_applied=applied,
        optimizer_steps_skipped=optimizer_stats.skipped,
        fp16_final_scale=scaler.get_scale(),
        train_seconds=seconds,
        gpu_hours=seconds / 3600,
        peak_allocated_gib=max(
            [torch.cuda.max_memory_allocated() / GIB]
            + [step["step_peak_allocated_gib"] for step in optimizer_stats.probe_steps]
        ),
        peak_reserved_gib=max(
            [torch.cuda.max_memory_reserved() / GIB]
            + [step["step_peak_reserved_gib"] for step in optimizer_stats.probe_steps]
        ),
        log_history=[
            {key: None if isinstance(value, float) and not math.isfinite(value) else value
             for key, value in entry.items()}
            for entry in trainer.state.log_history
        ],
    )
    if probe:
        report["batch_probe_steps"] = optimizer_stats.probe_steps
        if result is None:
            return report

    if not math.isfinite(result.training_loss):
        raise RuntimeError(f"Non-finite training loss: {result.training_loss}")
    print(f"Optimizer steps: {applied} applied, {optimizer_stats.skipped} skipped; final scale {scaler.get_scale():g}")
    if not any(torch.count_nonzero(param).item() for name, param in model.named_parameters()
               if param.requires_grad and ".lora_B." in name):
        raise RuntimeError("LoRA weights were not updated. Check for skipped FP16 optimizer steps.")
    report.update(status="ok", train_loss=result.training_loss)
    if probe:
        return report

    model.save_pretrained(args.out_dir / "adapter")
    tokenizer.save_pretrained(args.out_dir / "adapter")
    # Preserve completed training even if exporting the larger merged model fails.
    report["export_status"] = "pending"
    (args.out_dir / "run.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(args.out_dir / "merged")
    AutoProcessor.from_pretrained(args.model, revision=revision).save_pretrained(args.out_dir / "merged")
    report["export_status"] = "ok"
    return report


def main() -> None:
    args = parse_args()
    experiment = start_experiment(name=args.out_dir.name) if args.comet else None
    try:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        report = train(args, experiment)
        path = args.out_dir / "run.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved {path}" if args.batch_probe else f"Saved adapter, merged model and run.json to {args.out_dir}")
        if experiment is not None:
            summary = {key: report[key] for key in (
                "train_seconds", "gpu_hours", "peak_allocated_gib", "peak_reserved_gib",
            )}
            if args.batch_probe:
                summary["fits"] = int(report["status"] == "ok")
                experiment.log_other("status", report["status"])
                if report["status"] == "cuda_oom":
                    experiment.log_other("cuda_oom_error", report["error"])
            experiment.log_metrics(summary, step=report.get("failed_at_step", report["global_step"]))
        if report["status"] == "cuda_oom":
            print(report["error"], file=sys.stderr)
            sys.exit(BATCH_PROBE_OOM_EXIT)
    finally:
        if experiment is not None:
            experiment.end()


if __name__ == "__main__":
    main()
