"""Тест осуществимости GRPO + LoRA на одной T4: помещается ли шаг в память и сколько он длится.

Это не обучение: несколько шагов GRPO на небольшой выборке Spider train. Один шаг =
prompts_per_step вопросов × num_generations ответов: генерация, награда, forward/backward
и шаг оптимизатора. Первый шаг — прогрев, в средние не входит.

Написано под trl 1.16.0 / peft 0.21.2 (имена параметров сверены через inspect.signature):
- строковый prompt trl токенизирует как есть, без chat template (GRPOTrainer._tokenize_prompts),
  поэтому в датасет кладём готовый текст из render_prompt;
- reward-функция вызывается как f(prompts=, completions=, completion_ids=, **колонки датасета)
  и возвращает list[float | None];
- шаг оптимизатора = gradient_accumulation_steps микробатчей, генерация — раз в шаг
  (steps_per_generation по умолчанию равен gradient_accumulation_steps).

Qwen3.5 мультимодальная: без vLLM грузим только языковую часть (Qwen3_5ForCausalLM), с vLLM colocate —
модель целиком, иначе trl не сможет синхронизировать веса в vLLM (см. load_model).

Запуск (из корня репозитория, одна GPU):
    CUDA_VISIBLE_DEVICES=0 DATA_DIR=/kaggle/input/<dataset> \
        python scripts/feasibility_grpo.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --smoke
"""

import argparse
import importlib.metadata
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path

import peft
import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedModel,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import DATA_DIR, filter_train, load_spider  # noqa: E402
from prompt import render_prompt  # noqa: E402
from reward import reward  # noqa: E402

GB = 1024**3
# Ключи из логов trl, которые сохраняем по каждому шагу
LOG_KEYS = ["reward", "reward_std", "frac_reward_zero_std", "completions/mean_length", "completions/clipped_ratio"]
VISION_MARKERS = ("visual", "vision")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-questions", type=int, default=64)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="ответов в одном forward/backward; по умолчанию num_generations (одна группа)",
    )
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.8, help="как в probe_models.py")
    p.add_argument("--enable-thinking", choices=["on", "off"], default="off")
    p.add_argument("--use-vllm", choices=["none", "colocate"], default="none")
    p.add_argument(
        "--vllm-gpu-memory-utilization", type=float, default=0.45,
        help="доля памяти GPU под vLLM (веса + KV-кэш); остальное — обучению",
    )
    p.add_argument(
        "--vllm-sleep", action="store_true",
        help="vLLM освобождает память на время обучения (sleep mode), будится только для генерации",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/feasibility")
    p.add_argument("--smoke", action="store_true", help="max_steps=2")
    args = p.parse_args()
    if args.smoke:
        args.max_steps = 2
    if args.micro_batch_size is None:
        args.micro_batch_size = args.num_generations
    completions_per_step = args.prompts_per_step * args.num_generations
    if completions_per_step % args.micro_batch_size != 0:
        p.error(f"prompts_per_step * num_generations ({completions_per_step}) must be divisible by --micro-batch-size")
    return args


def versions() -> dict[str, str | None]:
    """Версии ключевых пакетов; vllm может быть не установлен."""
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = None
    return {
        "trl": trl.__version__,
        "peft": peft.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm_version,
    }


def build_dataset(tokenizer, args: argparse.Namespace) -> tuple[Dataset, list[int]]:
    """Готовые промпты из render_prompt + поля для награды; возвращает датасет и длины промптов."""
    examples = filter_train(load_spider(DATA_DIR, "train"))
    examples = random.Random(args.seed).sample(examples, min(args.num_questions, len(examples)))
    enable_thinking = args.enable_thinking == "on"

    rows, prompt_lens = [], []
    for ex in examples:
        prompt, prefix = render_prompt(tokenizer, ex, enable_thinking)
        # trl токенизирует строковый prompt вызовом tokenizer(text=...) с add_special_tokens по умолчанию.
        # Если токенизатор добавляет BOS, промпт разойдётся с render_prompt — проверяем.
        ids = tokenizer(text=prompt)["input_ids"]
        if ids != tokenizer(prompt, add_special_tokens=False)["input_ids"]:
            raise RuntimeError("Tokenizer adds special tokens: trl prompt would differ from render_prompt")
        prompt_lens.append(len(ids))
        rows.append({
            "prompt": prompt,
            "completion_prefix": prefix,
            "gold_sql": ex["gold_sql"],
            "db_path": ex["db_path"],
        })
    return Dataset.from_list(rows), prompt_lens


def sql_reward(
    completions: list[str],
    completion_prefix: list[str],
    gold_sql: list[str],
    db_path: list[str],
    **kwargs,
) -> list[float]:
    """Награда из src/reward.py; к ответу приклеивается тег, который шаблон дописал в конец промпта."""
    return [
        reward(prefix + text, gold, db)
        for text, prefix, gold, db in zip(completions, completion_prefix, gold_sql, db_path, strict=True)
    ]


def load_model(model_id: str, text_only: bool) -> tuple[PreTrainedModel, dict]:
    """fp16 + sdpa. Мультимодальную модель при text_only грузим без энкодера изображений."""
    kwargs = {"dtype": torch.float16, "attn_implementation": "sdpa"}
    multimodal = getattr(AutoConfig.from_pretrained(model_id), "vision_config", None) is not None
    load_note = None
    if not multimodal:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    elif text_only:
        # Для Qwen3.5 AutoModelForCausalLM даёт Qwen3_5ForCausalLM: веса model.visual.* не загружаются
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except ValueError as e:
            load_note = f"AutoModelForCausalLM failed ({e}), loaded full multimodal model"
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    else:
        # trl синхронизирует веса в vLLM по именам, а vLLM поднимает мультимодальную архитектуру:
        # имена текстовой модели (model.layers.*) его маппер не знает, нужны model.language_model.*
        load_note = "loaded full multimodal model: vLLM colocate weight sync needs multimodal parameter names"
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.to("cuda")

    vision_params = sum(
        p.numel() for n, p in model.named_parameters() if any(m in n for m in VISION_MARKERS)
    )
    info = {
        "model_class": type(model).__name__,
        "multimodal_checkpoint": multimodal,
        "text_only_loaded": vision_params == 0,
        "vision_params": vision_params,
        "load_note": load_note,
    }
    return model, info


def add_lora(model: PreTrainedModel, args: argparse.Namespace, has_vision: bool) -> PreTrainedModel:
    """LoRA на все линейные слои языковой модели (lm_head и энкодер изображений не трогаем)."""
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        # all-linear в peft исключает выходной слой (lm_head)
        target_modules="all-linear",
        exclude_modules=r".*(visual|vision).*" if has_vision else None,
        task_type="CAUSAL_LM",
    )
    # autocast_adapter_dtype=True (по умолчанию) держит веса LoRA в fp32 поверх fp16-модели,
    # без этого GradScaler при fp16=True падает на fp16-градиентах
    model = get_peft_model(model, config)
    lora_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    if lora_dtypes != {torch.float32}:
        raise RuntimeError(f"Expected fp32 LoRA weights for fp16 training, got {lora_dtypes}")
    return model


def linear_attn_kernels() -> dict[str, bool]:
    """Есть ли быстрые ядра для linear attention (Qwen3.5); без них transformers берёт медленный fallback на torch."""
    return {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d", "kernels")}


class StepStats(TrainerCallback):
    """Время и пиковая память каждого шага оптимизатора, награда и длина ответа из логов trl."""

    def __init__(self) -> None:
        self.steps: list[dict] = []
        self._t0 = 0.0

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs) -> None:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        self.steps.append({
            "step": state.global_step,
            "time_s": time.perf_counter() - self._t0,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / GB,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB,
            "device_used_gb": (total - free) / GB,
        })

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        # on_log вызывается после on_step_end того же шага (logging_steps=1)
        if not logs or not self.steps or "reward" not in logs:
            return
        step = self.steps[-1]
        step.update({k: logs[k] for k in LOG_KEYS if k in logs})
        print(
            f"step {step['step']}: {step['time_s']:.1f} s, peak {step['peak_allocated_gb']:.2f} GB, "
            f"reward {step['reward']:.3f}, length {step.get('completions/mean_length', float('nan')):.0f}",
            flush=True,
        )


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(steps: list[dict]) -> dict:
    """Средние без первого шага (прогрев); пиковая память — максимум по всем шагам."""
    measured = steps[1:]
    return {
        "steps_done": len(steps),
        "peak_memory_gb": max((s["peak_allocated_gb"] for s in steps), default=None),
        "peak_reserved_gb": max((s["peak_reserved_gb"] for s in steps), default=None),
        "mean_step_time_s": mean([s["time_s"] for s in measured]),
        "mean_reward": mean([s["reward"] for s in measured if s.get("reward") is not None]),
        "mean_completion_length": mean(
            [s["completions/mean_length"] for s in measured if s.get("completions/mean_length") is not None]
        ),
        "warmup_step_time_s": steps[0]["time_s"] if steps else None,
    }


def error_status(e: BaseException) -> str:
    """Текст ошибки для отчёта; OOM (в т.ч. от vLLM) помечаем отдельно."""
    msg = str(e).strip().splitlines()[0][:500] if str(e).strip() else ""
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
        return f"CUDA OOM: {msg}"
    return f"{type(e).__name__}: {msg}"


def run(args: argparse.Namespace, report: dict, stats: StepStats) -> None:
    """Загрузка модели, LoRA, GRPOTrainer и обучение на max_steps шагов; заполняет report."""
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset, prompt_lens = build_dataset(tokenizer, args)
    report["prompt_tokens_mean"] = sum(prompt_lens) / len(prompt_lens)
    report["prompt_tokens_max"] = max(prompt_lens)
    print(f"Dataset: {len(dataset)} prompts, tokens mean {report['prompt_tokens_mean']:.0f}, max {max(prompt_lens)}")

    model, info = load_model(args.model, text_only=args.use_vllm == "none")
    report.update(info)
    model = add_lora(model, args, has_vision=info["vision_params"] > 0)
    trainable, total = model.get_nb_trainable_parameters()
    report["num_params"] = total - trainable
    report["trainable_params"] = trainable
    print(f"{info['model_class']}: {(total - trainable) / 1e6:.1f}M params, LoRA trainable {trainable / 1e6:.2f}M")
    if info["load_note"]:
        print(f"NOTE: {info['load_note']}")

    grad_accum = args.prompts_per_step * args.num_generations // args.micro_batch_size
    config = GRPOConfig(
        output_dir=str(Path(args.out_dir) / "trainer"),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        # Шаг оптимизатора = grad_accum микробатчей = prompts_per_step групп по num_generations ответов
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        use_vllm=args.use_vllm == "colocate",
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_sleep,
        # Без ограничения vLLM берёт max_position_embeddings из конфига (у Qwen3.5 — 262k) и не влезает
        vllm_max_model_length=max(prompt_lens) + args.max_completion_length,
    )
    report["grad_accum_steps"] = grad_accum
    report["generation_batch_size"] = config.generation_batch_size

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=sql_reward,
        args=config,
        train_dataset=dataset,
        # Токенизатор, а не AutoProcessor: промпты текстовые, процессор Qwen3.5 включил бы режим VLM
        processing_class=tokenizer,
        callbacks=[stats],
    )
    report["memory_after_init_gb"] = torch.cuda.memory_allocated() / GB
    print(f"Memory after init: {report['memory_after_init_gb']:.2f} GB")
    trainer.train()


def main() -> None:
    args = parse_args()
    vers = versions()
    print("Versions: " + ", ".join(f"{k} {v}" for k, v in vers.items()))
    if torch.cuda.device_count() != 1:
        sys.exit(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}: set CUDA_VISIBLE_DEVICES=0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.model.split('/')[-1]}_{args.use_vllm}"
    name += "_sleep" if args.use_vllm == "colocate" and args.vllm_sleep else ""
    name += "_think" if args.enable_thinking == "on" else ""
    name += "_smoke" if args.smoke else ""

    report: dict = {
        "versions": vers,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / GB,
        "model": args.model,
        "args": vars(args),
        "linear_attn_kernels": linear_attn_kernels(),
    }
    stats = StepStats()
    try:
        run(args, report, stats)
        report["status"] = "ok"
    except Exception as e:  # noqa: BLE001 — любая ошибка (включая OOM) идёт в отчёт
        traceback.print_exc()
        report["status"] = error_status(e)
        report["failed_at_step"] = len(stats.steps) + 1
        report["peak_allocated_at_failure_gb"] = torch.cuda.max_memory_allocated() / GB

    report.update(summarize(stats.steps))
    report["steps"] = stats.steps
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== {args.model} (vllm: {args.use_vllm}) ===")
    for key in (
        "status", "model_class", "text_only_loaded", "num_params", "trainable_params",
        "peak_memory_gb", "peak_reserved_gb", "mean_step_time_s", "mean_reward",
        "mean_completion_length", "steps_done",
    ):
        value = report.get(key)
        print(f"{key:24}{value:.3f}" if isinstance(value, float) else f"{key:24}{value}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
