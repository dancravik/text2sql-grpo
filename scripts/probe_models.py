"""Прогон моделей-кандидатов без обучения, чтобы выбрать модель для GRPO.

Для каждой модели: N вопросов из отфильтрованного Spider train, по n ответов на вопрос,
награда каждого ответа, сводная статистика.

Запуск (из корня репозитория):
    DATA_DIR=/kaggle/input/<dataset> python scripts/probe_models.py --smoke
    DATA_DIR=/kaggle/input/<dataset> python scripts/probe_models.py
"""

import argparse
import gc
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import filter_train, load_spider  # noqa: E402
from prompt import render_prompt  # noqa: E402
from sampling import load_llm, sample_and_score  # noqa: E402

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
]
OUTCOMES = ["no_sql", "exec_error", "wrong", "correct"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--num-questions", type=int, default=500)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/probe")
    p.add_argument(
        "--enable-thinking", choices=["on", "off"], default="off",
        help="enable_thinking для chat template; по умолчанию off (ответ сразу в <answer>)",
    )
    p.add_argument("--smoke", action="store_true", help="20 вопросов, только первая модель")
    args = p.parse_args()
    if args.smoke:
        args.num_questions = 20
        args.models = args.models[:1]
    args.enable_thinking = args.enable_thinking == "on"
    return args


def sample_examples(num: int, seed: int) -> list[dict]:
    """Одна и та же выборка для всех моделей."""
    data_dir = os.environ.get("DATA_DIR", "data")
    examples = filter_train(load_spider(data_dir, "train"))
    rng = random.Random(seed)
    return rng.sample(examples, min(num, len(examples)))


def probe_model(model_id: str, examples: list[dict], args) -> tuple[list[dict], float, str]:
    """Генерирует ответы одной моделью и оценивает их.

    Возвращает записи, время генерации и последние 40 символов первого промпта.
    """
    import torch

    llm = load_llm(model_id, args.max_model_len, args.seed)
    records, gen_time = sample_and_score(
        llm, examples, args.n, args.temperature, args.max_tokens, args.seed, args.enable_thinking
    )
    prompt_tail = render_prompt(llm.get_tokenizer(), examples[0], args.enable_thinking)[0][-40:]
    # Вопрос, эталон и путь к базе — как раньше, чтобы ответы можно было читать глазами
    records = [
        {"question": ex["question"], "gold_sql": ex["gold_sql"], "db_path": ex["db_path"], **r}
        for ex, r in zip(examples, records, strict=True)
    ]

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return records, gen_time, prompt_tail


def summarize(records: list[dict], n: int, gen_time: float) -> dict:
    """Сводная статистика по одной модели."""
    k_correct = np.array([r["k_correct"] for r in records])
    comps = [c for r in records for c in r["completions"]]
    outcomes = Counter(c["outcome"] for c in comps)
    prompt_lens = np.array([r["prompt_tokens"] for r in records])
    out_tokens = np.array([c["num_tokens"] for c in comps])
    return {
        "frac_0_of_n": float(np.mean(k_correct == 0)),
        "frac_signal": float(np.mean((k_correct > 0) & (k_correct < n))),
        "frac_n_of_n": float(np.mean(k_correct == n)),
        "pass@1": float(np.mean(k_correct / n)),
        f"pass@{n}": float(np.mean(k_correct > 0)),
        "format_ok": float(np.mean([c["format_ok"] for c in comps])),
        **{f"out_{o}": outcomes[o] / len(comps) for o in OUTCOMES},
        "truncated": float(np.mean([c["finish_reason"] == "length" for c in comps])),
        "prompt_tokens_mean": float(prompt_lens.mean()),
        "prompt_tokens_p95": float(np.percentile(prompt_lens, 95)),
        "completion_tokens_mean": float(out_tokens.mean()),
        "gen_time_s": gen_time,
        "output_tokens_per_s": float(out_tokens.sum() / gen_time),
    }


def print_table(summary: dict) -> None:
    models = list(summary)
    # enable_thinking и prompt_tail — не числа, в таблицу не идут
    first = next(iter(summary.values()))
    keys = [k for k, v in first.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    names = [m.split("/")[-1] for m in models]
    width = max(len(n) for n in names) + 2
    print("\n" + "metric".ljust(24) + "".join(n.rjust(width) for n in names))
    for k in keys:
        row = k.ljust(24)
        for m in models:
            v = summary[m][k]
            row += (f"{v:.3f}" if v < 10 else f"{v:.0f}").rjust(width)
        print(row)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = sample_examples(args.num_questions, args.seed)
    print(f"Questions: {len(examples)}, samples per question: {args.n}")

    summary = {}
    for model_id in args.models:
        print(f"\n=== {model_id} ===")
        records, gen_time, prompt_tail = probe_model(model_id, examples, args)
        name = model_id.split("/")[-1]
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[model_id] = {
            "enable_thinking": args.enable_thinking,
            "prompt_tail": prompt_tail,
            **summarize(records, args.n, gen_time),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print_table(summary)


if __name__ == "__main__":
    main()
