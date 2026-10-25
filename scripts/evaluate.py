"""Оценка модели на Spider dev и BIRD Mini-Dev: execution accuracy и разбивка ошибок.

По умолчанию greedy (temperature 0, один ответ на вопрос): доля верных ответов = execution accuracy.
С --n 8 --temperature 0.8 получаются pass@1 и pass@8 при сэмплировании.

Вопросы, эталон которых нельзя проверить (ошибка, таймаут, больше 10 000 строк), исключаются
(filter_eval); их число пишется в metrics.json.

Модель — id на Hugging Face или локальная папка (для обученной: базовая модель с влитым LoRA).

Запуск (из корня репозитория):
    DATA_DIR=/kaggle/input/<dataset> python scripts/evaluate.py --model Qwen/Qwen3.5-2B --smoke
    DATA_DIR=/kaggle/input/<dataset> python scripts/evaluate.py --model Qwen/Qwen3.5-2B
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import DATA_DIR, filter_eval, load_bird_minidev, load_spider  # noqa: E402
from prompt import render_prompt  # noqa: E402
from sampling import load_llm, sample_and_score  # noqa: E402

LOADERS = {
    "spider_dev": lambda: load_spider(DATA_DIR, "dev"),
    "bird_minidev": lambda: load_bird_minidev(DATA_DIR),
}
OUTCOMES = ["no_sql", "exec_error", "wrong", "correct"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", default=None, help="имя папки с результатами; по умолчанию последняя часть --model")
    p.add_argument("--datasets", nargs="+", choices=list(LOADERS), default=list(LOADERS))
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--enable-thinking", choices=["on", "off"], default="off",
        help="enable_thinking для chat template; по умолчанию off (ответ сразу в <answer>)",
    )
    p.add_argument("--out-dir", default="results/eval")
    p.add_argument("--smoke", action="store_true", help="первые 20 вопросов каждого набора")
    args = p.parse_args()
    args.enable_thinking = args.enable_thinking == "on"
    args.name = args.name or Path(args.model.rstrip("/")).name
    return args


def check_lengths(tokenizer, examples: list[dict], args: argparse.Namespace) -> None:
    """Падаем до генерации, если промпт + max_tokens не влезает в max_model_len."""
    too_long = []
    for ex in examples:
        prompt, _ = render_prompt(tokenizer, ex, args.enable_thinking)
        n_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if n_tokens + args.max_tokens > args.max_model_len:
            too_long.append((ex["id"], n_tokens))
    if too_long:
        raise ValueError(
            f"{len(too_long)} prompts don't fit max_model_len={args.max_model_len} with max_tokens={args.max_tokens}, "
            f"e.g. {too_long[:3]}; raise --max-model-len"
        )


def accuracy(records: list[dict], n: int) -> float:
    return sum(r["k_correct"] for r in records) / (n * len(records))


def summarize(records: list[dict], n: int, gen_time: float) -> dict:
    """Метрики по одному набору; для BIRD — ещё точность по сложности."""
    comps = [c for r in records for c in r["completions"]]
    outcomes = Counter(c["outcome"] for c in comps)
    metrics = {
        "num_questions": len(records),
        # Greedy и n=1 — execution accuracy; при сэмплировании — pass@1
        "accuracy": accuracy(records, n),
    }
    if n > 1:
        metrics[f"pass@{n}"] = sum(r["k_correct"] > 0 for r in records) / len(records)
    metrics.update({
        **{f"out_{o}": outcomes[o] / len(comps) for o in OUTCOMES},
        "format_ok": sum(c["format_ok"] for c in comps) / len(comps),
        "truncated": sum(c["finish_reason"] == "length" for c in comps) / len(comps),
        "prompt_tokens_mean": sum(r["prompt_tokens"] for r in records) / len(records),
        "prompt_tokens_max": max(r["prompt_tokens"] for r in records),
        "completion_tokens_mean": sum(c["num_tokens"] for c in comps) / len(comps),
        "gen_time_s": gen_time,
    })

    by_difficulty = defaultdict(list)
    for r in records:
        if r["difficulty"] is not None:
            by_difficulty[r["difficulty"]].append(r)
    if by_difficulty:
        metrics["accuracy_by_difficulty"] = {
            d: {"num_questions": len(rs), "accuracy": accuracy(rs, n)} for d, rs in sorted(by_difficulty.items())
        }
    return metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = load_llm(args.model, args.max_model_len, args.seed)
    report = {"model": args.model, "args": vars(args), "datasets": {}}

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        loaded = LOADERS[dataset]()
        examples = filter_eval(loaded)
        num_excluded = len(loaded) - len(examples)
        if args.smoke:
            examples = examples[:20]
        check_lengths(llm.get_tokenizer(), examples, args)

        records, gen_time = sample_and_score(
            llm, examples, args.n, args.temperature, args.max_tokens, args.seed, args.enable_thinking
        )
        # Вопрос, эталон и сложность рядом с ответами — чтобы разбирать ошибки глазами
        records = [
            {
                "question": ex["question"], "evidence": ex["evidence"], "gold_sql": ex["gold_sql"],
                "db_id": ex["db_id"], "difficulty": ex["difficulty"], **r,
            }
            for ex, r in zip(examples, records, strict=True)
        ]
        with open(out_dir / f"{dataset}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        metrics = {"num_excluded": num_excluded, **summarize(records, args.n, gen_time)}
        report["datasets"][dataset] = metrics
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"accuracy {metrics['accuracy']:.3f} on {metrics['num_questions']} questions ({num_excluded} excluded)")
        for o in OUTCOMES:
            print(f"  {o:<11}{metrics[f'out_{o}']:.3f}")
        for d, m in metrics.get("accuracy_by_difficulty", {}).items():
            print(f"  {d:<11}{m['accuracy']:.3f} ({m['num_questions']})")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
