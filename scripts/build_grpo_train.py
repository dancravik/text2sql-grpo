"""Отбор вопросов для GRPO: оставляем те, где модель решает часть из n попыток (1..n-1).

Если все n ответов одинаково верны или неверны, награды в группе равны, advantage нулевой,
и вопрос не даёт сигнала для обучения.

Результаты пишутся по кускам в {out_dir}/counts.jsonl; при перезапуске уже обработанные
вопросы пропускаются. Итог — {out_dir}/grpo_train_ids.json (id восстанавливаются через load_spider).

Запуск (из корня репозитория):
    DATA_DIR=/kaggle/input/<dataset> python scripts/build_grpo_train.py --smoke
    DATA_DIR=/kaggle/input/<dataset> python scripts/build_grpo_train.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import DATA_DIR, filter_train, load_spider  # noqa: E402
from sampling import load_llm, sample_and_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument(
        "--enable-thinking", choices=["on", "off"], default="off",
        help="enable_thinking для chat template; по умолчанию off (ответ сразу в <answer>)",
    )
    p.add_argument("--out-dir", default="results/grpo_train")
    p.add_argument("--smoke", action="store_true", help="первые 40 вопросов")
    args = p.parse_args()
    args.enable_thinking = args.enable_thinking == "on"
    return args


def read_counts(path: Path, n: int) -> dict[str, dict]:
    """Уже посчитанные строки counts.jsonl по id."""
    if not path.exists():
        return {}
    counts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["n"] != n:
                raise ValueError(f"{path} was built with n={row['n']}, got --n {n}; use another --out-dir")
            counts[row["id"]] = row
    return counts


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = out_dir / "counts.jsonl"

    examples = filter_train(load_spider(DATA_DIR, "train"))
    if args.smoke:
        examples = examples[:40]

    counts = read_counts(counts_path, args.n)
    todo = [ex for ex in examples if ex["id"] not in counts]
    print(f"Questions: {len(examples)}, already done: {len(examples) - len(todo)}, to do: {len(todo)}")

    gen_time = 0.0
    if todo:
        llm = load_llm(args.model, args.max_model_len, args.seed)
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start:start + args.chunk_size]
            records, chunk_time = sample_and_score(
                llm, chunk, args.n, args.temperature, args.max_tokens, args.seed, args.enable_thinking
            )
            gen_time += chunk_time
            rows = [
                {
                    "id": r["id"],
                    "k_correct": r["k_correct"],
                    "n": args.n,
                    "format_ok_frac": sum(c["format_ok"] for c in r["completions"]) / len(r["completions"]),
                }
                for r in records
            ]
            with open(counts_path, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(row) + "\n" for row in rows)
            counts.update((row["id"], row) for row in rows)
            done = start + len(chunk)
            print(f"chunk {done}/{len(todo)}: generation {chunk_time:.0f}s")

    # Итог считаем по вопросам текущего запуска: в counts.jsonl могут быть и другие id
    k = [counts[ex["id"]]["k_correct"] for ex in examples]
    train_ids = [ex["id"] for ex, kc in zip(examples, k, strict=True) if 1 <= kc <= args.n - 1]
    with open(out_dir / "grpo_train_ids.json", "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=1)

    total = len(examples)
    n = args.n
    print()
    for label, value in [
        ("total questions", total),
        (f"0/{n}", f"{sum(kc == 0 for kc in k) / total:.3f}"),
        (f"1..{n - 1}/{n}", f"{len(train_ids) / total:.3f}"),
        (f"{n}/{n}", f"{sum(kc == n for kc in k) / total:.3f}"),
        ("GRPO train set", f"{len(train_ids)} -> {out_dir / 'grpo_train_ids.json'}"),
        ("generation time", f"{gen_time:.0f}s (this run only)"),
    ]:
        print(f"{label + ':':<18}{value}")


if __name__ == "__main__":
    main()
