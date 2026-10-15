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
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import filter_train, load_spider  # noqa: E402
from prompt import build_messages  # noqa: E402
from reward import check_format, compare, execute, extract_sql, reward  # noqa: E402

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
    p.add_argument("--smoke", action="store_true", help="20 вопросов, только первая модель")
    args = p.parse_args()
    if args.smoke:
        args.num_questions = 20
        args.models = args.models[:1]
    return args


def sample_examples(num: int, seed: int) -> list[dict]:
    """Одна и та же выборка для всех моделей."""
    data_dir = os.environ.get("DATA_DIR", "data")
    examples = filter_train(load_spider(data_dir, "train"))
    rng = random.Random(seed)
    return rng.sample(examples, min(num, len(examples)))


def classify(text: str, gold_rows: list[tuple], ex: dict) -> str:
    """Тип исхода: no_sql | exec_error | wrong | correct."""
    sql = extract_sql(text)
    if sql is None:
        return "no_sql"
    rows, err = execute(ex["db_path"], sql)
    if err is not None:
        return "exec_error"
    return "correct" if compare(rows, gold_rows, ex["gold_sql"]) else "wrong"


def probe_model(model_id: str, examples: list[dict], gold_rows: list, args) -> tuple[list[dict], float]:
    """Генерирует ответы одной моделью и оценивает их. Возвращает записи и время генерации."""
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        dtype="half",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
        seed=args.seed,
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(build_messages(ex), tokenize=False, add_generation_prompt=True)
        for ex in examples
    ]
    prompt_lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    params = SamplingParams(
        n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed
    )

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t0

    records = []
    for ex, out, gold, plen in zip(examples, outputs, gold_rows, prompt_lens):
        completions = []
        for c in out.outputs:
            completions.append({
                "text": c.text,
                "reward": reward(c.text, ex["gold_sql"], ex["db_path"]),
                "outcome": classify(c.text, gold, ex),
                "format_ok": check_format(c.text),
                "finish_reason": c.finish_reason,
                "num_tokens": len(c.token_ids),
            })
        records.append({
            "question": ex["question"],
            "gold_sql": ex["gold_sql"],
            "db_path": ex["db_path"],
            "prompt_tokens": plen,
            "completions": completions,
        })

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return records, gen_time


def summarize(records: list[dict], n: int, gen_time: float) -> dict:
    """Сводная статистика по одной модели."""
    k_correct = np.array([sum(c["outcome"] == "correct" for c in r["completions"]) for r in records])
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
    keys = list(next(iter(summary.values())))
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
    gold_rows = []
    for ex in examples:
        rows, err = execute(ex["db_path"], ex["gold_sql"])
        if err is not None:
            raise RuntimeError(f"Gold SQL failed ({err}) after filtering: {ex['gold_sql']}")
        gold_rows.append(rows)
    print(f"Questions: {len(examples)}, samples per question: {args.n}")

    summary = {}
    for model_id in args.models:
        print(f"\n=== {model_id} ===")
        records, gen_time = probe_model(model_id, examples, gold_rows, args)
        name = model_id.split("/")[-1]
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[model_id] = summarize(records, args.n, gen_time)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print_table(summary)


if __name__ == "__main__":
    main()
