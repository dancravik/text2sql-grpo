"""Оценка модели на фиксированных выборках: execution accuracy и разбивка ошибок.

По умолчанию greedy (temperature 0, один ответ на вопрос): доля верных ответов = execution accuracy.
С --n 8 --temperature 0.8 получаются pass@1 и pass@8 при сэмплировании.
--from-run <dir> добавляет новые метрики к сохранённой генерации без загрузки модели.

Списки ID и исключения загружаются из splits/v1.json без повторной фильтрации.
По умолчанию используется validation. Финальные test_spider/test_bird запускаются явно.

Модель — id на Hugging Face или локальная папка (для обученной: базовая модель с влитым LoRA).

Запуск (из корня репозитория):
    DATA_DIR=/kaggle/input/<dataset> python scripts/evaluate.py --model Qwen/Qwen3.5-2B --smoke
    DATA_DIR=/kaggle/input/<dataset> python scripts/evaluate.py --model Qwen/Qwen3.5-2B --datasets validation_fast
"""

import argparse
import importlib.metadata
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import DATA_DIR  # noqa: E402
from comet_tracking import start_experiment, training_step  # noqa: E402
from prompt import render_prompt  # noqa: E402
from reward import compare, execute  # noqa: E402
from sampling import load_llm, sample_and_score  # noqa: E402
from splits import DEFAULT_MANIFEST, load_split, read_manifest, sha256_file  # noqa: E402

EVAL_SPLITS = ("validation", "validation_fast", "test_spider", "test_bird", "train")
OUTCOMES = ["no_sql", "exec_error", "wrong", "correct"]
RAW_SQL = re.compile(r"^(?:SELECT|WITH)\b", re.IGNORECASE)
SQL_FENCE = re.compile(r"```(?:sql)?\s*\n(.*?)\n?```", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--from-run", type=Path, help="Re-score an existing evaluation directory on CPU")
    p.add_argument("--revision", help="Hugging Face revision for the model and tokenizer")
    p.add_argument("--name", default=None, help="имя папки с результатами; по умолчанию последняя часть --model")
    p.add_argument("--datasets", nargs="+", choices=EVAL_SPLITS, default=["validation"])
    p.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    p.add_argument("--comet-run", type=Path, help="SFT run.json to attach evaluation metrics to its Comet run")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--low-memory", action="store_true", help="Use eager vLLM with at most 8 concurrent sequences")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--enable-thinking", choices=["on", "off"], default="off",
        help="enable_thinking для chat template; по умолчанию off (ответ сразу в <answer>)",
    )
    p.add_argument("--out-dir", default="results/eval")
    p.add_argument("--smoke", action="store_true", help="первые 20 вопросов каждого набора")
    args = p.parse_args()
    args.enable_thinking = args.enable_thinking == "on"
    if args.from_run is None:
        if args.n < 1 or args.temperature < 0 or (args.n > 1 and args.temperature == 0):
            p.error("Use --n >= 1 and --temperature > 0 when sampling multiple answers")
        args.name = args.name or Path(args.model.rstrip("/")).name
    else:
        args.name = args.name or f"{args.from_run.name}_rescored"
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


def score_untagged(records: list[dict], examples: list[dict]) -> None:
    """Keep the strict result and try plain SQL or a full Markdown fence on no_sql."""
    for record, ex in zip(records, examples, strict=True):
        if record["id"] != ex["id"]:
            raise ValueError(f"Saved example ID does not match the manifest: {record['id']}")
        gold_rows = None
        for completion in record["completions"]:
            outcome = completion["outcome"]
            if outcome == "no_sql" and completion["finish_reason"] != "length":
                sql = (completion.get("prefix", "") + completion["text"]).strip()
                fence = SQL_FENCE.fullmatch(sql)
                if fence:
                    sql = fence.group(1).strip()
                if RAW_SQL.match(sql):
                    if gold_rows is None:
                        gold_rows, error = execute(ex["db_path"], ex["gold_sql"])
                        if error is not None:
                            raise RuntimeError(f"Gold SQL failed for {ex['id']}: {error}")
                    rows, error = execute(ex["db_path"], sql)
                    if error is not None:
                        outcome = "exec_error"
                    else:
                        outcome = "correct" if compare(rows, gold_rows, ex["gold_sql"]) else "wrong"
            completion["outcome_allow_untagged"] = outcome


def summarize(records: list[dict], n: int, gen_time: float, temperature: float) -> dict:
    """Метрики по одному набору; для BIRD — ещё точность по сложности."""
    if not records or any(len(r["completions"]) != n for r in records):
        raise ValueError(f"Expected a nonempty evaluation with {n} completions per question")
    comps = [c for r in records for c in r["completions"]]
    outcomes = Counter(c["outcome"] for c in comps)
    metrics = {
        "num_questions": len(records),
        # Greedy и n=1 — execution accuracy; при сэмплировании — pass@1
        "accuracy": accuracy(records, n),
    }
    one_answer = "greedy_execution_accuracy" if temperature == 0 else "sampled_pass@1"
    metrics[one_answer] = metrics["accuracy"]
    metrics[f"{one_answer}_allow_untagged"] = sum(
        c["outcome_allow_untagged"] == "correct" for c in comps
    ) / len(comps)
    metrics["out_no_sql_allow_untagged"] = sum(c["outcome_allow_untagged"] == "no_sql" for c in comps) / len(comps)
    if n > 1:
        metrics[f"pass@{n}"] = sum(r["k_correct"] > 0 for r in records) / len(records)
        metrics[f"pass@{n}_allow_untagged"] = sum(
            any(c["outcome_allow_untagged"] == "correct" for c in r["completions"]) for r in records
        ) / len(records)
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
    manifest_path = Path(args.manifest)
    manifest_sha = sha256_file(manifest_path)
    saved = None
    if args.from_run is not None:
        saved = json.loads((args.from_run / "metrics.json").read_text(encoding="utf-8"))
        if saved["manifest_sha256"] != manifest_sha:
            raise ValueError("The saved evaluation uses a different dataset manifest")
        args.model = saved["model"]
        args.datasets = list(saved["datasets"])
        args.n = saved["args"]["n"]
        args.temperature = saved["args"]["temperature"]
        args.smoke = saved["args"]["smoke"]
    training_run = None
    if args.comet_run is not None:
        training_run = json.loads(args.comet_run.read_text(encoding="utf-8"))
        training_step(training_run)  # Validate metadata before allocating GPU memory.
        if not training_run.get("comet_experiment_key"):
            raise ValueError("The SFT run has no Comet experiment. Train with --comet first.")
        if training_run["manifest_sha256"] != manifest_sha:
            raise ValueError("The SFT run and evaluation use different dataset manifests")
        if Path(args.model).resolve() != (args.comet_run.parent / "merged").resolve():
            raise ValueError("--model must point to the merged model next to the SFT run.json")

    out_dir = Path(args.out_dir) / args.name
    if args.from_run is not None and out_dir.resolve() == args.from_run.resolve():
        raise ValueError("Choose a separate output directory for the re-score")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(manifest_path)
    # Verify data before allocating the GPU; never silently remove failed golds.
    datasets = {name: load_split(name, DATA_DIR, manifest_path) for name in args.datasets}
    llm = (load_llm(args.model, args.max_model_len, args.seed, revision=args.revision, low_memory=args.low_memory)
           if saved is None else None)
    report = {
        "model": args.model,
        "args": saved["args"] if saved is not None else {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "datasets": {}, "manifest_sha256": manifest_sha,
    }
    if saved is not None:
        report["rescored_from"] = str(args.from_run)
        if "vllm_version" in saved:
            report["vllm_version"] = saved["vllm_version"]
    else:
        report["vllm_version"] = importlib.metadata.version("vllm")

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        examples = datasets[dataset]
        num_excluded = len(manifest["splits"][dataset]["excluded"])
        if args.smoke:
            examples = examples[:20]
        if saved is None:
            check_lengths(llm.get_tokenizer(), examples, args)
            # Bound the number of SQLite result sets and vLLM outputs retained during the train audit.
            chunk = 96 if dataset == "train" else len(examples)
            records, gen_time = [], 0.0
            for offset in range(0, len(examples), chunk):
                batch, seconds = sample_and_score(
                    llm, examples[offset:offset + chunk], args.n, args.temperature,
                    args.max_tokens, args.seed, args.enable_thinking
                )
                records.extend(batch)
                gen_time += seconds
            records = [
                {**{k: ex[k] for k in ("question", "evidence", "gold_sql", "db_id", "difficulty")}, **r}
                for ex, r in zip(examples, records, strict=True)
            ]
        else:
            with (args.from_run / f"{dataset}.jsonl").open(encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            gen_time = saved["datasets"][dataset]["gen_time_s"]
        score_untagged(records, examples)
        metrics = {"num_excluded": num_excluded, **summarize(records, args.n, gen_time, args.temperature)}
        with open(out_dir / f"{dataset}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        report["datasets"][dataset] = metrics
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"accuracy {metrics['accuracy']:.3f} on {metrics['num_questions']} questions ({num_excluded} excluded)")
        for name, value in metrics.items():
            if name.startswith(("greedy_execution_accuracy", "sampled_pass@", "pass@")):
                print(f"  {name}: {value:.3f}")
        for o in OUTCOMES:
            print(f"  {o:<11}{metrics[f'out_{o}']:.3f}")
        for d, m in metrics.get("accuracy_by_difficulty", {}).items():
            print(f"  {d:<11}{m['accuracy']:.3f} ({m['num_questions']})")

    print(f"\nSaved to {out_dir}")
    if training_run is not None:
        experiment = start_experiment(key=training_run["comet_experiment_key"])
        try:
            prefix = ("smoke/" if args.smoke else "") + (f"sampled_n{args.n}/" if args.temperature > 0 else "")
            for dataset, metrics in report["datasets"].items():
                experiment.log_metrics(
                    {f"{prefix}{dataset}/{name}": value for name, value in metrics.items()
                     if isinstance(value, (int, float))},
                    step=training_step(training_run),
                )
        finally:
            experiment.end()


if __name__ == "__main__":
    main()
