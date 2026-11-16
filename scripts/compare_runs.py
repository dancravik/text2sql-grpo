"""Build the four-model comparison from saved greedy evaluations on CPU."""

import argparse
import json
import sys
from pathlib import Path

from evaluate import DEFAULT_MANIFEST, EVAL_SPLITS, OUTCOMES, read_manifest, sha256_file

VARIANTS = {
    "base": "Base",
    "sft": "SFT",
    "grpo": "GRPO",
    "sft_grpo": "SFT + GRPO",
}
MATCHED_SETTINGS = ("n", "temperature", "max_tokens", "max_model_len", "seed", "enable_thinking", "smoke")
COMPARISONS = (("base", "sft"), ("base", "grpo"), ("sft", "sft_grpo"), ("grpo", "sft_grpo"))


def read_run(directory: Path, manifest: dict, manifest_hash: str) -> tuple[dict, dict]:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if report["manifest_sha256"] != manifest_hash:
        raise ValueError(f"{directory}: evaluation uses a different split manifest")
    settings = {key: report["args"][key] for key in MATCHED_SETTINGS}
    settings["vllm_version"] = report.get("vllm_version")
    if settings["n"] != 1 or settings["temperature"] != 0:
        raise ValueError(f"{directory}: expected one greedy answer per question")

    datasets = {}
    for name, metrics in report["datasets"].items():
        if name not in EVAL_SPLITS:
            raise ValueError(f"{directory}: unknown evaluation split {name}")
        expected = manifest["splits"][name]["ids"]
        if settings["smoke"]:
            expected = expected[:20]
        records = {}
        with (directory / f"{name}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                example_id = record["id"]
                if example_id in records:
                    raise ValueError(f"{directory}/{name}: duplicate question {example_id}")
                (answer,) = record["completions"]
                if "outcome_allow_untagged" not in answer:
                    raise ValueError(f"{directory}: evaluate or rescore with the current evaluate.py")
                if answer["outcome"] not in OUTCOMES or answer["outcome_allow_untagged"] not in OUTCOMES:
                    raise ValueError(f"{directory}/{name}: unknown outcome for {example_id}")
                records[example_id] = answer
        if not expected or records.keys() != set(expected) or len(records) != metrics["num_questions"]:
            raise ValueError(f"{directory}/{name}: saved question IDs do not match the manifest or metrics")
        datasets[name] = records
    if not datasets:
        raise ValueError(f"{directory}: no evaluated splits")
    return settings, datasets


def correctness_changes(before: dict, after: dict, outcome: str) -> tuple[int, int]:
    fixed = lost = 0
    for example_id, previous in before.items():
        was_correct = previous[outcome] == "correct"
        is_correct = after[example_id][outcome] == "correct"
        fixed += is_correct and not was_correct
        lost += was_correct and not is_correct
    return fixed, lost


def render_comparison(runs: dict) -> str:
    base_settings, base_datasets = runs["base"]
    for variant, (settings, datasets) in runs.items():
        differences = [key for key in base_settings if settings[key] != base_settings[key]]
        if differences:
            raise ValueError(f"{VARIANTS[variant]} differs from Base on {', '.join(differences)}")
        if datasets.keys() != base_datasets.keys():
            raise ValueError(f"{VARIANTS[variant]} evaluated different splits from Base")

    lines = ["# Сравнение моделей", ""]
    for split, base_records in base_datasets.items():
        count = len(base_records)
        lines.extend([
            f"## {split} · {count} вопросов" + (" · smoke" if base_settings["smoke"] else ""),
            "",
            "| Модель | Execution accuracy | Точность с допуском SQL без тегов | Формат соблюдён |",
            "|---|---:|---:|---:|",
        ])
        for variant, label in VARIANTS.items():
            answers = runs[variant][1][split].values()
            strict = sum(answer["outcome"] == "correct" for answer in answers)
            relaxed = sum(answer["outcome_allow_untagged"] == "correct" for answer in answers)
            formatted = sum(answer["format_ok"] for answer in answers)
            lines.append(f"| {label} | {strict / count:.2%} | {relaxed / count:.2%} | {formatted / count:.2%} |")

        lines.extend([
            "",
            "| Сравнение | Исправлено с тегами | Потеряно с тегами | Исправлено с допуском | Потеряно с допуском |",
            "|---|---:|---:|---:|---:|",
        ])
        for before, after in COMPARISONS:
            old_records, new_records = runs[before][1][split], runs[after][1][split]
            fixed, lost = correctness_changes(old_records, new_records, "outcome")
            relaxed_fixed, relaxed_lost = correctness_changes(old_records, new_records, "outcome_allow_untagged")
            lines.append(
                f"| {VARIANTS[before]} → {VARIANTS[after]} | {fixed} | {lost} | {relaxed_fixed} | {relaxed_lost} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for variant, label in VARIANTS.items():
        parser.add_argument(f"--{variant.replace('_', '-')}", type=Path, required=True, help=f"Evaluation of {label}")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    directories = {variant: getattr(args, variant).resolve() for variant in VARIANTS}
    if len(set(directories.values())) != len(VARIANTS):
        parser.error("Provide four distinct evaluation directories")
    try:
        manifest = read_manifest(args.manifest)
        manifest_hash = sha256_file(args.manifest)
        runs = {variant: read_run(path, manifest, manifest_hash) for variant, path in directories.items()}
        print(render_comparison(runs))
    except (OSError, KeyError, ValueError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
