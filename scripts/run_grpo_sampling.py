"""Run a budget-matched ordinary/dynamic GRPO comparison from saved SFT weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from pilot_grpo_lr import ROOT, execute

BASE = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def predictions(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != 956 or len(by_id) != 956 or any(len(row["completions"]) != 1 for row in rows):
        raise ValueError(f"Expected 956 unique greedy predictions: {path}")
    return by_id


def correct(row):
    return row["completions"][0]["outcome"] == "correct"


def paired(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("Evaluation question IDs differ")
    for key, row in candidate.items():
        if any(row[field] != reference[key][field] for field in ("question", "gold_sql", "db_id")):
            raise ValueError("Evaluation content differs")
    gains = sum(correct(row) and not correct(reference[key]) for key, row in candidate.items())
    losses = sum(not correct(row) and correct(reference[key]) for key, row in candidate.items())
    return {"gains": gains, "losses": losses, "delta_percentage_points": (gains - losses) / 956 * 100}


def compare(folder, sft, budget):
    reference = predictions(sft / "evaluation/validation.jsonl")
    baseline = json.loads((sft / "evaluation/metrics.json").read_text())
    original_hash = hashlib.sha256((sft / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    selection = json.loads((folder / "selection.json").read_text())
    if selection["failed"] or len(selection["completed"]) != 2:
        raise ValueError("Both sampling runs must finish successfully")
    results, streams, generated = {}, [], {}
    for trial in selection["completed"]:
        mode = trial["sampling_mode"]
        source = folder / Path(trial["source"]).name
        training = json.loads((source / "training/run.json").read_text())
        evaluation = folder / "eval" / source.name
        report = json.loads((evaluation / "metrics.json").read_text())
        if (training["question_presentations"] != budget
                or training["sampling_metrics"]["generated_groups"] != budget
                or len(training["generated_question_ids"]) != budget):
            raise ValueError("Generated question budgets differ")
        if training["initial_sft"]["adapter_sha256"] != original_hash:
            raise ValueError("Starting SFT weights differ")
        if report["manifest_sha256"] != baseline["manifest_sha256"]:
            raise ValueError("Evaluation manifests differ")
        for name in ("n", "temperature", "seed", "max_tokens", "max_model_len", "enable_thinking"):
            if report["args"][name] != baseline["args"][name]:
                raise ValueError(f"Evaluation setting differs: {name}")
        rows = predictions(evaluation / "validation.jsonl")
        count = sum(correct(row) for row in rows.values())
        if abs(count / 956 - report["datasets"]["validation"]["greedy_execution_accuracy"]) > 1e-12:
            raise ValueError("Saved metrics disagree with predictions")
        generated[mode] = rows
        streams.append(training["generated_question_ids"])
        results[mode] = {
            "correct": count, "accuracy": count / 956, "vs_sft": paired(reference, rows),
            "optimizer_steps": training["global_step"],
            "optimizer_steps_skipped": training["optimizer_steps_skipped"],
            "question_presentations": training["question_presentations"],
            "unique_questions": len(training["seen_example_ids"]),
            "generated_completion_tokens": training["generated_completion_tokens"],
            "train_seconds": training["train_seconds"], "gpu_hours": training["gpu_hours"],
            "sampling_metrics": training["sampling_metrics"],
            "metrics": report["datasets"]["validation"],
        }
    if set(results) != {"ordinary", "dynamic"} or streams[0] != streams[1]:
        raise ValueError("Sampling modes must generate the same ordered question stream")
    return {
        "num_questions": 956, "sft_accuracy": sum(correct(row) for row in reference.values()) / 956,
        "generated_question_budget": budget, "same_question_stream": True,
        "results": results, "dynamic_vs_ordinary": paired(generated["ordinary"], generated["dynamic"]),
    }


def make_archive(folder, sft):
    archive = folder.parent / f"{folder.name}_complete.zip"
    complete = (folder / "comparison.json").is_file()
    temporary = archive.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED, allowZip64=True) as bundle:
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder)
            if not path.is_file() or path.suffix == ".zip" or "merged" in relative.parts:
                continue
            if complete and "trainer" in relative.parts:
                continue
            bundle.write(path, folder.name + "/" + relative.as_posix())
        for path in [sft / "run.json", *[p for p in (sft / "adapter").rglob("*") if p.is_file()],
                     sft / "evaluation/metrics.json", sft / "evaluation/validation.jsonl"]:
            bundle.write(path, "initial_sft/" + path.relative_to(sft).as_posix())
        for name in ("scripts", "src"):
            for path in (ROOT / name).rglob("*.py"):
                bundle.write(path, "repository/" + path.relative_to(ROOT).as_posix())
        bundle.write(ROOT / "splits/v1.json", "repository/splits/v1.json")
        bundle.writestr("environment/pip_freeze.txt", subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True))
    with ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Artifact ZIP failed integrity checking")
    temporary.replace(archive)
    print(f"Archive: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--init-sft-run", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--rollout-budget", type=int, default=2048)
    args = parser.parse_args()
    if args.out_dir.exists() or args.rollout_budget < 8 or args.rollout_budget % 2:
        parser.error("Use a new output directory and an even budget of at least 8 question groups")
    if len(set(args.gpus)) != 2:
        parser.error("Use two distinct GPUs")
    args.out_dir = args.out_dir.resolve()
    args.init_sft_run = args.init_sft_run.resolve()
    for name in ("run.json", "adapter/adapter_model.safetensors", "evaluation/validation.jsonl", "evaluation/metrics.json"):
        if not (args.init_sft_run / name).is_file():
            parser.error(f"Missing SFT input: {name}")
    args.out_dir.mkdir(parents=True)
    try:
        print("Checking the dynamic sampling path on GPU", flush=True)
        execute("feasibility_grpo.py", [
            "--model", BASE, "--revision", REVISION, "--branch", "sft_grpo",
            "--init-sft-run", args.init_sft_run, "--out-dir", args.out_dir / "smoke",
            "--name", args.out_dir.name + "_dynamic_smoke", "--comet", "--full-model",
            "--num-questions", 5819, "--prompts-per-step", 2, "--num-generations", 4,
            "--micro-batch-size", 2, "--max-completion-length", 1024, "--max-steps", 4,
            "--lr", 2.5e-6, "--lr-scheduler-type", "constant", "--beta", 0,
            "--sampling-mode", "dynamic", "--rollout-budget", 8, "--max-sampling-rounds", 2,
            "--enable-thinking", "off", "--use-vllm", "none", "--seed", 42,
        ], args.out_dir / "smoke.log", {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus[0]})
        experiment = args.out_dir / "experiment"
        execute("pilot_grpo_lr.py", [
            "--out-dir", experiment, "--init-sft-run", args.init_sft_run, "--gpus", *args.gpus,
            "--lrs", 2.5e-6, "--steps", args.rollout_budget // 2,
            "--sampling-modes", "ordinary", "dynamic", "--rollout-budget", args.rollout_budget,
            "--num-questions", 5819, "--micro-batch", 2, "--save-steps", 128,
            "--keep-checkpoints", 2, "--dataset", "validation",
        ], args.out_dir / "experiment.log", os.environ.copy())
        result = compare(experiment, args.init_sft_run, args.rollout_budget)
        comparison_path = args.out_dir / "comparison.json"
        comparison_path.write_text(json.dumps(result, indent=2) + "\n")
        sys.path.insert(0, str(ROOT / "src"))
        from comet_tracking import start_experiment

        selection = json.loads((experiment / "selection.json").read_text())
        for trial in selection["training"]:
            training = json.loads((Path(trial["source"]) / "training/run.json").read_text())
            mode = trial["sampling_mode"]
            comet = start_experiment(key=training["comet_experiment_key"])
            try:
                comet.log_metrics({
                    "comparison/sft_accuracy": result["sft_accuracy"],
                    "comparison/delta_vs_sft_pp": result["results"][mode]["vs_sft"]["delta_percentage_points"],
                    "comparison/generated_question_budget": args.rollout_budget,
                }, step=training["global_step"])
                comet.log_asset(str(comparison_path), file_name="sampling_comparison.json", overwrite=True)
                if not comet.flush(timeout=300):
                    raise RuntimeError("Comet did not confirm comparison delivery; results remain in the local archive")
            finally:
                comet.end()
        print(json.dumps(result, indent=2), flush=True)
    finally:
        make_archive(args.out_dir, args.init_sft_run)


if __name__ == "__main__":
    main()
