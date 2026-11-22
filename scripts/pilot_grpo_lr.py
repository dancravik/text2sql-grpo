"""Train independent GRPO LR pilots, then export and evaluate sequentially."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

BACKUP = r'''
import json, sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from comet_tracking import start_experiment
source, evaluation = map(Path, sys.argv[1:3])
archive = source / (sys.argv[4] if len(sys.argv) > 4 else "grpo_pilot_artifacts.zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    z.write(source / "run.json", "run.json")
    for path in (source / "adapter").rglob("*"):
        if path.is_file():
            z.write(path, "adapter/" + path.relative_to(source / "adapter").as_posix())
    for path in source.glob("*.log"):
        z.write(path, "logs/" + path.name)
    for path in evaluation.glob("*.json*"):
        z.write(path, "evaluation/" + path.name)
    repo = Path(sys.argv[3])
    z.write(repo / "splits/v1.json", "splits/v1.json")
report = json.loads((source / "run.json").read_text())
experiment = start_experiment(key=report["comet_experiment_key"])
try:
    experiment.log_asset(str(archive), file_name=archive.name, overwrite=True)
    if not experiment.flush(timeout=300):
        raise RuntimeError("Comet did not confirm artifact delivery; local archive retained")
finally:
    experiment.end()
'''


def execute(script, arguments, logfile, env):
    with logfile.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, arguments)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                lines = logfile.read_text(errors="replace").splitlines()
                last = lines[-1][-250:] if lines else "starting"
                print(f"[{logfile.parent.name}/{script}] alive: {last}", flush=True)
    if code:
        tail = "\n".join(logfile.read_text(errors="replace").splitlines()[-15:])
        raise RuntimeError(f"{script}: exit {code}; log {logfile}\n{tail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-6, 1e-5])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--init-sft-run", type=Path, help="Completed SFT run with adapter/ and run.json")
    parser.add_argument("--dataset", choices=["validation_fast", "validation"], default="validation_fast")
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--num-questions", type=int, default=1024)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("GPU IDs must be distinct")
    if len(args.lrs) > len(args.gpus) or len(args.lrs) != len(set(args.lrs)):
        parser.error("Use distinct learning rates and at most one training per GPU")
    if args.steps < 1 or args.save_steps < 0 or any(not 0 < lr < 1 for lr in args.lrs):
        parser.error("Invalid steps or learning rate")
    if args.num_questions < 1 or args.keep_checkpoints < 1:
        parser.error("Question count and checkpoint retention must be positive")
    branch = "sft_grpo" if args.init_sft_run else "base_grpo"
    initialization = []
    if args.init_sft_run:
        args.init_sft_run = args.init_sft_run.resolve()
        if not (args.init_sft_run / "adapter/adapter_model.safetensors").is_file():
            parser.error("SFT adapter weights are missing")
        initialization = ["--init-sft-run", str(args.init_sft_run)]
    if args.out_dir.exists():
        parser.error("Output directory exists; choose a new directory")
    if not os.environ.get("COMET_API_KEY") or not os.environ.get("DATA_DIR"):
        parser.error("Set COMET_API_KEY and DATA_DIR first")
    args.out_dir = args.out_dir.resolve()
    if shutil.disk_usage(args.out_dir.parent).free < 10 * 1024**3:
        parser.error("Leave at least 10 GiB of disk space")
    args.out_dir.mkdir()
    summary = {"branch": branch, "stage": "lr_pilot", "dataset": args.dataset,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "completed": [], "failed": [], "training": []}

    def save():
        (args.out_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")

    def train(lr, gpu):
        label = f"lr_{lr:g}"
        source = args.out_dir / label
        source.mkdir()
        print(f"Training {label} on GPU {gpu}; log: {source / 'train.log'}", flush=True)
        execute("feasibility_grpo.py", [
            "--model", "Qwen/Qwen3.5-2B", "--revision", "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "--manifest", ROOT / "splits/v1.json", "--out-dir", source / "training",
            "--name", f"{args.out_dir.name}_{label}", "--branch", branch, "--comet", "--save-adapter",
            "--save-steps", args.save_steps, *initialization,
            "--keep-checkpoints", args.keep_checkpoints,
            "--num-questions", args.num_questions, "--sample-policy", "random", "--prompts-per-step", 2,
            "--num-generations", 4, "--micro-batch-size", args.micro_batch,
            "--max-completion-length", 1024, "--max-steps", args.steps,
            "--lr", lr, "--lr-scheduler-type", "constant", "--beta", 0,
            "--temperature", 0.8, "--enable-thinking", "off", "--use-vllm", "none", "--seed", 42,
        ], source / "train.log", {**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        print(f"Training finished: {label}", flush=True)
        return {"learning_rate": lr, "source": str(source), "gpu": gpu}

    # No merging or vLLM while either training process is still alive.
    with ThreadPoolExecutor(max_workers=len(args.lrs)) as pool:
        jobs = {pool.submit(train, lr, gpu): lr for lr, gpu in zip(args.lrs, args.gpus)}
        for future in as_completed(jobs):
            try:
                summary["training"].append(future.result())
            except Exception as error:
                summary["failed"].append({"learning_rate": jobs[future], "stage": "training", "error": str(error)})
                print(str(error), flush=True)
            save()

    for trial in sorted(summary["training"], key=lambda row: row["learning_rate"]):
        source = Path(trial["source"])
        training = source / "training"
        exported = source / "exported"
        evaluation = args.out_dir / "eval" / source.name
        try:
            print(f"Sequential export and evaluation: {source.name}", flush=True)
            execute("export_sft.py", ["--run-dir", training, "--out-dir", exported],
                    source / "export.log", {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            execute("evaluate.py", [
                "--model", exported / "merged", "--name", source.name,
                "--datasets", args.dataset, "--out-dir", evaluation.parent,
                "--comet-run", exported / "run.json", "--n", 1, "--temperature", 0,
                "--enable-thinking", "off", "--max-tokens", 1024, "--max-model-len", 4096, "--seed", 42,
            ], source / "evaluation.log", {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus[0]})
            # Keep all pilot logs beside the source report for the archive.
            for logfile in source.glob("*.log"):
                shutil.copy2(logfile, training / logfile.name)
            subprocess.run([sys.executable, "-u", "-c", BACKUP, str(training), str(evaluation), str(ROOT)],
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
            metrics = json.loads((evaluation / "metrics.json").read_text())["datasets"][args.dataset]
            summary["completed"].append({**trial, "metrics": metrics})
            shutil.rmtree(exported / "merged")
            print(f"{source.name}: accuracy {metrics['greedy_execution_accuracy']:.2%}", flush=True)
        except Exception as error:
            summary["failed"].append({**trial, "stage": "export_or_evaluation", "error": str(error)})
            print(str(error), flush=True)
        save()

    print(f"Summary: {args.out_dir / 'selection.json'}", flush=True)
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
