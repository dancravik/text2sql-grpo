"""Evaluate saved GRPO snapshots sequentially and back them up to Comet."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pilot_grpo_lr import BACKUP, ROOT, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if any(step < 1 for step in args.steps) or len(set(args.steps)) != len(args.steps):
        parser.error("Snapshot steps must be positive and distinct")
    root = args.run_dir.resolve()
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["failed"]:
        parser.error("Resolve failed training/final evaluation stages before evaluating snapshots")
    dataset = selection["dataset"]
    completed = {Path(row["source"]).name for row in selection["completed"]}
    for trial in selection["training"]:
        label = Path(trial["source"]).name
        for step in sorted(args.steps):
            snapshot_label = f"{label}_step_{step}"
            if snapshot_label in completed:
                print(f"Already completed: {snapshot_label}", flush=True)
                continue
            checkpoint = root / label / "training/trainer" / f"checkpoint-{step}"
            report = json.loads((checkpoint / "run.json").read_text())
            if report.get("global_step") != step or not report.get("snapshot"):
                raise ValueError(f"Invalid snapshot metadata: {checkpoint}")
            source = root / "snapshots" / snapshot_label
            source.mkdir(parents=True, exist_ok=True)
            if not (source / "adapter").exists():
                shutil.copytree(checkpoint, source / "adapter")
            (source / "run.json").write_text(json.dumps(report, indent=2) + "\n")
            exported = source / "exported"
            evaluation = root / "eval" / snapshot_label
            metrics_path = evaluation / "metrics.json"
            if not metrics_path.exists():
                if not (exported / "run.json").exists():
                    execute("export_sft.py", ["--run-dir", source, "--out-dir", exported],
                            source / "export.log", {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                execute("evaluate.py", [
                    "--model", exported / "merged", "--name", snapshot_label,
                    "--datasets", dataset, "--out-dir", evaluation.parent,
                    "--comet-run", exported / "run.json", "--n", 1, "--temperature", 0,
                    "--enable-thinking", "off", "--max-tokens", 1024,
                    "--max-model-len", 4096, "--seed", 42,
                ], source / "evaluation.log", {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu})
            evaluation_report = json.loads(metrics_path.read_text())
            if evaluation_report["manifest_sha256"] != report["manifest_sha256"]:
                raise ValueError("Snapshot and evaluation manifests differ")
            subprocess.run([sys.executable, "-u", "-c", BACKUP, str(source), str(evaluation), str(ROOT),
                            f"grpo_snapshot_{step}_artifacts.zip"],
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
            metrics = evaluation_report["datasets"][dataset]
            selection["completed"].append({
                "learning_rate": trial["learning_rate"], "source": str(source),
                "global_step": step, "metrics": metrics,
            })
            selection_path.write_text(json.dumps(selection, indent=2) + "\n")
            completed.add(snapshot_label)
            if (exported / "merged").exists():
                shutil.rmtree(exported / "merged")
            print(f"{snapshot_label}: {metrics['greedy_execution_accuracy']:.2%}", flush=True)


if __name__ == "__main__":
    main()
