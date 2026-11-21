"""Finish Comet logging and backups for saved GRPO evaluations, using no GPU."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pilot_grpo_lr import BACKUP, ROOT

sys.path.insert(0, str(ROOT / "src"))
from comet_tracking import start_experiment, training_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    original = json.loads((root / "selection.json").read_text())
    recovered = {"branch": "base_grpo", "stage": "lr_pilot", "recovered_from": str(root / "selection.json"),
                 "previous_failures": original.get("failed", []), "completed": [], "failed": []}

    for trial in original["training"]:
        source = root / Path(trial["source"]).name
        training = source / "training"
        evaluation = root / "eval" / source.name
        try:
            report = json.loads((training / "run.json").read_text())
            metrics_report = json.loads((evaluation / "metrics.json").read_text())
            metrics = metrics_report["datasets"]["validation_fast"]
            records = [json.loads(line) for line in (evaluation / "validation_fast.jsonl").read_text().splitlines() if line.strip()]
            if metrics_report["manifest_sha256"] != report["manifest_sha256"]:
                raise ValueError("Evaluation and training manifest mismatch")
            if metrics["num_questions"] != 200 or len(records) != 200 or len({r["id"] for r in records}) != 200:
                raise ValueError("Expected 200 unique saved validation questions")
            if metrics_report["args"]["n"] != 1 or metrics_report["args"]["temperature"] != 0:
                raise ValueError("Expected a greedy evaluation")
            outcomes = Counter(row["completions"][0]["outcome"] for row in records)
            if abs(outcomes["correct"] / 200 - metrics["greedy_execution_accuracy"]) > 1e-12:
                raise ValueError("Saved predictions and accuracy disagree")
            step = training_step(report)
            # Add the current field without discarding the legacy metadata.
            for path in (training / "run.json", source / "exported" / "run.json"):
                if path.exists():
                    data = json.loads(path.read_text())
                    if "global_step" not in data:
                        data["global_step"] = step
                        path.write_text(json.dumps(data, indent=2) + "\n")

            experiment = start_experiment(key=report["comet_experiment_key"])
            try:
                experiment.log_metrics({
                    f"validation_fast/{key}": value for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }, step=step)
                experiment.log_other("evaluation_logging_recovered", True)
                if not experiment.flush(timeout=300):
                    raise RuntimeError("Comet metric delivery not confirmed")
            finally:
                experiment.end()

            for logfile in source.glob("*.log"):
                shutil.copy2(logfile, training / logfile.name)
            subprocess.run([sys.executable, "-u", "-c", BACKUP, str(training), str(evaluation), str(ROOT)],
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
            recovered["completed"].append({**trial, "metrics": metrics})
            print(f"Recovered LR {trial['learning_rate']:g}: {metrics['greedy_execution_accuracy']:.2%}", flush=True)
        except Exception as error:
            recovered["failed"].append({**trial, "error": str(error)})
            print(f"Recovery failed: {source.name}: {error}", flush=True)
        (root / "selection_recovered.json").write_text(json.dumps(recovered, indent=2) + "\n")

    print(f"Recovered summary: {root / 'selection_recovered.json'}")
    if recovered["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
