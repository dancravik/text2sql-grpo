import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_grpo_snapshots


class SnapshotEvaluationTests(unittest.TestCase):
    def test_saved_steps_are_evaluated_once_and_keep_separate_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = root / "lr_5e-06"
            selection = {
                "dataset": "validation", "failed": [], "completed": [],
                "training": [{"source": str(trial), "learning_rate": 5e-6}],
            }
            (root / "selection.json").write_text(json.dumps(selection))
            for step in [128, 256]:
                checkpoint = trial / "training/trainer" / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_model.safetensors").write_bytes(b"saved weights")
                (checkpoint / "run.json").write_text(json.dumps({
                    "snapshot": True, "global_step": step, "manifest_sha256": "hash",
                }))

            def execute(script, arguments, logfile, env):
                if script == "export_sft.py":
                    exported = Path(arguments[3])
                    (exported / "merged").mkdir(parents=True)
                    (exported / "run.json").write_text("{}")
                else:
                    options = dict(zip(arguments[::2], arguments[1::2]))
                    evaluation = Path(options["--out-dir"]) / options["--name"]
                    evaluation.mkdir(parents=True)
                    (evaluation / "metrics.json").write_text(json.dumps({
                        "manifest_sha256": "hash", "datasets": {"validation": {
                            "greedy_execution_accuracy": 0.72,
                        }},
                    }))

            with patch.object(sys, "argv", ["evaluate", "--run-dir", str(root)]), \
                 patch.object(evaluate_grpo_snapshots, "execute", side_effect=execute) as run, \
                 patch.object(evaluate_grpo_snapshots.subprocess, "run") as backup:
                evaluate_grpo_snapshots.main()
                self.assertEqual(run.call_count, 4)
                self.assertEqual(backup.call_args_list[0].args[0][-1], "grpo_snapshot_128_artifacts.zip")
                self.assertEqual(backup.call_args_list[1].args[0][-1], "grpo_snapshot_256_artifacts.zip")
                evaluate_grpo_snapshots.main()
                self.assertEqual(run.call_count, 4)
            completed = json.loads((root / "selection.json").read_text())["completed"]
            self.assertEqual([row["global_step"] for row in completed], [128, 256])
            self.assertTrue((root / "snapshots/lr_5e-06_step_128/adapter/adapter_model.safetensors").exists())
