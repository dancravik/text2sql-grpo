import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recover_grpo_pilot


class GrpoRecoveryTests(unittest.TestCase):
    def test_saved_evaluation_is_logged_without_rerunning_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "lr_5e-06"
            training = source / "training"
            exported = source / "exported"
            evaluation = root / "eval" / source.name
            for folder in (training, exported, evaluation):
                folder.mkdir(parents=True)
            report = {"steps_done": 128, "comet_experiment_key": "test-key", "manifest_sha256": "hash"}
            for folder in (training, exported):
                (folder / "run.json").write_text(json.dumps(report))
            metrics = {"num_questions": 200, "greedy_execution_accuracy": 0.5}
            (evaluation / "metrics.json").write_text(json.dumps({
                "manifest_sha256": "hash", "args": {"n": 1, "temperature": 0},
                "datasets": {"validation_fast": metrics},
            }))
            (evaluation / "validation_fast.jsonl").write_text("\n".join(
                json.dumps({"id": str(i), "completions": [{"outcome": "correct" if i < 100 else "wrong"}]})
                for i in range(200)
            ))
            original = {"training": [{"source": str(source), "learning_rate": 5e-6}],
                        "failed": [{"error": "KeyError: global_step"}]}
            (root / "selection.json").write_text(json.dumps(original))
            experiment = Mock()
            experiment.flush.return_value = True
            with patch.object(sys, "argv", ["recover", "--run-dir", str(root)]), \
                 patch.object(recover_grpo_pilot, "start_experiment", return_value=experiment), \
                 patch.object(recover_grpo_pilot.subprocess, "run") as backup:
                recover_grpo_pilot.main()
            self.assertEqual(json.loads((training / "run.json").read_text())["global_step"], 128)
            self.assertEqual(json.loads((exported / "run.json").read_text())["global_step"], 128)
            summary = json.loads((root / "selection_recovered.json").read_text())
            self.assertEqual(len(summary["completed"]), 1)
            self.assertEqual(summary["failed"], [])
            self.assertEqual(json.loads((root / "selection.json").read_text()), original)
            experiment.log_metrics.assert_called_once_with({
                "validation_fast/num_questions": 200,
                "validation_fast/greedy_execution_accuracy": 0.5,
            }, step=128)
            backup.assert_called_once()
            self.assertIn("-c", backup.call_args.args[0])
