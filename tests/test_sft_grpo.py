import argparse
import ast
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def definitions(*names):
    source = Path(__file__).resolve().parents[1] / "scripts/feasibility_grpo.py"
    tree = ast.parse(source.read_text())
    selected = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])
    namespace = {
        "argparse": argparse, "math": math, "Path": Path, "json": json,
        "DEFAULT_MANIFEST": Path("splits/v1.json"), "TrainerCallback": object,
    }
    exec(compile(selected, str(source), "exec"), namespace)
    return namespace


class SftGrpoTests(unittest.TestCase):
    def test_sft_branch_requires_saved_initial_weights(self):
        parse = definitions("parse_args")["parse_args"]
        with patch.object(sys, "argv", ["grpo", "--model", "base", "--branch", "sft_grpo"]):
            with self.assertRaises(SystemExit):
                parse()
        with patch.object(sys, "argv", ["grpo", "--model", "base", "--branch", "sft_grpo",
                                       "--init-sft-run", "saved-sft", "--comet", "--save-steps", "32"]):
            args = parse()
        self.assertEqual(args.init_sft_run, "saved-sft")
        self.assertEqual(args.save_steps, 32)

    def test_format_bonus_does_not_count_as_correct_sql(self):
        namespace = definitions("StepStats")
        namespace["sql_reward"] = Mock(return_value=[0.1] * 4 + [1.1] * 4 + [0.1, 1.1, 0, 1])
        stats = namespace["StepStats"](num_generations=4)
        values = stats.sql_reward([], [], [], [])
        self.assertEqual(len(values), 12)
        self.assertEqual(stats.reward_groups, {"all_correct": 1, "none_correct": 1, "mixed": 1})

    def test_snapshot_records_actual_step_and_sft_provenance(self):
        namespace = definitions("AdapterSnapshots")
        upload = Mock()
        namespace["upload_training_bundle"] = upload
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "trainer/checkpoint-32"
            checkpoint.mkdir(parents=True)
            stats = SimpleNamespace(experiment=Mock(), optimizer_steps_skipped=0,
                                    reward_groups={"mixed": 20})
            callback = namespace["AdapterSnapshots"](
                {"initial_sft": {"adapter_sha256": "original"}}, stats, Mock(),
                SimpleNamespace(out_dir=root, manifest=root / "manifest.json"),
            )
            callback.on_save(SimpleNamespace(output_dir=root / "trainer"),
                             SimpleNamespace(global_step=32, log_history=[]), None)
            report = json.loads((checkpoint / "run.json").read_text())
            self.assertEqual(report["global_step"], 32)
            self.assertEqual(report["initial_sft"]["adapter_sha256"], "original")
            self.assertEqual(upload.call_args.kwargs["step"], 32)


if __name__ == "__main__":
    unittest.main()
