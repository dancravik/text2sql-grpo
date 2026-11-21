import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from comet_tracking import training_step


class TrainingStepTests(unittest.TestCase):
    def test_current_sft_and_grpo_reports(self):
        self.assertEqual(training_step({"global_step": 128, "steps_done": 120}), 128)

    def test_legacy_grpo_report(self):
        self.assertEqual(training_step({"steps_done": 128, "optimizer_steps_applied": 125}), 128)

    def test_invalid_metadata_fails_before_evaluation(self):
        for report in ({}, {"global_step": -1}, {"global_step": True}, {"steps_done": "128"}):
            with self.subTest(report=report), self.assertRaises(ValueError):
                training_step(report)
