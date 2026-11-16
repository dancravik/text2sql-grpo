import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import probe_sft_batch
import train_sft


class SFTBatchProbeTests(unittest.TestCase):
    def test_sweep_logs_oom_and_continues_with_larger_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep"
            experiment = Mock()
            experiment.get_key.return_value = "sweep-key"
            common = {
                "peak_allocated_gib": 12.0, "peak_reserved_gib": 14.0, "gpu_total_gib": 15.0,
                "optimizer_steps_applied": 3, "train_seconds": 42.0, "comet_experiment_key": "child-key",
            }
            reports = [common | {"status": status} for status in ("ok", "cuda_oom", "cuda_oom")]
            argv = ["probe_sft_batch.py", "--out-dir", str(output), "--revision", "model-sha",
                    "--micro-batches", "1", "2", "4"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(probe_sft_batch, "write_long_manifest", return_value=(output / "manifest.json", 1400, 2722)),
                patch.object(probe_sft_batch, "start_experiment", return_value=experiment),
                patch.object(probe_sft_batch, "probe", side_effect=reports),
            ):
                probe_sft_batch.main()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["recommended_micro_batch_size"], 1)
            calls = experiment.log_metrics.call_args_list
            self.assertEqual([call.kwargs["step"] for call in calls], [1, 2, 4])
            self.assertEqual([call.args[0]["fits"] for call in calls], [1, 0, 0])
            self.assertNotIn("train_seconds", calls[1].args[0])
            experiment.end.assert_called_once()

    def test_failed_batch_probe_writes_report_before_returning_oom_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch_2x4"
            experiment = Mock()
            report = {
                "status": "cuda_oom", "error": "test CUDA OOM", "global_step": 0, "failed_at_step": 1,
                "train_seconds": 2.0, "gpu_hours": 2.0 / 3600, "peak_allocated_gib": 13.0,
                "peak_reserved_gib": 14.0,
            }
            argv = ["train_sft.py", "--out-dir", str(output), "--max-steps", "3", "--batch-probe", "--comet"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(train_sft, "start_experiment", return_value=experiment),
                patch.object(train_sft, "train", return_value=report),
                self.assertRaises(SystemExit) as error,
            ):
                train_sft.main()
            self.assertEqual(error.exception.code, train_sft.BATCH_PROBE_OOM_EXIT)
            self.assertEqual(json.loads((output / "run.json").read_text()), report)
            self.assertEqual(experiment.log_metrics.call_args.kwargs["step"], 1)
            self.assertEqual(experiment.log_metrics.call_args.args[0]["fits"], 0)
            experiment.end.assert_called_once()


if __name__ == "__main__":
    unittest.main()
