import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_grpo_sampling import compare


class SamplingComparisonTests(unittest.TestCase):
    def test_comparison_checks_question_stream_and_counts_paired_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft = root / "sft"
            (sft / "adapter").mkdir(parents=True)
            (sft / "adapter/adapter_model.safetensors").write_bytes(b"original")
            original_hash = hashlib.sha256(b"original").hexdigest()
            settings = dict(n=1, temperature=0, seed=42, max_tokens=1024,
                            max_model_len=4096, enable_thinking=False)

            def save_evaluation(folder, correct_count):
                folder.mkdir(parents=True)
                rows = [{"id": i, "question": "q", "gold_sql": "SELECT 1", "db_id": "db",
                         "completions": [{"outcome": "correct" if i < correct_count else "wrong"}]}
                        for i in range(956)]
                (folder / "validation.jsonl").write_text("\n".join(map(json.dumps, rows)))
                (folder / "metrics.json").write_text(json.dumps({
                    "args": settings, "manifest_sha256": "manifest", "datasets": {"validation": {
                        "greedy_execution_accuracy": correct_count / 956,
                    }},
                }))

            save_evaluation(sft / "evaluation", 679)
            trials = []
            for mode, count in [("ordinary", 690), ("dynamic", 700)]:
                source = root / mode
                (source / "training").mkdir(parents=True)
                training = {
                    "question_presentations": 8, "generated_question_ids": list(range(8)),
                    "seen_example_ids": list(range(8)), "sampling_metrics": {"generated_groups": 8},
                    "initial_sft": {"adapter_sha256": original_hash}, "global_step": 4,
                    "optimizer_steps_skipped": 0, "generated_completion_tokens": 500,
                    "train_seconds": 10, "gpu_hours": 10 / 3600,
                }
                (source / "training/run.json").write_text(json.dumps(training))
                save_evaluation(root / "eval" / mode, count)
                trials.append({"sampling_mode": mode, "source": str(source)})
            (root / "selection.json").write_text(json.dumps({"failed": [], "completed": trials}))
            result = compare(root, sft, 8)
            self.assertEqual(result["dynamic_vs_ordinary"]["gains"], 10)
            self.assertEqual(result["dynamic_vs_ordinary"]["losses"], 0)
            path = root / "dynamic/training/run.json"
            training = json.loads(path.read_text())
            training["generated_question_ids"].reverse()
            path.write_text(json.dumps(training))
            with self.assertRaises(ValueError):
                compare(root, sft, 8)
