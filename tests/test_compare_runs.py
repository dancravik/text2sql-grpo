import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_runs import read_run, render_comparison


class CompareRunsTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"splits": {"validation": {"ids": ["q0", "q1"]}}}
        self.settings = {
            "n": 1, "temperature": 0, "max_tokens": 1024, "max_model_len": 4096,
            "seed": 42, "enable_thinking": False, "smoke": False,
        }
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write_run(self, name, rows, **settings):
        directory = Path(self.tmp.name) / name
        directory.mkdir()
        report = {
            "args": self.settings | settings,
            "manifest_sha256": "test-hash",
            "vllm_version": "test-version",
            "datasets": {"validation": {"num_questions": len(rows)}},
        }
        (directory / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
        records = [
            {"id": example_id, "completions": [{
                "outcome": outcome,
                "outcome_allow_untagged": relaxed,
                "format_ok": outcome != "no_sql",
            }]}
            for example_id, outcome, relaxed in rows
        ]
        (directory / "validation.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
        )
        return read_run(directory, self.manifest, "test-hash")

    def test_pairs_questions_by_id_even_if_files_have_different_order(self):
        base = self.write_run("base", [("q0", "correct", "correct"), ("q1", "no_sql", "correct")])
        grpo = self.write_run("grpo", [("q1", "correct", "correct"), ("q0", "wrong", "wrong")])
        table = render_comparison({"base": base, "sft": base, "grpo": grpo, "sft_grpo": grpo})
        self.assertIn("| Base | 50.00% | 100.00% | 50.00% |", table)
        self.assertIn("| Base → GRPO | 1 | 1 | 0 | 1 |", table)

    def test_rejects_different_question_ids(self):
        with self.assertRaisesRegex(ValueError, "question IDs do not match"):
            self.write_run("base", [("q0", "correct", "correct"), ("other", "wrong", "wrong")])

    def test_rejects_comparison_with_different_generation_limits(self):
        rows = [("q0", "correct", "correct"), ("q1", "wrong", "wrong")]
        base = self.write_run("base", rows)
        sft = self.write_run("sft", rows, max_tokens=512)
        with self.assertRaisesRegex(ValueError, "SFT differs from Base on max_tokens"):
            render_comparison({"base": base, "sft": sft})


if __name__ == "__main__":
    unittest.main()
