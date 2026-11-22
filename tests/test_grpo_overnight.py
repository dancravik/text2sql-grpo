import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_grpo_overnight as night


class OvernightStudyTest(unittest.TestCase):
    def test_evaluation_reuses_merge_after_vllm_oom(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)

            def execute(script, options, log, env, **kwargs):
                calls.append((script, options))
                if script == "evaluate.py" and "--low-memory" not in options:
                    log.write_text("CUDA out of memory")
                    raise RuntimeError("exit 1")
            with patch.object(night, "execute", side_effect=execute):
                night.evaluate(folder, folder / "dest", "validation", "1")
        self.assertEqual([x[0] for x in calls], ["export_sft.py", "evaluate.py", "evaluate.py"])
        self.assertEqual(calls[1][1] + ["--low-memory"], calls[2][1])

    def test_smoke_keeps_microbatch_that_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            ids = folder / "ids.json"
            night.write(ids, list("abcd"))
            def execute(script, options, log, env, **kwargs):
                if options[options.index("--micro-batch-size") + 1] == 2:
                    log.write_text("CUDA OOM")
                    raise RuntimeError("exit 1")
                out = options[options.index("--out-dir") + 1]
                out.mkdir()
                night.write(out / "Qwen3.5-2B_none.json", {"status": "ok", "loss_type": "grpo",
                                                          "train_example_ids": list("abcd")})
            with patch.object(night, "execute", side_effect=execute):
                self.assertEqual(night.smoke(folder, folder, ids, "grpo", "1"), 1)

    def test_cohorts_never_duplicate_or_import_validation_questions(self):
        ids = list(map(str, range(40)))
        wrong = set(ids[:5])
        result = night.cohorts(ids, 12, wrong)
        self.assertEqual(len(set(result["enriched"])), 12)
        self.assertEqual(wrong & set(result["enriched"]), wrong)
        self.assertEqual(result["uniform"], night.cohorts(ids, 12)["uniform"])
        with self.assertRaises(ValueError):
            night.cohorts(ids, 12, wrong | {"foreign"})

    def test_six_trials_oom_fallback_partial_evaluation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sft = root / "sft"
            study = root / "study"
            study.mkdir()
            train_ids = [f"t{i}" for i in range(40)]
            valid = [f"v{i}" for i in range(956)]
            uniform = night.cohorts(train_ids, 8)["uniform"]
            wrong = set(k for k in train_ids if k not in uniform)
            manifest = root / "manifest.json"
            night.write(manifest, {"splits": {"train": {"ids": train_ids}, "validation": {"ids": valid}}})
            (sft / "adapter").mkdir(parents=True)
            weights = sft / "adapter/adapter_model.safetensors"
            weights.write_bytes(b"sft")
            sft_sha = night.sha(weights)
            digest = night.sha(manifest)
            night.write(sft / "run.json", {"manifest_sha256": digest, "status": "ok", "base_revision": night.REVISION})

            def generate(source, dest, split, gpu):
                dest.mkdir(exist_ok=True)
                ids = train_ids if split == "train" else valid
                rows = [{"id": k, "db_id": "db", "question": k, "gold_sql": "SELECT 1", "prompt_tokens": 100,
                         "completions": [{"outcome": "wrong" if k in wrong else "correct"}]} for k in ids]
                (dest / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
                night.write(dest / "metrics.json", {"manifest_sha256": digest, "args": night.EVAL,
                            "datasets": {split: {"num_questions": len(ids),
                                                 "greedy_execution_accuracy": sum(map(night.correct, rows))/len(ids)}}})
            generate(sft, sft / "evaluation", "validation", "0")
            failed_eval = []
            attempts = []

            def eval_with_failure(source, dest, split, gpu):
                if dest.parent.name.startswith("uniform_grpo_5e-06") and not failed_eval:
                    failed_eval.append(dest)
                    raise RuntimeError("interrupted validation")
                generate(source, dest, split, gpu)

            def fake_training(script, args, logfile, env, **kwargs):
                value = lambda flag: args[args.index(flag) + 1]
                path = value("--out-dir")
                micro = value("--micro-batch-size")
                attempts.append((path.parent.name, micro))
                if path.parent.name.startswith("uniform_dapo") and micro == 2:
                    logfile.write_text("CUDA out of memory")
                    raise RuntimeError("exit 1")
                (path / "adapter").mkdir(parents=True)
                (path / "adapter/adapter_model.safetensors").write_bytes(path.parent.name.encode())
                ids = night.read(value("--train-ids"))
                stream = ids.copy()
                random.Random(night.SEED).shuffle(stream)
                night.write(path / "run.json", dict(
                    status="ok", loss_type=value("--loss-type"), global_step=value("--max-steps"),
                    args=dict(lr=value("--lr"), micro_batch_size=micro), train_example_ids=ids,
                    generated_question_ids=stream, question_presentations=len(ids), optimizer_steps_applied=4,
                    initial_sft=dict(adapter_sha256=sft_sha), gpu_hours=1, peak_memory_gb=12, reward_groups={},
                    generated_completion_tokens=350, comet_experiment_key="example"))
            with (patch.object(night, "MANIFEST", manifest), patch.object(night, "SFT_SHA", sft_sha),
                  patch.object(night, "evaluate", side_effect=eval_with_failure) as evaluator,
                  patch.object(night, "execute", side_effect=fake_training), patch.object(night, "smoke", return_value=2),
                  patch.object(night, "make_archive"), patch.object(night, "log_evaluation"),
                  patch.object(night, "usable_host_gib", return_value=19)):
                night.run(study, sft, ["0", "1"], 4)
                summary = night.read(study / "selection.json")
                self.assertEqual(summary["status"], "partial")
                self.assertEqual(len(summary["results"]), 5)
                self.assertEqual(summary["results"]["uniform_dapo_2.5e-06"]["micro_batch"], 1)
                self.assertEqual(len(attempts), 7)
                evaluator.reset_mock()
                night.run(study, sft, ["0", "1"], 4)
                summary = night.read(study / "selection.json")
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(len(summary["results"]), 6)
                self.assertEqual(len(summary["contrasts"]), 7)
                self.assertEqual(len(attempts), 7)
                self.assertEqual(evaluator.call_count, 1)


if __name__ == "__main__":
    unittest.main()
