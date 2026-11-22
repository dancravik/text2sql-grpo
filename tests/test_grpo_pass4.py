import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_grpo_pass4 as p


class SampledFollowUpTests(unittest.TestCase):
    def test_guard_pairing_archive_and_comet_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study, sft = root / "study", root / "sft"
            name = "uniform_dapo_2.5e-06"
            train = study / name / "train-unique"
            for source in (sft, train):
                (source / "adapter").mkdir(parents=True)
                (source / "adapter/adapter_model.safetensors").write_bytes(source.name.encode())
            (study / name / "evaluation").mkdir()
            p.write(study / name / "evaluation/source.json", {"adapter_sha256": p.sha(train / "adapter/adapter_model.safetensors")})
            main_zip = root / "study_complete.zip"
            main_zip.write_bytes(b"primary")
            results = {f"{c}_{loss}_{lr:g}": {"correct": 700, "training_dir": train.name}
                       for loss, lr in p.WAVES for c in ("uniform", "enriched")}
            manifest = root / "manifest.json"
            p.write(manifest, {"splits": {p.SPLIT: {"ids": ["q1", "q2"]}}})
            digest = p.sha(manifest)
            p.write(study / "plan.json", {"manifest_sha256": digest})
            lock = study / "study.lock"
            lock.touch()

            def generate(source, dest, gpu, deadline):
                counts = [2, 0] if source == sft else [1, 4]
                data = [{"id": q, "completions": [{"outcome": "correct" if i < k else "wrong"}
                         for i in range(4)]} for q, k in zip(("q1", "q2"), counts)]
                (dest / "validation_fast.jsonl").write_text("".join(json.dumps(row) + "\n" for row in data))
                p.write(dest / "metrics.json", {"manifest_sha256": digest,
                        "args": dict(n=4, temperature=0.8, seed=42, max_tokens=1024, max_model_len=4096,
                                     enable_thinking=False, smoke=False, low_memory=True),
                        "datasets": {p.SPLIT: {"num_questions": 2, "sampled_pass@1": sum(counts)/8,
                                               "pass@4": sum(k > 0 for k in counts)/2}}})

            with (patch.object(p, "MANIFEST", manifest), patch.object(p, "SFT_SHA", p.sha(sft / "adapter/adapter_model.safetensors")),
                  patch.object(p, "evaluate", side_effect=generate) as generate_mock,
                  patch.object(p, "upload", side_effect=[None, TimeoutError("Comet"), None]) as upload_mock,
                  patch.object(p.shutil, "disk_usage", return_value=type("Disk", (), {"free": 8 * 1024**3})())):
                p.write(study / "selection.json", {"status": "partial", "results": results})
                p.run(study, sft, "0", 9)
                generate_mock.assert_not_called()
                p.write(study / "selection.json", {"status": "complete", "results": results})
                os.utime(lock, (time.time()-8.5*3600,)*2)
                p.run(study, sft, "0", 9)
                generate_mock.assert_not_called()
                os.utime(lock, None)
                p.run(study, sft, "0", 9)
                path = root / "study_pass4/pass4_summary.json"
                self.assertEqual(p.read(path)["status"], "partial")
                self.assertEqual(p.read(path)["comparison"]["pass4_delta_pp"], 50)
                p.run(study, sft, "0", 9)
                self.assertEqual(upload_mock.call_count, 3)
                self.assertEqual(generate_mock.call_count, 2)
                self.assertEqual(p.read(path)["status"], "complete")
                self.assertEqual(main_zip.read_bytes(), b"primary")
                with ZipFile(root / "study_pass4.zip") as z:
                    self.assertIsNone(z.testzip())
                    self.assertIn("sft/validation_fast.jsonl", z.namelist())
