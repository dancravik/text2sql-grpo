import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_sft import upload_training_bundle


class CheckpointBackupTests(unittest.TestCase):
    def test_checkpoint_contains_resume_state_before_confirmed_local_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-50"
            checkpoint.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
                (checkpoint / name).write_bytes(b"saved state")
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            archive = root / "checkpoint-50.zip"
            experiment = Mock()

            def confirm(timeout):
                with ZipFile(archive) as bundle:
                    self.assertIn("checkpoint-50/optimizer.pt", bundle.namelist())
                    self.assertIn("checkpoint-50/rng_state.pth", bundle.namelist())
                    self.assertIn("splits/v1.json", bundle.namelist())
                return True

            experiment.flush.side_effect = confirm
            self.assertTrue(upload_training_bundle(
                experiment, checkpoint, archive, step=50,
                extra_files=[(manifest, "splits/v1.json")],
            ))
            self.assertFalse(archive.exists())
            self.assertTrue((checkpoint / "optimizer.pt").exists())
            self.assertFalse(experiment.log_asset.call_args.kwargs["copy_to_tmp"])

    def test_unconfirmed_upload_preserves_local_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            archive = root / "final_adapter.zip"
            experiment = Mock()
            experiment.flush.return_value = False
            self.assertFalse(upload_training_bundle(experiment, adapter, archive, step=100))
            self.assertTrue(archive.is_file())


if __name__ == "__main__":
    unittest.main()
