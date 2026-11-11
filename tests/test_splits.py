"""Dataset boundaries and snapshot integrity, without GPU or downloaded datasets."""

import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_splits import audit  # noqa: E402
from splits import (  # noqa: E402
    database_hash, fast_validation, load_split, read_manifest, sha256_file, validation_databases,
)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.db_relative = "spider/database/alpha/alpha.sqlite"
        self.db = self.data / self.db_relative
        self.db.parent.mkdir(parents=True)
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE items (value INTEGER)")
            conn.execute("INSERT INTO items VALUES (1)")
        self.source = self.data / "spider/train_spider.json"
        self.source.write_text(json.dumps([{
            "db_id": "alpha", "question": "How many items?", "query": "SELECT count(*) FROM items",
        }]))
        parts = {}
        for name, source, example_id, db in [
            ("train", "spider_train", "spider_train_0", "alpha"),
            ("validation", "spider_train", "spider_train_1", "beta"),
            ("validation_fast", "spider_train", "spider_train_1", "beta"),
            ("test_spider", "spider_dev", "spider_dev_0", "gamma"),
            ("test_bird", "bird_minidev", "bird_minidev_0", "delta"),
        ]:
            parts[name] = {"source": source, "ids": [example_id], "db_ids": [db]}
        self.manifest = {"version": 1, "splits": parts, "files_sha256": {
            "spider/train_spider.json": sha256_file(self.source),
            self.db_relative: database_hash(self.db),
        }}
        self.path = self.root / "manifest.json"
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def test_database_split_is_order_independent(self):
        examples = [{"db_id": f"db_{i}", "id": f"{i}_{j}"} for i in range(30) for j in range(3)]
        selected = validation_databases(examples, count=5)
        self.assertEqual(selected, validation_databases(list(reversed(examples)), count=5))
        self.assertEqual(len(selected), 5)

    def test_fast_validation_has_all_databases_and_no_duplicates(self):
        examples = [{"db_id": f"db_{i}", "id": f"{i}_{j}"} for i in range(20) for j in range(15)]
        selected = fast_validation(examples)
        self.assertEqual(len(selected), 200)
        self.assertEqual(len(set(selected)), 200)
        self.assertEqual({i.split("_")[0] for i in selected}, {str(i) for i in range(20)})
        self.assertEqual(selected, fast_validation(list(reversed(examples))))

    def test_cross_split_database_leakage_is_rejected(self):
        self.manifest["splits"]["test_bird"]["db_ids"] = ["alpha"]
        self.save()
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_manifest(self.path)

    def test_cross_split_id_leakage_is_rejected(self):
        self.manifest["splits"]["validation"]["ids"] = ["spider_train_0"]
        self.save()
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_manifest(self.path)

    def test_fast_validation_cannot_include_test(self):
        self.manifest["splits"]["validation_fast"]["ids"] = ["spider_dev_0"]
        self.save()
        with self.assertRaisesRegex(ValueError, "subset"):
            read_manifest(self.path)

    def test_snapshot_loads_after_relocation_without_gold_refiltering(self):
        relocated = self.root / "kaggle_input"
        shutil.copytree(self.data, relocated)
        with patch("data.execute", side_effect=AssertionError("Unexpected gold re-filter")):
            examples = load_split("train", relocated, self.path)
        self.assertEqual(examples[0]["id"], "spider_train_0")
        self.assertTrue(examples[0]["db_path"].startswith(str(relocated)))

    def test_changed_source_is_rejected(self):
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Source data changed"):
            load_split("train", self.data, self.path)

    def test_changed_database_is_rejected(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO items VALUES (2)")
        with self.assertRaisesRegex(ValueError, "Database changed"):
            load_split("train", self.data, self.path)

    def test_empty_gold_is_excluded_only_from_train(self):
        examples = [{"id": "empty", "db_id": "alpha", "db_path": str(self.db),
                     "gold_sql": "SELECT * FROM items WHERE 0"}]
        train = audit(examples, "spider_train", True, set())
        val = audit(examples, "spider_train", False, set())
        self.assertEqual(train["excluded"][0]["reason"], "empty")
        self.assertEqual(val["ids"], ["empty"])
        self.assertEqual(val["empty_gold_ids"], ["empty"])

    def test_overlapping_bird_database_is_excluded_without_execution(self):
        examples = [{"id": "bird", "db_id": "formula_1"}]
        with patch("freeze_splits.execute", side_effect=AssertionError("Should not execute")):
            result = audit(examples, "bird_minidev", False, {"formula_1"})
        self.assertEqual(result["ids"], [])
        self.assertEqual(result["excluded"][0]["reason"], "database_overlap_with_spider")


if __name__ == "__main__":
    unittest.main()
