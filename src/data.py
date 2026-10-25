"""Load Spider and BIRD mini-dev into a common example format."""

import json
import os
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

from reward import execute

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

SPIDER_FILES = {"train": "train_spider.json", "dev": "dev.json"}


@lru_cache(maxsize=None)
def get_schema(db_path: str) -> str:
    """Return CREATE TABLE statements of all user tables, separated by ";\\n"."""
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        conn.close()
    return ";\n".join(row[0] for row in rows) + ";"


def _make_example(
    id: str,
    db_id: str,
    question: str,
    gold_sql: str,
    db_path: Path,
    evidence: str,
    difficulty: str | None,
    source: str,
) -> dict:
    """Build one example in the common format."""
    return {
        "id": id,
        "db_id": db_id,
        "question": question,
        "gold_sql": gold_sql,
        "db_path": str(db_path),
        "schema": get_schema(str(db_path)),
        "evidence": evidence,
        "difficulty": difficulty,
        "source": source,
    }


def load_spider(data_dir: Path = DATA_DIR, split: str = "train") -> list[dict]:
    """Load Spider train or dev split; id is "spider_{split}_{index in the source json}"."""
    if split not in SPIDER_FILES:
        raise ValueError(f"Unknown Spider split: {split!r}, expected one of {list(SPIDER_FILES)}")
    root = Path(data_dir) / "spider"
    with open(root / SPIDER_FILES[split], encoding="utf-8") as f:
        raw = json.load(f)
    return [
        _make_example(
            id=f"spider_{split}_{i}",
            db_id=item["db_id"],
            question=item["question"],
            gold_sql=item["query"],
            db_path=root / "database" / item["db_id"] / f"{item['db_id']}.sqlite",
            evidence="",
            difficulty=None,
            source="spider",
        )
        for i, item in enumerate(raw)
    ]


def load_bird_minidev(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load BIRD mini-dev (SQLite version), dropping duplicate (db_id, question) pairs.

    id is "bird_minidev_{index in the source json}", so it stays stable after deduplication.
    """
    root = Path(data_dir) / "minidev" / "MINIDEV"
    with open(root / "mini_dev_sqlite.json", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    seen = set()
    for i, item in enumerate(raw):
        key = (item["db_id"], item["question"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            _make_example(
                id=f"bird_minidev_{i}",
                db_id=item["db_id"],
                question=item["question"],
                gold_sql=item["SQL"],
                db_path=root / "dev_databases" / item["db_id"] / f"{item['db_id']}.sqlite",
                evidence=item["evidence"],
                difficulty=item["difficulty"],
                source="bird_minidev",
            )
        )
    print(f"bird_minidev: removed {len(raw) - len(examples)} duplicates")
    return examples


def filter_train(examples: list[dict]) -> list[dict]:
    """Drop examples whose gold SQL fails, times out, returns too many rows or an empty result.

    Empty gold results are dropped because any query returning nothing would get full reward.
    """
    kept = []
    removed = Counter()
    for ex in examples:
        rows, err = execute(ex["db_path"], ex["gold_sql"])
        if err is None and not rows:
            err = "empty"
        if err is None:
            kept.append(ex)
        else:
            removed[err] += 1
    print(f"filter_train: kept {len(kept)} of {len(examples)}, removed {dict(removed)}")
    return kept


if __name__ == "__main__":
    datasets = {
        "spider train": load_spider(DATA_DIR, "train"),
        "spider dev": load_spider(DATA_DIR, "dev"),
        "bird minidev": load_bird_minidev(DATA_DIR),
    }

    for name, examples in datasets.items():
        n_dbs = len({ex["db_path"] for ex in examples})
        print(f"{name}: {len(examples)} examples, {n_dbs} unique databases")

    filter_train(datasets["spider train"])

    overlap = {ex["db_path"] for ex in datasets["spider train"]} & {ex["db_path"] for ex in datasets["spider dev"]}
    print(f"spider train/dev shared databases: {len(overlap)}")
    assert not overlap, f"Spider dev databases overlap with train: {sorted(overlap)}"

    for name in ("spider train", "bird minidev"):
        print(f"\n=== Example from {name} ===")
        for key, value in datasets[name][0].items():
            print(f"{key}: {value}")
