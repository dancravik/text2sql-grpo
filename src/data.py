"""Load Spider and BIRD mini-dev into a common example format."""

import json
import os
import sqlite3
from functools import lru_cache
from pathlib import Path

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
    question: str,
    gold_sql: str,
    db_path: Path,
    evidence: str,
    difficulty: str | None,
    source: str,
) -> dict:
    """Build one example in the common format."""
    return {
        "question": question,
        "gold_sql": gold_sql,
        "db_path": str(db_path),
        "schema": get_schema(str(db_path)),
        "evidence": evidence,
        "difficulty": difficulty,
        "source": source,
    }


def load_spider(data_dir: Path = DATA_DIR, split: str = "train") -> list[dict]:
    """Load Spider train or dev split."""
    if split not in SPIDER_FILES:
        raise ValueError(f"Unknown Spider split: {split!r}, expected one of {list(SPIDER_FILES)}")
    root = Path(data_dir) / "spider"
    with open(root / SPIDER_FILES[split], encoding="utf-8") as f:
        raw = json.load(f)
    return [
        _make_example(
            question=item["question"],
            gold_sql=item["query"],
            db_path=root / "database" / item["db_id"] / f"{item['db_id']}.sqlite",
            evidence="",
            difficulty=None,
            source="spider",
        )
        for item in raw
    ]


def load_bird_minidev(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load BIRD mini-dev (SQLite version), dropping duplicate (db_id, question) pairs."""
    root = Path(data_dir) / "minidev" / "MINIDEV"
    with open(root / "mini_dev_sqlite.json", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    seen = set()
    for item in raw:
        key = (item["db_id"], item["question"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            _make_example(
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


if __name__ == "__main__":
    datasets = {
        "spider train": load_spider(DATA_DIR, "train"),
        "spider dev": load_spider(DATA_DIR, "dev"),
        "bird minidev": load_bird_minidev(DATA_DIR),
    }

    for name, examples in datasets.items():
        n_dbs = len({ex["db_path"] for ex in examples})
        print(f"{name}: {len(examples)} examples, {n_dbs} unique databases")

    overlap = {ex["db_path"] for ex in datasets["spider train"]} & {ex["db_path"] for ex in datasets["spider dev"]}
    print(f"spider train/dev shared databases: {len(overlap)}")
    assert not overlap, f"Spider dev databases overlap with train: {sorted(overlap)}"

    for name in ("spider train", "bird minidev"):
        print(f"\n=== Example from {name} ===")
        for key, value in datasets[name][0].items():
            print(f"{key}: {value}")
