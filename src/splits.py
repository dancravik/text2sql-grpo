"""Frozen experiment splits. Loading never re-filters gold SQL or model outputs."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from data import DATA_DIR, load_bird_minidev, load_spider

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "splits" / "v1.json"
SOURCE_FILES = {
    "spider_train": "spider/train_spider.json",
    "spider_dev": "spider/dev.json",
    "bird_minidev": "minidev/MINIDEV/mini_dev_sqlite.json",
}
SPLIT_NAMES = ("train", "validation", "validation_fast", "test_spider", "test_bird")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def database_hash(path: Path) -> str:
    # A nonempty WAL can contain data absent from the main file's fingerprint.
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"Database has an uncheckpointed WAL; use a consistent dataset snapshot: {path}")
    return sha256_file(path)


def validation_databases(examples: list[dict], count: int = 20, seed: int = 42) -> set[str]:
    databases = sorted({ex["db_id"] for ex in examples})
    if not 0 < count < len(databases):
        raise ValueError("Validation must reserve some, but not all databases")
    random.Random(seed).shuffle(databases)
    return set(databases[:count])


def fast_validation(examples: list[dict], size: int = 200, seed: int = 42) -> list[str]:
    """Round-robin across databases, shuffled within each, for a cheap diagnostic set."""
    groups = defaultdict(list)
    for ex in examples:
        groups[ex["db_id"]].append(ex["id"])
    rng = random.Random(seed)
    groups = {db: sorted(ids) for db, ids in sorted(groups.items())}
    for ids in groups.values():
        rng.shuffle(ids)
    selected = []
    while len(selected) < min(size, len(examples)):
        for ids in groups.values():
            if ids and len(selected) < size:
                selected.append(ids.pop())
    return selected


def read_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest["version"] != 1:
        raise ValueError("Unsupported split manifest version")
    splits = manifest["splits"]
    seen_ids, seen_dbs = set(), set()
    for name in SPLIT_NAMES:
        part = splits[name]
        ids, dbs = set(part["ids"]), set(part["db_ids"])
        if len(ids) != len(part["ids"]):
            raise ValueError(f"Duplicate IDs in {name}")
        if name == "validation_fast":
            if not ids <= set(splits["validation"]["ids"]) or not dbs <= set(splits["validation"]["db_ids"]):
                raise ValueError("validation_fast is not a subset of validation")
            continue
        if seen_ids & ids or seen_dbs & dbs:
            raise ValueError(f"Split leakage in {name}")
        seen_ids.update(ids)
        seen_dbs.update(dbs)
    return manifest


def load_split(
    name: str, data_dir: Path = DATA_DIR, manifest_path: Path = DEFAULT_MANIFEST
) -> list[dict]:
    """Resolve frozen IDs and verify source JSON + databases, portable across DATA_DIRs."""
    manifest = read_manifest(manifest_path)
    part = manifest["splits"][name]
    root = Path(data_dir).resolve()
    source = part["source"]
    source_path = SOURCE_FILES[source]
    if sha256_file(root / source_path) != manifest["files_sha256"][source_path]:
        raise ValueError(f"Source data changed: {source_path}; do not silently regenerate splits")
    if source == "bird_minidev":
        examples = load_bird_minidev(root)
    else:
        examples = load_spider(root, "train" if source == "spider_train" else "dev")
    by_id = {ex["id"]: ex for ex in examples}
    selected = [by_id[example_id] for example_id in part["ids"]]
    if {ex["db_id"] for ex in selected} != set(part["db_ids"]):
        raise ValueError(f"Database membership mismatch in {name}")
    for db_path in sorted({ex["db_path"] for ex in selected}):
        relative = Path(db_path).relative_to(root).as_posix()
        if database_hash(Path(db_path)) != manifest["files_sha256"][relative]:
            raise ValueError(f"Database changed: {relative}")
    return selected
