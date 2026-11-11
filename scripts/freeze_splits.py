"""Create a versioned split manifest ONCE. Existing files are never overwritten.

Run: python scripts/freeze_splits.py
Later: python scripts/freeze_splits.py --verify
No GPU or model inference required. Gold execution checks use the existing reward limits.
"""

import argparse
import json
import platform
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import DATA_DIR, load_bird_minidev, load_spider  # noqa: E402
from reward import execute  # noqa: E402
from splits import (  # noqa: E402
    DEFAULT_MANIFEST, SOURCE_FILES, SPLIT_NAMES, database_hash, fast_validation,
    load_split, read_manifest, sha256_file, validation_databases,
)


def audit(examples: list[dict], source: str, drop_empty: bool, blocked_dbs: set[str]) -> dict:
    kept, excluded, empty_ids = [], [], []
    for ex in examples:
        if ex["db_id"] in blocked_dbs:
            error = "database_overlap_with_spider"
        else:
            rows, error = execute(ex["db_path"], ex["gold_sql"], timeout=5.0, max_rows=10000)
            if error is None and not rows:
                if drop_empty:
                    error = "empty"
                else:
                    empty_ids.append(ex["id"])
        if error is None:
            kept.append(ex)
        else:
            excluded.append({"id": ex["id"], "db_id": ex["db_id"], "reason": error})
    return {
        "source": source,
        "ids": [ex["id"] for ex in kept],
        "db_ids": sorted({ex["db_id"] for ex in kept}),
        "raw_count": len(examples),
        "empty_gold_ids": empty_ids,
        "excluded": excluded,
    }


def print_summary(name: str, part: dict) -> None:
    reasons = dict(Counter(ex["reason"] for ex in part["excluded"]))
    print(f"{name}: {len(part['ids'])} questions, {len(part['db_ids'])} databases; excluded {reasons}", flush=True)


def build(data_dir: Path) -> dict:
    root = data_dir.resolve()
    train = load_spider(root, "train")
    dev = load_spider(root, "dev")
    bird = load_bird_minidev(root)
    val_dbs = validation_databases(train)
    spider_dbs = {ex["db_id"] for ex in train + dev}
    bird_overlap = spider_dbs & {ex["db_id"] for ex in bird}
    pools = {
        "train": ([ex for ex in train if ex["db_id"] not in val_dbs], "spider_train", True, set()),
        "validation": ([ex for ex in train if ex["db_id"] in val_dbs], "spider_train", False, set()),
        "test_spider": (dev, "spider_dev", False, set()),
        "test_bird": (bird, "bird_minidev", False, bird_overlap),
    }
    parts = {}
    for name, (examples, source, drop_empty, blocked) in pools.items():
        parts[name] = audit(examples, source, drop_empty, blocked)
        print_summary(name, parts[name])
    val_ids = set(parts["validation"]["ids"])
    val_examples = [ex for ex in train if ex["id"] in val_ids]
    quick_ids = fast_validation(val_examples)
    quick_set = set(quick_ids)
    parts["validation_fast"] = {
        "source": "spider_train", "parent": "validation", "ids": quick_ids,
        "db_ids": sorted({ex["db_id"] for ex in val_examples if ex["id"] in quick_set}),
        "raw_count": len(quick_ids), "excluded": [],
        "empty_gold_ids": [i for i in parts["validation"]["empty_gold_ids"] if i in quick_set],
    }
    print_summary("validation_fast", parts["validation_fast"])
    fingerprints = {rel: sha256_file(root / rel) for rel in SOURCE_FILES.values()}
    for path in sorted({ex["db_path"] for ex in train + dev + bird}):
        fingerprints[Path(path).relative_to(root).as_posix()] = database_hash(Path(path))
    with open(root / SOURCE_FILES["bird_minidev"], encoding="utf-8") as f:
        bird_raw = json.load(f)
    seen, duplicates = set(), []
    for i, ex in enumerate(bird_raw):
        key = (ex["db_id"], ex["question"])
        if key in seen:
            duplicates.append(f"bird_minidev_{i}")
        seen.add(key)
    return {
        "version": 1, "seed": 42,
        "policy": {
            "validation_databases_from_spider_train": 20,
            "validation_fast_size": 200,
            "gold_timeout_s": 5.0, "gold_max_rows": 10000,
            "drop_empty_gold": "train only",
            "model_dependent_selection": False,
            "bird_excluded_databases": sorted(bird_overlap),
            "bird_duplicate_ids": duplicates,
            "test_spider": "Spider dev reserved as project final test; NOT official Spider test",
            "prior_exposure": "Model selection probes used Spider train, including potential validation examples; final tests were not used in the recorded model selection.",
        },
        "audit_environment": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version},
        "files_sha256": fingerprints,
        "splits": parts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = read_manifest(args.manifest)
        # Verify all frozen source files, including excluded databases.
        for relative, expected in manifest["files_sha256"].items():
            path = args.data_dir / relative
            actual = database_hash(path) if path.suffix == ".sqlite" else sha256_file(path)
            if actual != expected:
                raise ValueError(f"Dataset snapshot changed: {relative}")
        for name in SPLIT_NAMES:
            load_split(name, args.data_dir, args.manifest)
            print_summary(name, manifest["splits"][name])
        print(f"Verified manifest SHA256: {sha256_file(args.manifest)}")
        return
    if args.manifest.exists():
        parser.error(f"Manifest already exists: {args.manifest}; use --verify, not regeneration")
    manifest = build(args.data_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    read_manifest(args.manifest)
    print(f"Frozen: {args.manifest}\nSHA256: {sha256_file(args.manifest)}")


if __name__ == "__main__":
    main()
