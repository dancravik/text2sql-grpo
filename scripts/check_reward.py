"""Sanity checks for src/reward.py: hand-written cases and gold-vs-gold on all datasets."""

import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import DATA_DIR, load_bird_minidev, load_spider  # noqa: E402
from reward import execute, extract_sql, reward  # noqa: E402

DB = str(DATA_DIR / "spider" / "database" / "concert_singer" / "concert_singer.sqlite")
GOLD = "SELECT count(*) FROM singer"


def wrap(sql: str) -> str:
    """Model answer in the correct format."""
    return f"<answer>{sql}</answer>"


def check(name: str, ok: bool) -> bool:
    print(f"{'OK  ' if ok else 'FAIL'} {name}")
    return ok


def table_exists(db_path: str, table: str) -> bool:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def manual_cases() -> bool:
    print("=== A. Manual cases ===")
    results = [
        check("no tags -> 0.0", reward(GOLD, GOLD, DB) == 0.0),
        check("format + typo SQL -> 0.1", reward(wrap("SELEC count(*) FROM singer"), GOLD, DB) == 0.1),
        check("format + correct SQL -> 1.1", reward(wrap(GOLD), GOLD, DB) == 1.1),
        check("text outside <answer> + correct SQL -> 1.0",
              reward(f"Here is the query: {wrap(GOLD)}", GOLD, DB) == 1.0),
        check(
            "```sql fence inside <answer> is stripped",
            extract_sql(f"<answer>\n```sql\n{GOLD}\n```\n</answer>") == GOLD
            and reward(wrap(f"```sql\n{GOLD}\n```"), GOLD, DB) == 1.1,
        ),
    ]

    _, err = execute(DB, "DROP TABLE singer")
    results.append(check(f"DROP TABLE -> 'error' (got {err!r}), table still exists",
                         err == "error" and table_exists(DB, "singer")))

    timeout = 1.0
    infinite = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    start = time.monotonic()
    _, err = execute(DB, infinite, timeout=timeout)
    elapsed = time.monotonic() - start
    results.append(check(f"infinite query -> 'timeout' (got {err!r}) in {elapsed:.2f}s (timeout {timeout}s)",
                         err == "timeout" and timeout <= elapsed < timeout + 0.5))
    return all(results)


def gold_vs_gold(name: str, examples: list[dict]) -> None:
    n_full, n_empty, errors, n_timeout, n_too_many = 0, 0, [], 0, 0
    exec_time = 0.0
    for ex in examples:
        start = time.monotonic()
        rows, err = execute(ex["db_path"], ex["gold_sql"])
        exec_time += time.monotonic() - start

        if err == "timeout":
            n_timeout += 1
        elif err == "too_many_rows":
            n_too_many += 1
        elif err == "error":
            try:  # re-run without the wrapper to get the sqlite message
                conn = sqlite3.connect(f"file:{ex['db_path']}?mode=ro", uri=True)
                conn.execute(ex["gold_sql"]).fetchall()
                msg = "?"
            except sqlite3.Error as e:
                msg = str(e)
            finally:
                conn.close()
            errors.append((ex, msg))
        elif not rows:
            n_empty += 1

        if err is None:
            try:
                n_full += reward(wrap(ex["gold_sql"]), ex["gold_sql"], ex["db_path"]) == 1.1
            except RuntimeError:
                pass  # gold flaked on the second run (e.g. timeout near the limit)

    print(f"\n--- {name}: {len(examples)} examples ---")
    print(f"reward 1.1:      {n_full}")
    print(f"gold errors:     {len(errors)}")
    for ex, msg in errors[:5]:
        print(f"    [{Path(ex['db_path']).stem}] {msg}\n        {ex['gold_sql']}")
    print(f"gold timeouts:   {n_timeout}")
    print(f"too many rows:   {n_too_many}")
    print(f"empty result:    {n_empty}")
    print(f"exec time:       {exec_time:.1f}s total, {exec_time / len(examples) * 1000:.1f} ms/query")


if __name__ == "__main__":
    all_ok = manual_cases()
    print("\n=== B. Gold vs gold ===")
    gold_vs_gold("spider train", load_spider(DATA_DIR, "train"))
    gold_vs_gold("spider dev", load_spider(DATA_DIR, "dev"))
    gold_vs_gold("bird minidev", load_bird_minidev(DATA_DIR))
    print(f"\nManual cases: {'ALL OK' if all_ok else 'SOME FAILED'}")
