"""Reward function for GRPO: format bonus + execution match against the gold SQL."""

import re
import sqlite3
import time
from collections import Counter

FORMAT_BONUS = 0.1

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```(?:sql)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
FORMAT_RE = re.compile(r"\s*<think>.*</think>\s*<answer>.*</answer>\s*", re.DOTALL)
ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
TAGS = ("<think>", "</think>", "<answer>", "</answer>")


def extract_sql(text: str) -> str | None:
    """Return SQL from the last <answer>...</answer> block, without a ``` fence."""
    answers = ANSWER_RE.findall(text)
    if not answers:
        return None
    sql = answers[-1].strip()
    fence = CODE_FENCE_RE.match(sql)
    if fence:
        sql = fence.group(1).strip()
    return sql or None


def check_format(text: str) -> bool:
    """True if text is <think>...</think><answer>...</answer>, each tag exactly once."""
    if any(text.count(tag) != 1 for tag in TAGS):
        return False
    return FORMAT_RE.fullmatch(text) is not None


def execute(
    db_path: str, sql: str, timeout: float = 5.0, max_rows: int = 10000
) -> tuple[list[tuple] | None, str | None]:
    """Run sql read-only; return (rows, None) or (None, "error" | "timeout" | "too_many_rows")."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    timed_out = False
    deadline = time.monotonic() + timeout

    def check_deadline() -> int:
        nonlocal timed_out
        timed_out = time.monotonic() > deadline
        return 1 if timed_out else 0

    conn.set_progress_handler(check_deadline, 1000)
    try:
        rows = conn.execute(sql).fetchmany(max_rows + 1)
    except sqlite3.Error:
        return None, "timeout" if timed_out else "error"
    finally:
        conn.close()
    if len(rows) > max_rows:
        return None, "too_many_rows"
    return rows, None


def compare(pred_rows: list[tuple], gold_rows: list[tuple], gold_sql: str) -> bool:
    """Ordered comparison if gold_sql has ORDER BY, otherwise compare as multisets."""
    if ORDER_BY_RE.search(gold_sql):
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


def reward(text: str, gold_sql: str, db_path: str) -> float:
    """Return execution correctness (0 or 1) plus format bonus (0 or 0.1)."""
    format_bonus = FORMAT_BONUS if check_format(text) else 0.0
    sql = extract_sql(text)
    if sql is None:
        return format_bonus

    gold_rows, gold_err = execute(db_path, gold_sql)
    if gold_err is not None:
        raise RuntimeError(f"Gold SQL failed ({gold_err}) on {db_path}: {gold_sql}")

    pred_rows, pred_err = execute(db_path, sql)
    correct = pred_err is None and compare(pred_rows, gold_rows, gold_sql)
    return (1.0 if correct else 0.0) + format_bonus
