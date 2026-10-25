"""Генерация ответов через vLLM и их оценка наградой.

vLLM импортируется внутри функций, чтобы скрипты запускались (--help и т.п.) без него.
"""

import time
from typing import TYPE_CHECKING

from prompt import render_prompt
from reward import check_format, compare, execute, extract_sql, reward

if TYPE_CHECKING:
    from vllm import LLM


def load_llm(model_id: str, max_model_len: int, seed: int) -> "LLM":
    """vLLM-модель в fp16 (T4 не поддерживает bf16)."""
    from vllm import LLM

    return LLM(
        model=model_id,
        dtype="half",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        seed=seed,
    )


def classify(text: str, gold_rows: list[tuple], ex: dict) -> str:
    """Тип исхода: no_sql | exec_error | wrong | correct."""
    sql = extract_sql(text)
    if sql is None:
        return "no_sql"
    rows, err = execute(ex["db_path"], sql)
    if err is not None:
        return "exec_error"
    return "correct" if compare(rows, gold_rows, ex["gold_sql"]) else "wrong"


def sample_and_score(
    llm: "LLM",
    examples: list[dict],
    n: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    enable_thinking: bool,
) -> tuple[list[dict], float]:
    """Генерирует n ответов на каждый пример и оценивает их.

    Возвращает записи {"id", "k_correct", "prompt_tokens", "completions"} и время генерации
    (только llm.generate, без оценки).
    """
    from vllm import SamplingParams

    gold_rows = []
    for ex in examples:
        rows, err = execute(ex["db_path"], ex["gold_sql"])
        if err is not None:
            raise RuntimeError(f"Gold SQL failed ({err}) after filtering: {ex['gold_sql']}")
        gold_rows.append(rows)

    tok = llm.get_tokenizer()
    rendered = [render_prompt(tok, ex, enable_thinking) for ex in examples]
    prompts = [prompt for prompt, _ in rendered]
    prefixes = [prefix for _, prefix in rendered]
    prompt_lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, seed=seed)

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t0

    records = []
    for ex, out, gold, plen, prefix in zip(examples, outputs, gold_rows, prompt_lens, prefixes, strict=True):
        completions = []
        for c in out.outputs:
            # Оцениваем ответ вместе с тегом, который шаблон дописал в конец промпта
            full = prefix + c.text
            completions.append({
                "text": c.text,
                "prefix": prefix,
                "reward": reward(full, ex["gold_sql"], ex["db_path"]),
                "outcome": classify(full, gold, ex),
                "format_ok": check_format(full),
                "finish_reason": c.finish_reason,
                "num_tokens": len(c.token_ids),
            })
        records.append({
            "id": ex["id"],
            "k_correct": sum(c["outcome"] == "correct" for c in completions),
            "prompt_tokens": plen,
            "completions": completions,
        })
    return records, gen_time
