"""Сборка промпта для модели: system (инструкция) + user (схема, evidence, вопрос)."""

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a SQLite database schema and a question, "
    "write a SQL query that answers the question. First think step by step inside "
    "<think></think>, then write only the final SQL query inside <answer></answer>."
)


def build_messages(example: dict) -> list[dict]:
    """Возвращает сообщения в формате chat для tokenizer.apply_chat_template."""
    user = f"Schema:\n{example['schema']}\n\n"
    if example.get("evidence"):
        user += f"Evidence: {example['evidence']}\n\n"
    user += f"Question: {example['question']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_prompt(tokenizer, example: dict, enable_thinking: bool | None = None) -> tuple[str, str]:
    """Возвращает (prompt_text, completion_prefix).

    Шаблоны Qwen3/Qwen3.5 сами дописывают в конец промпта "<think>\\n" (рассуждение включено)
    или пустой "<think>\\n\\n</think>\\n\\n" (выключено). В первом случае ответ модели
    начинается без открывающего тега, поэтому для оценки нужно приклеить completion_prefix.
    enable_thinking=None — не передавать флаг, поведение шаблона по умолчанию.
    """
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    prompt = tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True, **kwargs
    )
    prefix = "<think>\n" if prompt.rstrip().endswith("<think>") else ""
    return prompt, prefix
