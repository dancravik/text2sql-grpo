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
