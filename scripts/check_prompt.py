"""Проверка render_prompt: completion_prefix для шаблонов с <think> и без.

Запуск (из корня репозитория, нужен transformers и доступ к HF Hub):
    python scripts/check_prompt.py
"""

import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompt import build_messages, render_prompt  # noqa: E402

EXAMPLE = {"schema": "CREATE TABLE singer (id INT)", "question": "How many singers are there?"}


def check(name: str, ok: bool) -> bool:
    print(f"{'OK  ' if ok else 'FAIL'} {name}")
    return ok


def main() -> None:
    results = []

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    prompt, prefix = render_prompt(tok, EXAMPLE, enable_thinking=True)
    print(f"Qwen3.5 on:  ...{prompt[-40:]!r}, prefix={prefix!r}")
    results.append(check("Qwen3.5 on: prompt ends with <think>\\n", prompt.endswith("<think>\n")))
    results.append(check("Qwen3.5 on: prefix == <think>\\n", prefix == "<think>\n"))

    prompt, prefix = render_prompt(tok, EXAMPLE, enable_thinking=False)
    print(f"Qwen3.5 off: ...{prompt[-40:]!r}, prefix={prefix!r}")
    results.append(check("Qwen3.5 off: prefix empty", prefix == ""))

    prompt, prefix = render_prompt(tok, EXAMPLE)
    print(f"Qwen3.5 default: ...{prompt[-40:]!r}, prefix={prefix!r}")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    prompt, prefix = render_prompt(tok, EXAMPLE)
    # Прежний способ сборки промпта в probe_models.py
    old = tok.apply_chat_template(build_messages(EXAMPLE), tokenize=False, add_generation_prompt=True)
    print(f"Qwen2.5-Coder: ...{prompt[-40:]!r}, prefix={prefix!r}")
    results.append(check("Qwen2.5-Coder: prompt unchanged", prompt == old))
    results.append(check("Qwen2.5-Coder: prefix empty", prefix == ""))

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
