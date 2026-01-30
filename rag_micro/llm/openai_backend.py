from __future__ import annotations
from typing import List, Dict, Optional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

SYS = "Ты — инженер по микроконтроллерам. Отвечай строго по источникам. Если точного ответа нет, скажи об этом."

def answer_openai(
    chunks: List[Dict],
    question: str,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 300,
) -> str:
    if OpenAI is None:
        raise RuntimeError("Пакет openai не установлен.")
    client = OpenAI(api_key=api_key, base_url=base_url)

    sources = "\n\n".join([f"- {c['doc']}, стр.{c['page']}: {c['text']}" for c in chunks[:5]])
    user = f"Источники:\n{sources}\n\nВопрос: {question}\n\nКраткий ответ со ссылками на (Документ, стр.):"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or SYS},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
