from __future__ import annotations
from typing import List, Dict
import ollama

SYS = "Ты — инженер по микроконтроллерам. Отвечай строго по источникам. Если точного ответа нет, скажи об этом и укажи, где смотреть в документации."

def answer_ollama(chunks: List[Dict], question: str, model: str, host: str) -> str:
    client = ollama.Client(host=host)

    sources = "\n\n".join([f"- {c['doc']}, стр.{c['page']}: {c['text']}" for c in chunks[:5]])
    user = f"Источники:\n{sources}\n\nВопрос: {question}\n\nКраткий ответ со ссылками на (Документ, стр.):"

    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.1, "num_ctx": 4096},
        stream=False,
    )
    # Формат ответа у ollama.chat: {'message': {'role': 'assistant', 'content': '...'}, ...}
    return resp["message"]["content"].strip()
