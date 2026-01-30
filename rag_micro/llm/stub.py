from __future__ import annotations
from typing import List, Dict

TEMPLATE = """Ответь кратко, опираясь только на источники ниже. 
Если ответа нет в источниках, напиши 'Не нашёл в документации.'

Источники:
{sources}

Вопрос: {question}

Ответ:"""

def answer_stub(chunks: List[Dict], question: str) -> str:
    # Простая заглушка: показать промпт и топ-3 источника, чтобы можно было проверить retrieval
    top = chunks[:3]
    src = "\n\n".join([f"({i+1}) {c['doc']}, стр.{c['page']}\n{c['text']}" for i, c in enumerate(top)])
    prompt = TEMPLATE.format(sources=src, question=question)
    return f"[STUB — включите llama_cpp или openai для реальных ответов]\n\n" + prompt
