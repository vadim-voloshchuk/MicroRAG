from __future__ import annotations
from typing import List, Dict
try:
    from llama_cpp import Llama
except Exception:
    Llama = None  # type: ignore

PROMPT = """Ты — инженер по микроконтроллерам. Отвечай строго по источникам.
Если точного ответа нет, напиши: 'Не нашёл в документации.'
В конце укажи ссылки (Документ, стр.).

Источники:
{sources}

Вопрос: {question}

Краткий ответ:"""

_model = None

def _get_model(path: str, n_ctx: int = 4096, n_threads: int = 8):
    global _model
    if _model is None:
        if Llama is None:
            raise RuntimeError("llama-cpp-python не установлен.")
        _model = Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
    return _model

def answer_llama(chunks: List[Dict], question: str, model_path: str, n_ctx: int = 4096, n_threads: int = 8) -> str:
    src = "\n\n".join([f"- {c['doc']}, стр.{c['page']}: {c['text']}" for c in chunks[:5]])
    llm = _get_model(model_path, n_ctx=n_ctx, n_threads=n_threads)
    out = llm(
        prompt=PROMPT.format(sources=src, question=question),
        max_tokens=300,
        temperature=0.1,
        top_p=0.9,
        stop=["\n\n\n"]
    )
    return out["choices"][0]["text"].strip()
