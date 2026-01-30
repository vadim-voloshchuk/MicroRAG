# RAG Micro (техническая документация микроконтроллеров)

Полный RAG‑пайплайн из отчёта: парсинг PDF → чанкинг → индексация (BM25 + FAISS) → генерация ответа LLM → оценка EM/F1/Recall@5/MRR.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .  # либо: pip install -r requirements.txt
```

> Примечание: для `sentence-transformers` потребуется загрузка модели `all-MiniLM-L6-v2` из HF. При первом запуске она скачается автоматически.
> Для локальной LLM можно использовать `llama-cpp-python` с gguf‑моделью или OpenAI‑совместимый эндпоинт.

## Структура данных

Ожидаются каталоги:
```
data/
  raw/           # PDF
  processed/
    text/        # txt (если уже есть)
  index/         # сюда пишутся индексы
  benchmark/
    qa.csv       # необязательный набор Q/A для оценки
```

## Быстрый старт

1) **Импорт и индексация**
```bash
rag-cli ingest --from-pdf data/raw --from-txt data/processed/text --out data/index
rag-cli index --build --index-root data/index --faiss --bm25
```

2) **Задайте вопрос**
```bash
rag-cli ask "What is the minimum VDD of STM32F103?" --mode hybrid --k 5 --index-root data/index
```

3) **Оценка на бенчмарке (если есть `data/benchmark/qa.csv`)**
```bash
rag-cli eval qa --index-root data/index --mode hybrid --limit 0
```

## Конфигурация LLM

По умолчанию используется «эхо‑модель» (заглушка). Включить реальные варианты:

- **LLama.cpp локально**: укажите путь к gguf и параметры в `.env`:
```
LLM_BACKEND=llama_cpp
LLAMA_MODEL_PATH=/path/to/model.gguf
LLAMA_CTX=4096
LLAMA_THREADS=8
```
- **OpenAI‑совместимый сервер** (в т.ч. локальный прокси):
```
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...            # если нужен
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_MODEL=gpt-4o-mini
```

## Команды

Смотрите `rag-cli --help`.
