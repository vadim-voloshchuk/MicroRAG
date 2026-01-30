# Воспроизведение экспериментов RAG Micro v2.0

## Быстрый старт

```bash
# 1. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate

# 2. Установить зависимости (CPU-версия PyTorch для экономии места)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Настроить API ключ
cp .env.example .env
# Отредактировать .env и вставить свой OPENAI_API_KEY
```

## Конфигурация

Основной конфиг: `config.yaml`

```bash
# Проверить конфигурацию
cat config.yaml
```

## Подготовка данных

```bash
# Индексация документов (PDF → чанки → индексы)
rag-cli ingest --from-pdf data/raw --out data/index

# Или с новыми параметрами чанкинга:
python -c "
from rag_micro.ingest.build_corpus_v2 import ingest_v2
ingest_v2(
    'data/raw', None, 'data/index',
    chunk_strategy='sentence_aware',
    chunk_size=512,
    chunk_overlap=128,
    include_tables=True
)
"
```

## Запуск экспериментов

### Пункт 15: Сравнение baseline-ов

```bash
python -m rag_micro.run_experiments baselines \
    --qa-csv data/benchmark/qa.csv \
    --index-root data/index \
    --output-dir results/baseline_comparison
```

Результаты: `results/baseline_comparison/baseline_comparison.csv`

### Пункт 5: Сравнение эмбеддеров

```bash
python -m rag_micro.run_experiments embedders \
    --qa-csv data/benchmark/qa.csv \
    --output-dir results/embedder_comparison
```

### Пункт 14: Абляция чанкинга

```bash
python -m rag_micro.run_experiments chunking \
    --qa-csv data/benchmark/qa.csv \
    --output-dir results/chunking_ablation
```

### Пункт 16: Тест табличного вклада

```bash
python -m rag_micro.run_experiments tables \
    --qa-csv data/benchmark/qa.csv \
    --output-dir results/table_ablation
```

### Пункт 9: End-to-end QA оценка

```bash
python -m rag_micro.run_experiments qa-eval \
    --qa-csv data/benchmark/qa.csv \
    --mode hybrid \
    --use-confidence
```

### Все эксперименты сразу

```bash
python -m rag_micro.run_experiments all \
    --qa-csv data/benchmark/qa.csv \
    --output-dir results
```

## Структура результатов

```
results/
├── baseline_comparison/
│   ├── baseline_comparison.csv      # Сводная таблица
│   ├── baseline_comparison.json     # Детальные результаты
│   └── *.tex                        # LaTeX таблицы
├── embedder_comparison/
├── chunking_ablation/
├── table_ablation/
└── qa/
    ├── qa_hybrid_results.jsonl      # Все ответы для воспроизводимости
    ├── qa_hybrid_summary.csv        # Метрики
    └── qa_hybrid_useful_cases.json  # Кейсы с низким F1, но полезным retrieval
```

## API сервер

```bash
# Запуск API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Endpoints:
# GET  /api/stats    - статистика
# GET  /api/search   - поиск
# POST /api/ask      - вопрос-ответ
# POST /api/ingest   - загрузка документов
```

## Версии зависимостей

См. `requirements.txt`. Ключевые:
- Python >= 3.10
- PyTorch >= 2.0 (CPU или CUDA)
- sentence-transformers >= 3.0
- openai >= 1.35

## Seed и воспроизводимость

Все эксперименты используют фиксированный seed=42 из `config.yaml`.

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

## Метрики

### Retrieval метрики (пункт 8)
- Recall@k (k=1,5,10)
- MRR@k
- nDCG@k
- Hit@k (evidence-in-top-k)
- Document-level recall
- Page-level recall

### QA метрики (пункт 9)
- Exact Match (EM)
- F1 (word-level)
- Support Ratio
- Hallucination Rate

### Доверительные интервалы (пункт 10)
Все метрики с 95% CI через bootstrap (1000 samples).

## Контакты

При возникновении проблем создайте issue в репозитории.
