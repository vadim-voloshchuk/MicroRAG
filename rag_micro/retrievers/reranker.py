"""
Reranker - Модуль переранжирования на базе Cross-Encoder.

Отвечает на требование пункта 6: "сильный baseline reranking".
Реализует режимы Dense+Reranker и Hybrid+Reranker.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankerInfo:
    """Информация о модели переранжирования."""
    name: str
    model_id: str
    description: str
    max_length: int = 512


class BaseReranker(ABC):
    """Базовый класс для всех переранжировщиков."""

    @property
    @abstractmethod
    def info(self) -> RerankerInfo:
        """Возвращает информацию о модели."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Переранжирует кандидатов по релевантности к запросу.

        Args:
            query: Поисковый запрос
            candidates: Список кандидатов (словарей с полем 'text')
            top_k: Вернуть top-k после переранжирования (None = все)

        Returns:
            List[Dict]: Переранжированные кандидаты с обновлёнными scores
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Reranker на базе Cross-Encoder из sentence-transformers.

    Cross-Encoder принимает пару (query, document) и выдаёт score релевантности.
    Это значительно точнее bi-encoder (dense retrieval), но медленнее.
    """

    # Популярные модели для reranking
    MODELS = {
        "ms-marco-MiniLM-L6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
    }

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._info = None

    @property
    def model(self):
        """Lazy loading модели."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder: {self.model_id}")
            self._model = CrossEncoder(self.model_id, device=self.device)
        return self._model

    @property
    def info(self) -> RerankerInfo:
        if self._info is None:
            self._info = RerankerInfo(
                name=self.model_id.split("/")[-1],
                model_id=self.model_id,
                description="Cross-Encoder reranker for passage ranking",
                max_length=512,
            )
        return self._info

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        if not candidates:
            return []

        # Формируем пары (query, document)
        pairs = [(query, c.get("text", "")) for c in candidates]

        # Получаем scores от cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Добавляем scores и сортируем
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
            candidates[i]["original_score"] = candidates[i].get("score", 0)
            candidates[i]["score"] = float(score)  # Заменяем score на rerank_score

        # Сортируем по убыванию rerank_score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked


class LLMReranker(BaseReranker):
    """
    Reranker на базе LLM (через API).
    Использует LLM для оценки релевантности документов.

    Это более дорогой, но потенциально более точный метод.
    Подходит для небольшого числа кандидатов.
    """

    PROMPT_TEMPLATE = """
Rate the relevance of the following document to the query on a scale of 0-10.
Only output the number, nothing else.

Query: {query}

Document: {document}

Relevance score (0-10):"""

    def __init__(self, llm_client, model: str = "openai/gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model
        self._info = RerankerInfo(
            name="llm-reranker",
            model_id=model,
            description="LLM-based reranker for high-quality relevance scoring",
        )

    @property
    def info(self) -> RerankerInfo:
        return self._info

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        if not candidates:
            return []

        # Ограничиваем документы для экономии токенов
        max_doc_len = 500

        for c in candidates:
            doc_text = c.get("text", "")[:max_doc_len]
            prompt = self.PROMPT_TEMPLATE.format(query=query, document=doc_text)

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=5,
                )
                score_text = response.choices[0].message.content.strip()
                score = float(score_text) / 10.0  # Нормализуем к [0, 1]
            except Exception as e:
                logger.warning(f"LLM reranking failed: {e}")
                score = c.get("score", 0)

            c["rerank_score"] = score
            c["original_score"] = c.get("score", 0)
            c["score"] = score

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked


# =============================================================================
# REGISTRY
# =============================================================================

RERANKER_REGISTRY: Dict[str, Dict] = {
    "ms-marco-mini": {
        "class": CrossEncoderReranker,
        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "Быстрый reranker (MiniLM-L6), хороший баланс",
    },
    "ms-marco-medium": {
        "class": CrossEncoderReranker,
        "model_id": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "description": "Средний reranker (MiniLM-L12), лучше качество",
    },
    "ms-marco-tiny": {
        "class": CrossEncoderReranker,
        "model_id": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "description": "Самый быстрый reranker (TinyBERT), для realtime",
    },
    "bge-reranker-base": {
        "class": CrossEncoderReranker,
        "model_id": "BAAI/bge-reranker-base",
        "description": "BGE reranker base, SOTA качество",
    },
    "bge-reranker-large": {
        "class": CrossEncoderReranker,
        "model_id": "BAAI/bge-reranker-large",
        "description": "BGE reranker large, лучшее качество",
    },
}


def get_reranker(name: str, device: str = "cpu") -> BaseReranker:
    """
    Фабрика для создания reranker'ов по имени.

    Args:
        name: Имя reranker'а из RERANKER_REGISTRY или полный model_id
        device: Устройство (cpu, cuda, mps)

    Returns:
        BaseReranker: Инстанс reranker'а
    """
    if name in RERANKER_REGISTRY:
        config = RERANKER_REGISTRY[name]
        reranker_class = config["class"]
        model_id = config["model_id"]
        return reranker_class(model_id=model_id, device=device)

    # Если не в реестре, пробуем как полный model_id
    if "/" in name:
        return CrossEncoderReranker(model_id=name, device=device)

    raise ValueError(f"Unknown reranker: {name}. Available: {list(RERANKER_REGISTRY.keys())}")


def list_rerankers() -> List[Dict]:
    """Возвращает список всех доступных reranker'ов."""
    return [
        {
            "name": name,
            "model_id": config["model_id"],
            "description": config["description"],
        }
        for name, config in RERANKER_REGISTRY.items()
    ]


# =============================================================================
# PIPELINE HELPERS
# =============================================================================

def retrieve_and_rerank(
    retriever_fn,
    reranker: BaseReranker,
    query: str,
    retrieve_k: int = 20,
    final_k: int = 5
) -> List[Dict]:
    """
    Полный pipeline: retrieval → reranking.

    Args:
        retriever_fn: Функция retrieval (возвращает List[Dict])
        reranker: Инстанс reranker'а
        query: Поисковый запрос
        retrieve_k: Сколько кандидатов получить от retriever
        final_k: Сколько вернуть после reranking

    Returns:
        List[Dict]: Переранжированные результаты
    """
    # Stage 1: Retrieval
    candidates = retriever_fn(query, k=retrieve_k)

    if not candidates:
        return []

    # Stage 2: Reranking
    reranked = reranker.rerank(query, candidates, top_k=final_k)

    # Добавляем метаданные о pipeline
    for i, r in enumerate(reranked):
        r["final_rank"] = i + 1
        r["pipeline"] = "retrieve+rerank"

    return reranked
