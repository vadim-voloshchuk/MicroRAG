"""
Embedders - Модульная система эмбеддингов.

Поддерживает несколько SOTA моделей для честного сравнения.
Отвечает на требование пункта 5: "embeddings matter".
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import List, Optional, Dict, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbedderInfo:
    """Информация о модели эмбеддингов."""
    name: str
    model_id: str
    dim: int
    description: str
    max_seq_length: int = 512


class BaseEmbedder(ABC):
    """Базовый класс для всех эмбеддеров."""

    @property
    @abstractmethod
    def info(self) -> EmbedderInfo:
        """Возвращает информацию о модели."""
        pass

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Кодирует тексты в эмбеддинги.

        Args:
            texts: Один текст или список текстов
            normalize: Нормализовать ли векторы (L2)
            show_progress: Показывать прогресс-бар

        Returns:
            np.ndarray: Матрица эмбеддингов (N, dim)
        """
        pass

    def encode_query(self, query: str) -> np.ndarray:
        """
        Кодирует запрос.
        Некоторые модели требуют специальные префиксы для запросов.
        """
        return self.encode(query, normalize=True)[0]

    def encode_documents(
        self,
        docs: List[str],
        show_progress: bool = False,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Кодирует документы.
        Некоторые модели требуют специальные префиксы для документов.
        """
        return self.encode(
            docs,
            normalize=True,
            show_progress=show_progress,
            batch_size=batch_size,
        )


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Эмбеддер на базе sentence-transformers.
    Поддерживает большинство популярных моделей.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.device = device
        self._model = None
        self._info = None

    @property
    def model(self):
        """Lazy loading модели."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.model_id}")
            self._model = SentenceTransformer(self.model_id, device=self.device)
            env_max_len = os.getenv("EMBED_MAX_SEQ_LEN")
            if env_max_len:
                try:
                    max_len = int(env_max_len)
                except ValueError:
                    max_len = None
                if max_len and max_len > 0:
                    self._model.max_seq_length = max_len
        return self._model

    @property
    def info(self) -> EmbedderInfo:
        if self._info is None:
            dim = self.model.get_sentence_embedding_dimension()
            max_len = self.model.max_seq_length
            self._info = EmbedderInfo(
                name=self.model_id.split("/")[-1],
                model_id=self.model_id,
                dim=dim,
                description=f"SentenceTransformer model",
                max_seq_length=max_len,
            )
        return self._info

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        encode_kwargs = {
            "normalize_embeddings": normalize,
            "show_progress_bar": show_progress,
            "convert_to_numpy": True,
        }
        if batch_size is not None:
            encode_kwargs["batch_size"] = batch_size

        embeddings = self.model.encode(texts, **encode_kwargs)
        return embeddings


class BGEEmbedder(SentenceTransformerEmbedder):
    """
    BGE (BAAI General Embedding) - SOTA модель для retrieval.
    Требует специальный префикс для запросов.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def encode_query(self, query: str) -> np.ndarray:
        """Добавляет специальный префикс для запросов BGE."""
        prefixed_query = self.QUERY_PREFIX + query
        return self.encode(prefixed_query, normalize=True)[0]


class E5Embedder(SentenceTransformerEmbedder):
    """
    E5 (Embeddings from bidirectional Encoder representations) - альтернатива BGE.
    Требует prefixы "query:" и "passage:" для запросов и документов соответственно.
    """

    def encode_query(self, query: str) -> np.ndarray:
        """Добавляет prefix 'query:' для запросов E5."""
        return self.encode(f"query: {query}", normalize=True)[0]

    def encode_documents(
        self,
        docs: List[str],
        show_progress: bool = False,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Добавляет prefix 'passage:' для документов E5."""
        prefixed_docs = [f"passage: {doc}" for doc in docs]
        return self.encode(
            prefixed_docs,
            normalize=True,
            show_progress=show_progress,
            batch_size=batch_size,
        )


# =============================================================================
# REGISTRY - Реестр доступных эмбеддеров
# =============================================================================

EMBEDDER_REGISTRY: Dict[str, Dict] = {
    # ==========================================================================
    # МУЛЬТИЯЗЫЧНЫЕ (русский + английский) - РЕКОМЕНДУЕМЫЕ для RU вопросов + EN даташитов
    # ==========================================================================
    "bge-m3": {
        "class": BGEEmbedder,
        "model_id": "BAAI/bge-m3",
        "description": "SOTA мультиязычная (1024d), 100+ языков, лучший выбор для RU+EN",
    },
    "multilingual-e5-base": {
        "class": E5Embedder,
        "model_id": "intfloat/multilingual-e5-base",
        "description": "Мультиязычная E5 (768d), хорошо для RU+EN, рекомендуется",
    },
    "multilingual-e5-large": {
        "class": E5Embedder,
        "model_id": "intfloat/multilingual-e5-large",
        "description": "Мультиязычная E5 large (1024d), лучшее качество",
    },
    "multilingual-minilm": {
        "class": SentenceTransformerEmbedder,
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Мультиязычная MiniLM (384d), быстрая, 50+ языков",
    },
    "multilingual-mpnet": {
        "class": SentenceTransformerEmbedder,
        "model_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "description": "Мультиязычная MPNet (768d), хороший баланс скорость/качество",
    },
    "labse": {
        "class": SentenceTransformerEmbedder,
        "model_id": "sentence-transformers/LaBSE",
        "description": "Language-agnostic BERT (768d), 109 языков, от Google",
    },

    # ==========================================================================
    # АНГЛИЙСКИЕ (baseline для сравнения)
    # ==========================================================================
    "minilm": {
        "class": SentenceTransformerEmbedder,
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Англ. MiniLM (384d), быстрая на CPU, baseline",
    },
    "bge-base-en": {
        "class": BGEEmbedder,
        "model_id": "BAAI/bge-base-en-v1.5",
        "description": "BGE англ. (768d), SOTA для чисто англ. текстов",
    },
}


def get_embedder(name: str, device: str = "cpu") -> BaseEmbedder:
    """
    Фабрика для создания эмбеддеров по имени.

    Args:
        name: Имя эмбеддера из EMBEDDER_REGISTRY или полный model_id
        device: Устройство (cpu, cuda, mps)

    Returns:
        BaseEmbedder: Инстанс эмбеддера
    """
    if name in EMBEDDER_REGISTRY:
        config = EMBEDDER_REGISTRY[name]
        embedder_class = config["class"]
        model_id = config["model_id"]
        return embedder_class(model_id=model_id, device=device)

    # Если не в реестре, пробуем как полный model_id
    if "/" in name:
        # Определяем тип по имени модели
        if "bge" in name.lower():
            return BGEEmbedder(model_id=name, device=device)
        elif "e5" in name.lower():
            return E5Embedder(model_id=name, device=device)
        else:
            return SentenceTransformerEmbedder(model_id=name, device=device)

    raise ValueError(f"Unknown embedder: {name}. Available: {list(EMBEDDER_REGISTRY.keys())}")


def list_embedders() -> List[Dict]:
    """Возвращает список всех доступных эмбеддеров с описаниями."""
    return [
        {
            "name": name,
            "model_id": config["model_id"],
            "description": config["description"],
        }
        for name, config in EMBEDDER_REGISTRY.items()
    ]


# =============================================================================
# CACHING - Кэширование эмбеддингов
# =============================================================================

class EmbeddingCache:
    """
    Простой кэш для эмбеддингов.
    Позволяет не пересчитывать эмбеддинги для одинаковых текстов.
    """

    def __init__(self, embedder: BaseEmbedder, max_size: int = 10000):
        self.embedder = embedder
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Кодирует тексты с использованием кэша."""
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = f"{text[:100]}_{normalize}"  # Ключ: начало текста + флаг
            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Кодируем некэшированные
        if uncached_texts:
            new_embeddings = self.embedder.encode(uncached_texts, normalize=normalize)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                cache_key = f"{text[:100]}_{normalize}"
                if len(self._cache) < self.max_size:
                    self._cache[cache_key] = emb
                results.append((idx, emb))

        # Сортируем по оригинальному порядку
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])

    def clear(self):
        """Очищает кэш."""
        self._cache.clear()


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def get_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Legacy функция для обратной совместимости."""
    embedder = get_embedder(model_name)
    return embedder.model
