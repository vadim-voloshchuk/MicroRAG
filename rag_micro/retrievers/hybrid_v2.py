"""
Hybrid Retrieval v2 - Корректное объединение BM25 и Dense с нормализацией.

Отвечает на вопросы 7, 15:
- Нормализация скорингов (minmax/zscore) для сопоставимости
- Интерпретируемый параметр alpha для взвешивания методов
- Логирование вкладов каждого метода
"""
from __future__ import annotations
from typing import List, Dict, Literal, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from .bm25_whoosh import search_whoosh
from .faiss_dense import search_faiss

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Конфигурация гибридного поиска."""
    alpha: float = 0.5  # Вес BM25: 0.0 = только dense, 1.0 = только BM25
    normalization: Literal["minmax", "zscore", "none"] = "minmax"
    log_contributions: bool = True


def normalize_scores_minmax(scores: List[float]) -> List[float]:
    """
    Min-Max нормализация в диапазон [0, 1].

    score_norm = (score - min) / (max - min)
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s

    if range_s == 0:
        return [0.5] * len(scores)  # Все одинаковые

    return [(s - min_s) / range_s for s in scores]


def normalize_scores_zscore(scores: List[float]) -> List[float]:
    """
    Z-score нормализация (стандартизация).

    score_norm = (score - mean) / std

    Затем преобразуем в [0, 1] через sigmoid-like функцию.
    """
    if not scores:
        return []

    arr = np.array(scores)
    mean = arr.mean()
    std = arr.std()

    if std == 0:
        return [0.5] * len(scores)

    z_scores = (arr - mean) / std

    # Преобразуем z-scores в [0, 1] через sigmoid
    # sigmoid(z) = 1 / (1 + exp(-z))
    normalized = 1 / (1 + np.exp(-z_scores))
    return normalized.tolist()


def normalize_scores(
    scores: List[float],
    method: Literal["minmax", "zscore", "none"] = "minmax"
) -> List[float]:
    """Нормализует scores выбранным методом."""
    if method == "none":
        return scores
    elif method == "minmax":
        return normalize_scores_minmax(scores)
    elif method == "zscore":
        return normalize_scores_zscore(scores)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def search_hybrid_v2(
    index_root: str,
    query: str,
    k: int = 5,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    config: Optional[HybridConfig] = None
) -> List[Dict]:
    """
    Гибридный поиск с корректной нормализацией и взвешиванием.

    Args:
        index_root: Корень индексов
        query: Поисковый запрос
        k: Количество результатов
        embed_model: Модель эмбеддингов для dense
        config: Конфигурация гибридного поиска

    Returns:
        List[Dict]: Объединённые результаты с нормализованными scores
    """
    if config is None:
        config = HybridConfig()

    alpha = config.alpha
    norm_method = config.normalization

    # Получаем результаты от обоих методов
    # Берём больше кандидатов для лучшего объединения
    retrieve_k = k * 3

    bm25_results = search_whoosh(f"{index_root}/bm25", query, k=retrieve_k)
    dense_results = search_faiss(f"{index_root}/faiss", query, k=retrieve_k, embed_model=embed_model)

    # Логируем статистику до нормализации
    if config.log_contributions:
        bm25_scores = [r["score"] for r in bm25_results]
        dense_scores = [r["score"] for r in dense_results]
        if bm25_scores:
            logger.info(
                "[hybrid] BM25 scores: min=%.3f, max=%.3f, mean=%.3f",
                min(bm25_scores),
                max(bm25_scores),
                np.mean(bm25_scores),
            )
        else:
            logger.info("[hybrid] BM25 scores: empty")
        if dense_scores:
            logger.info(
                "[hybrid] Dense scores: min=%.3f, max=%.3f, mean=%.3f",
                min(dense_scores),
                max(dense_scores),
                np.mean(dense_scores),
            )
        else:
            logger.info("[hybrid] Dense scores: empty")

    # Нормализуем scores
    bm25_scores_norm = normalize_scores([r["score"] for r in bm25_results], norm_method)
    dense_scores_norm = normalize_scores([r["score"] for r in dense_results], norm_method)

    # Обновляем scores в результатах
    for r, score_norm in zip(bm25_results, bm25_scores_norm):
        r["score_original"] = r["score"]
        r["score_normalized"] = score_norm
        r["retriever"] = "bm25"

    for r, score_norm in zip(dense_results, dense_scores_norm):
        r["score_original"] = r["score"]
        r["score_normalized"] = score_norm
        r["retriever"] = "dense"

    # Объединяем результаты с учётом alpha
    # Создаём словарь id -> результат с агрегированным score
    combined: Dict[str, Dict] = {}

    for r in bm25_results:
        rid = r["id"]
        if rid not in combined:
            combined[rid] = r.copy()
            combined[rid]["bm25_score"] = r["score_normalized"]
            combined[rid]["dense_score"] = 0.0
        else:
            combined[rid]["bm25_score"] = r["score_normalized"]

    for r in dense_results:
        rid = r["id"]
        if rid not in combined:
            combined[rid] = r.copy()
            combined[rid]["dense_score"] = r["score_normalized"]
            combined[rid]["bm25_score"] = 0.0
        else:
            combined[rid]["dense_score"] = r["score_normalized"]

    # Вычисляем финальный score: alpha * BM25 + (1 - alpha) * Dense
    for rid, r in combined.items():
        bm25_contrib = alpha * r.get("bm25_score", 0)
        dense_contrib = (1 - alpha) * r.get("dense_score", 0)
        r["score"] = bm25_contrib + dense_contrib
        r["bm25_contribution"] = bm25_contrib
        r["dense_contribution"] = dense_contrib

    # Сортируем по финальному score
    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

    # Логируем статистику после объединения
    if config.log_contributions:
        avg_bm25_contrib = np.mean([r["bm25_contribution"] for r in sorted_results[:k]])
        avg_dense_contrib = np.mean([r["dense_contribution"] for r in sorted_results[:k]])
        logger.info(f"[hybrid] alpha={alpha:.2f}, avg BM25 contrib={avg_bm25_contrib:.3f}, avg Dense contrib={avg_dense_contrib:.3f}")

    return sorted_results[:k]


def search_hybrid_with_rerank(
    index_root: str,
    query: str,
    reranker,
    k: int = 5,
    retrieve_k: int = 20,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    config: Optional[HybridConfig] = None
) -> List[Dict]:
    """
    Hybrid + Reranker pipeline.

    Args:
        index_root: Корень индексов
        query: Поисковый запрос
        reranker: Инстанс reranker'а
        k: Финальное количество результатов
        retrieve_k: Сколько кандидатов получить для reranking
        embed_model: Модель эмбеддингов
        config: Конфигурация гибридного поиска

    Returns:
        List[Dict]: Переранжированные результаты
    """
    # Stage 1: Hybrid retrieval
    candidates = search_hybrid_v2(index_root, query, k=retrieve_k, embed_model=embed_model, config=config)

    if not candidates:
        return []

    # Stage 2: Reranking
    reranked = reranker.rerank(query, candidates, top_k=k)

    # Добавляем метаданные
    for i, r in enumerate(reranked):
        r["final_rank"] = i + 1
        r["pipeline"] = "hybrid+rerank"

    return reranked


# =============================================================================
# ABLATION HELPERS - Для экспериментов с alpha
# =============================================================================

def run_alpha_ablation(
    index_root: str,
    query: str,
    embed_model: str,
    alphas: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    k: int = 5,
    normalization: str = "minmax"
) -> Dict[float, List[Dict]]:
    """
    Прогоняет поиск с разными значениями alpha.

    Полезно для:
    - Понимания вклада каждого метода
    - Выбора оптимального alpha
    - Демонстрации интерпретируемости hybrid

    Returns:
        Dict[float, List[Dict]]: Результаты для каждого alpha
    """
    results = {}

    for alpha in alphas:
        config = HybridConfig(alpha=alpha, normalization=normalization, log_contributions=False)
        results[alpha] = search_hybrid_v2(index_root, query, k=k, embed_model=embed_model, config=config)

    return results


def analyze_method_agreement(
    index_root: str,
    query: str,
    embed_model: str,
    k: int = 10
) -> Dict:
    """
    Анализирует согласованность BM25 и Dense методов.

    Показывает:
    - Перекрытие результатов (overlap)
    - Разницу в ранжировании (rank correlation)
    - Уникальные находки каждого метода

    Returns:
        Dict: Статистика согласованности
    """
    bm25_results = search_whoosh(f"{index_root}/bm25", query, k=k)
    dense_results = search_faiss(f"{index_root}/faiss", query, k=k, embed_model=embed_model)

    bm25_ids = set(r["id"] for r in bm25_results)
    dense_ids = set(r["id"] for r in dense_results)

    overlap = bm25_ids & dense_ids
    bm25_only = bm25_ids - dense_ids
    dense_only = dense_ids - bm25_ids

    return {
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / k,
        "bm25_unique_count": len(bm25_only),
        "dense_unique_count": len(dense_only),
        "overlap_ids": list(overlap),
        "bm25_only_ids": list(bm25_only),
        "dense_only_ids": list(dense_only),
    }


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def search_hybrid(index_root: str, query: str, k: int, embed_model: str) -> List[Dict]:
    """Legacy функция для обратной совместимости."""
    return search_hybrid_v2(index_root, query, k=k, embed_model=embed_model)


def dedup(results: List[Dict], max_items: int) -> List[Dict]:
    """Legacy функция для обратной совместимости."""
    seen = set()
    out = []
    sorted_res = sorted(results, key=lambda x: x["score"], reverse=True)
    for r in sorted_res:
        if r["id"] in seen:
            continue
        out.append(r)
        seen.add(r["id"])
        if len(out) >= max_items:
            break
    return out
