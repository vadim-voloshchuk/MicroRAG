#!/usr/bin/env python3
"""
Расширенные SOTA эксперименты для Q1 публикации.

Включает:
1. Baseline: BM25 + SOTA dense alternatives
2. Fusion методы: Hybrid (MinMax), RRF, CombSUM, CombMNZ
3. Полная сетка alpha: 0.1 - 0.9
4. Альтернативные rerankers: ms-marco-MiniLM, BGE-reranker
5. Ablation study по fusion методам
6. Bootstrap 95% CI для всех метрик
"""
import os
import sys
import json
import csv
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np

# Добавляем корень проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_micro.retrievers.embedders import EMBEDDER_REGISTRY, get_embedder
from rag_micro.retrievers.faiss_dense import build_faiss_index
from rag_micro.retrievers.splade_sparse import build_splade_index, SpladeRetriever
from rag_micro.retrievers.unicoil_sparse import build_unicoil_index, UniCOILRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================================
# CROSS-LINGUAL KEYWORD EXTRACTION (RU -> EN)
# =========================================================================

import re

RU_EN_KEYWORDS = {
    "частота": "frequency MHz clock",
    "память": "memory RAM flash SRAM",
    "напряжение": "voltage supply VDD VCC",
    "ток": "current mA consumption",
    "температура": "temperature operating",
    "gpio": "GPIO pins I/O",
    "uart": "UART serial",
    "spi": "SPI",
    "i2c": "I2C",
    "adc": "ADC analog",
    "pwm": "PWM timer",
    "таймер": "timer",
    "максимальн": "maximum max",
    "минимальн": "minimum min",
    "питание": "power supply",
    "потребление": "consumption power",
    "интерфейс": "interface",
    "wifi": "WiFi wireless",
    "bluetooth": "Bluetooth BLE",
    "ядро": "core CPU ARM",
    "регистр": "register",
    "прерывание": "interrupt IRQ",
    "dma": "DMA",
    "usb": "USB",
    "can": "CAN bus",
    "ethernet": "Ethernet MAC",
    "размер": "size",
    "корпус": "package QFP LQFP",
}


def extract_search_terms(query: str) -> str:
    """Извлекает английские поисковые термины из русского запроса."""
    query_lower = query.lower()
    terms = []

    # Английские слова/числа напрямую
    english_terms = re.findall(r'[A-Za-z0-9][\w\-\.]*', query)
    terms.extend(english_terms)

    # Переводы русских ключевых слов
    for ru_key, en_terms in RU_EN_KEYWORDS.items():
        if ru_key in query_lower:
            terms.extend(en_terms.split())

    # Убираем дубликаты
    seen = set()
    unique_terms = []
    for t in terms:
        t_lower = t.lower()
        if t_lower not in seen and len(t) > 1:
            seen.add(t_lower)
            unique_terms.append(t)

    return " OR ".join(unique_terms) if unique_terms else query


# =========================================================================
# BM25 RETRIEVER
# =========================================================================

def search_bm25(index_dir: str, query: str, k: int = 5) -> List[Dict]:
    """BM25 поиск с кросс-языковым извлечением терминов."""
    from whoosh import index
    from whoosh.qparser import MultifieldParser, OrGroup

    ix = index.open_dir(index_dir)
    parser = MultifieldParser(["text", "doc"], schema=ix.schema, group=OrGroup)
    search_query = extract_search_terms(query)
    q = parser.parse(search_query)

    out = []
    with ix.searcher() as s:
        results = s.search(q, limit=k)
        for r in results:
            out.append({
                "id": r["id"],
                "doc": r["doc"],
                "page": int(r["page"]),
                "text": r["text"],
                "score": float(r.score),
                "retriever": "bm25",
            })
    return out


def resolve_unicoil_batch_size() -> int:
    value = os.getenv("UNICOIL_BATCH_SIZE")
    if not value:
        return 8
    try:
        batch_size = int(value)
    except ValueError:
        return 8
    return batch_size if batch_size > 0 else 8


def resolve_unicoil_max_length() -> int:
    value = os.getenv("UNICOIL_MAX_LENGTH")
    if not value:
        return 256
    try:
        max_length = int(value)
    except ValueError:
        return 256
    return max_length if max_length > 0 else 256


def resolve_unicoil_top_k() -> int:
    value = os.getenv("UNICOIL_TOP_K")
    if not value:
        return 128
    try:
        top_k = int(value)
    except ValueError:
        return 128
    return top_k if top_k > 0 else 128


# =========================================================================
# DENSE RETRIEVER
# =========================================================================

class DenseRetriever:
    """Dense retriever с мультиязычными эмбеддингами."""

    def __init__(self, index_dir: str, model_name: str, device: str = "cpu"):
        import faiss

        self.model_name = model_name
        self.device = device
        self.embedder = get_embedder(model_name, device=device)

        # FAISS индекс
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        logger.info(f"FAISS index: {self.index.ntotal} vectors")

        # Метаданные
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.items = json.load(f)["items"]

        # Эмбеддер
        logger.info(f"Loading embedder: {self.embedder.info.model_id}")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        embedding = self.embedder.encode_query(query)
        embedding = np.asarray(embedding, dtype="float32").reshape(1, -1)

        D, I = self.index.search(embedding, k)

        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if 0 <= idx < len(self.items):
                rec = self.items[idx]
                out.append({
                    "id": rec["id"],
                    "doc": rec["doc"],
                    "page": int(rec["page"] or 0),
                    "text": rec["text"],
                    "score": float(score),
                    "retriever": "dense",
                })
        return out


def resolve_model_id(embedder_name: str) -> str:
    """Возвращает model_id для имени из реестра или raw model_id."""
    if embedder_name in EMBEDDER_REGISTRY:
        return EMBEDDER_REGISTRY[embedder_name]["model_id"]
    return embedder_name


def sanitize_model_id(model_id: str) -> str:
    """Безопасное имя для директории индекса."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def resolve_faiss_index_dir(index_root: str, embedder_name: str) -> str:
    model_id = resolve_model_id(embedder_name)
    default_dir = os.path.join(index_root, "faiss")
    if embedder_name in (
        "multilingual-minilm",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        return default_dir
    return os.path.join(index_root, f"faiss_{sanitize_model_id(model_id)}")


def ensure_faiss_index(
    corpus_jsonl: str,
    index_root: str,
    embedder_name: str,
    device: str = "cpu",
    batch_size: int | None = None,
) -> str:
    index_dir = resolve_faiss_index_dir(index_root, embedder_name)
    index_path = os.path.join(index_dir, "faiss.index")
    if not os.path.exists(corpus_jsonl):
        raise FileNotFoundError(f"Corpus not found: {corpus_jsonl}")
    if not os.path.exists(index_path):
        model_id = resolve_model_id(embedder_name)
        logger.info(f"Building FAISS index for {model_id} -> {index_dir}")
        build_faiss_index(
            corpus_jsonl,
            index_dir,
            embedder_name,
            device=device,
            batch_size=batch_size,
        )
    return index_dir


def resolve_splade_index_dir(index_root: str, model_id: str) -> str:
    return os.path.join(index_root, f"splade_{sanitize_model_id(model_id)}")


def ensure_splade_index(
    corpus_jsonl: str,
    index_root: str,
    model_id: str,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 256,
    top_k: int = 128,
) -> str:
    index_dir = resolve_splade_index_dir(index_root, model_id)
    index_path = os.path.join(index_dir, "splade.npz")
    if not os.path.exists(corpus_jsonl):
        raise FileNotFoundError(f"Corpus not found: {corpus_jsonl}")
    if not os.path.exists(index_path):
        logger.info(f"Building SPLADE index for {model_id} -> {index_dir}")
        build_splade_index(
            corpus_jsonl,
            index_dir,
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            top_k=top_k,
        )
    return index_dir


def resolve_unicoil_index_dir(index_root: str, model_id: str) -> str:
    return os.path.join(index_root, f"unicoil_{sanitize_model_id(model_id)}")


def ensure_unicoil_index(
    corpus_jsonl: str,
    index_root: str,
    model_id: str,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 256,
    top_k: int = 128,
) -> str:
    index_dir = resolve_unicoil_index_dir(index_root, model_id)
    index_path = os.path.join(index_dir, "unicoil.npz")
    if not os.path.exists(corpus_jsonl):
        raise FileNotFoundError(f"Corpus not found: {corpus_jsonl}")
    if not os.path.exists(index_path):
        logger.info(f"Building UniCOIL index for {model_id} -> {index_dir}")
        build_unicoil_index(
            corpus_jsonl,
            index_dir,
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            top_k=top_k,
        )
    return index_dir


def resolve_embed_device() -> str:
    """Определяет устройство для эмбеддингов (cpu/cuda/mps)."""
    env_device = os.getenv("EMBED_DEVICE")
    if env_device:
        return env_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_embed_batch_size() -> int | None:
    """Batch size для эмбеддингов из окружения."""
    value = os.getenv("EMBED_BATCH_SIZE")
    if not value:
        return None
    try:
        batch_size = int(value)
    except ValueError:
        return None
    return batch_size if batch_size > 0 else None


def resolve_splade_batch_size() -> int:
    """Batch size for SPLADE indexing (separate from dense)."""
    value = os.getenv("SPLADE_BATCH_SIZE")
    if not value:
        return 8
    try:
        batch_size = int(value)
    except ValueError:
        return 8
    return batch_size if batch_size > 0 else 8


def resolve_splade_max_length() -> int:
    value = os.getenv("SPLADE_MAX_LENGTH")
    if not value:
        return 256
    try:
        max_length = int(value)
    except ValueError:
        return 256
    return max_length if max_length > 0 else 256


def resolve_rerank_device() -> str:
    """Устройство для reranker (по умолчанию как EMBED_DEVICE)."""
    env_device = os.getenv("RERANK_DEVICE")
    if env_device:
        return env_device
    return resolve_embed_device()


def resolve_rerank_batch_size() -> int | None:
    """Batch size для rerank из окружения."""
    value = os.getenv("RERANK_BATCH_SIZE")
    if not value:
        return None
    try:
        batch_size = int(value)
    except ValueError:
        return None
    return batch_size if batch_size > 0 else None


# =========================================================================
# FUSION METHODS (SOTA)
# =========================================================================

def normalize_minmax(scores: List[float]) -> List[float]:
    """Min-Max нормализация в [0, 1]."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-9:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def normalize_zscore(scores: List[float]) -> List[float]:
    """Z-score нормализация."""
    if not scores:
        return []
    arr = np.array(scores)
    mean, std = arr.mean(), arr.std()
    if std < 1e-9:
        return [0.0] * len(scores)
    normalized = (arr - mean) / std
    # Преобразуем в [0, 1] для сравнимости
    min_n, max_n = normalized.min(), normalized.max()
    if max_n - min_n < 1e-9:
        return [0.5] * len(scores)
    return ((normalized - min_n) / (max_n - min_n)).tolist()


def hybrid_weighted(
    bm25_results: List[Dict],
    dense_results: List[Dict],
    alpha: float = 0.5,
    k: int = 5,
    normalization: str = "minmax"
) -> List[Dict]:
    """
    Weighted Hybrid Fusion.
    score = alpha * normalized_dense + (1 - alpha) * normalized_bm25
    """
    normalize_fn = normalize_minmax if normalization == "minmax" else normalize_zscore

    bm25_scores = [r["score"] for r in bm25_results]
    dense_scores = [r["score"] for r in dense_results]

    bm25_norm = normalize_fn(bm25_scores)
    dense_norm = normalize_fn(dense_scores)

    combined = {}

    for i, r in enumerate(bm25_results):
        doc_id = r["id"]
        combined[doc_id] = {
            **r,
            "bm25_score_norm": bm25_norm[i] if i < len(bm25_norm) else 0,
            "dense_score_norm": 0,
            "retriever": "hybrid",
        }

    for i, r in enumerate(dense_results):
        doc_id = r["id"]
        if doc_id in combined:
            combined[doc_id]["dense_score_norm"] = dense_norm[i] if i < len(dense_norm) else 0
        else:
            combined[doc_id] = {
                **r,
                "bm25_score_norm": 0,
                "dense_score_norm": dense_norm[i] if i < len(dense_norm) else 0,
                "retriever": "hybrid",
            }

    for doc_id, r in combined.items():
        r["score"] = alpha * r["dense_score_norm"] + (1 - alpha) * r["bm25_score_norm"]

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:k]


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k_rrf: int = 60,
    k: int = 5
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) - SOTA fusion method.

    RRF_score(d) = Σ 1 / (k_rrf + rank(d))

    References:
    - Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet and
      individual Rank Learning Methods"
    """
    scores = defaultdict(float)
    docs = {}

    for result_list in result_lists:
        for rank, r in enumerate(result_list):
            doc_id = r["id"]
            scores[doc_id] += 1.0 / (k_rrf + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = r.copy()

    for doc_id in docs:
        docs[doc_id]["score"] = scores[doc_id]
        docs[doc_id]["retriever"] = "rrf"

    sorted_results = sorted(docs.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:k]


def combsum_fusion(
    bm25_results: List[Dict],
    dense_results: List[Dict],
    k: int = 5,
    normalization: str = "minmax"
) -> List[Dict]:
    """
    CombSUM Fusion - простое суммирование нормализованных скоров.

    score = normalized_bm25 + normalized_dense

    Reference: Fox & Shaw (1994) "Combination of Multiple Searches"
    """
    normalize_fn = normalize_minmax if normalization == "minmax" else normalize_zscore

    bm25_scores = [r["score"] for r in bm25_results]
    dense_scores = [r["score"] for r in dense_results]

    bm25_norm = normalize_fn(bm25_scores)
    dense_norm = normalize_fn(dense_scores)

    combined = {}

    for i, r in enumerate(bm25_results):
        doc_id = r["id"]
        combined[doc_id] = {
            **r,
            "bm25_score_norm": bm25_norm[i] if i < len(bm25_norm) else 0,
            "dense_score_norm": 0,
            "retriever": "combsum",
        }

    for i, r in enumerate(dense_results):
        doc_id = r["id"]
        if doc_id in combined:
            combined[doc_id]["dense_score_norm"] = dense_norm[i] if i < len(dense_norm) else 0
        else:
            combined[doc_id] = {
                **r,
                "bm25_score_norm": 0,
                "dense_score_norm": dense_norm[i] if i < len(dense_norm) else 0,
                "retriever": "combsum",
            }

    for doc_id, r in combined.items():
        r["score"] = r["bm25_score_norm"] + r["dense_score_norm"]

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:k]


def combmnz_fusion(
    bm25_results: List[Dict],
    dense_results: List[Dict],
    k: int = 5,
    normalization: str = "minmax"
) -> List[Dict]:
    """
    CombMNZ Fusion - сумма с множителем на количество источников.

    score = (normalized_bm25 + normalized_dense) * num_sources

    Reference: Fox & Shaw (1994) "Combination of Multiple Searches"
    """
    normalize_fn = normalize_minmax if normalization == "minmax" else normalize_zscore

    bm25_scores = [r["score"] for r in bm25_results]
    dense_scores = [r["score"] for r in dense_results]

    bm25_norm = normalize_fn(bm25_scores)
    dense_norm = normalize_fn(dense_scores)

    combined = {}

    for i, r in enumerate(bm25_results):
        doc_id = r["id"]
        combined[doc_id] = {
            **r,
            "bm25_score_norm": bm25_norm[i] if i < len(bm25_norm) else 0,
            "dense_score_norm": 0,
            "num_sources": 1,
            "retriever": "combmnz",
        }

    for i, r in enumerate(dense_results):
        doc_id = r["id"]
        if doc_id in combined:
            combined[doc_id]["dense_score_norm"] = dense_norm[i] if i < len(dense_norm) else 0
            combined[doc_id]["num_sources"] = 2  # Найден в обоих
        else:
            combined[doc_id] = {
                **r,
                "bm25_score_norm": 0,
                "dense_score_norm": dense_norm[i] if i < len(dense_norm) else 0,
                "num_sources": 1,
                "retriever": "combmnz",
            }

    for doc_id, r in combined.items():
        sum_scores = r["bm25_score_norm"] + r["dense_score_norm"]
        r["score"] = sum_scores * r["num_sources"]

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:k]


# =========================================================================
# RERANKERS
# =========================================================================

class CrossEncoderReranker:
    """Cross-Encoder для переранжирования."""

    def __init__(
        self,
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int | None = None,
    ):
        from sentence_transformers import CrossEncoder
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        logger.info(f"Loading CrossEncoder: {model_id} on {device}")
        self.model = CrossEncoder(model_id, device=device)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not candidates:
            return []

        pairs = [(query, c.get("text", "")) for c in candidates]
        predict_kwargs = {"show_progress_bar": False}
        if self.batch_size is not None:
            predict_kwargs["batch_size"] = self.batch_size
        scores = self.model.predict(pairs, **predict_kwargs)

        for i, score in enumerate(scores):
            candidates[i]["original_score"] = candidates[i].get("score", 0)
            candidates[i]["rerank_score"] = float(score)
            candidates[i]["score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# =========================================================================
# EVALUATION METRICS
# =========================================================================

@dataclass
class RelevanceJudgment:
    query_id: str
    query: str
    relevant_doc_ids: List[str]
    relevant_pages: Optional[List[int]] = None
    relevant_chunk_ids: Optional[List[str]] = None


def recall_at_k(retrieved: List[Dict], judgment: RelevanceJudgment, k: int) -> float:
    """Recall@K (бинарная метрика - Hit@K)."""
    if not judgment.relevant_doc_ids:
        return 0.0
    for r in retrieved[:k]:
        doc_name = r.get("doc", "")
        if doc_name in judgment.relevant_doc_ids:
            return 1.0
    return 0.0


def chunk_recall_at_k(retrieved: List[Dict], judgment: RelevanceJudgment, k: int) -> float:
    """Recall@K на уровне чанков."""
    if not judgment.relevant_chunk_ids:
        if not judgment.relevant_doc_ids:
            return 0.0
        for r in retrieved[:k]:
            doc_name = r.get("doc", "")
            page = r.get("page", -1)
            if doc_name in judgment.relevant_doc_ids:
                if judgment.relevant_pages:
                    for rel_doc, rel_page in zip(judgment.relevant_doc_ids, judgment.relevant_pages):
                        if doc_name == rel_doc and page == rel_page:
                            return 1.0
                else:
                    return 1.0
        return 0.0

    for r in retrieved[:k]:
        chunk_id = r.get("id", "")
        if chunk_id in judgment.relevant_chunk_ids:
            return 1.0
    return 0.0


def mrr_at_k(retrieved: List[Dict], judgment: RelevanceJudgment, k: int) -> float:
    """MRR@K."""
    for i, r in enumerate(retrieved[:k]):
        doc_name = r.get("doc", "")
        if doc_name in judgment.relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[Dict], judgment: RelevanceJudgment, k: int) -> float:
    """NDCG@K с бинарной релевантностью."""
    def dcg(rels: List[float]) -> float:
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels))

    relevances = []
    for r in retrieved[:k]:
        doc_name = r.get("doc", "")
        rel = 1.0 if doc_name in judgment.relevant_doc_ids else 0.0
        relevances.append(rel)

    num_rel = min(len(judgment.relevant_doc_ids), k)
    ideal = [1.0] * num_rel + [0.0] * (k - num_rel)

    dcg_score = dcg(relevances)
    idcg_score = dcg(ideal)

    if idcg_score < 1e-9:
        return 0.0
    return min(dcg_score / idcg_score, 1.0)


def bootstrap_ci(values: List[float], n_iterations: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap 95% confidence interval."""
    if not values:
        return (0.0, 0.0)

    arr = np.array(values)
    n = len(arr)

    bootstrap_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(arr, size=n, replace=True)
        bootstrap_means.append(sample.mean())

    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (lower, upper)


def evaluate_retrieval(
    retriever_fn: Callable[[str, int], List[Dict]],
    judgments: Dict[str, RelevanceJudgment],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """Оценка с метриками на doc и chunk level + bootstrap CI."""
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"chunk_recall@{k}": [] for k in k_values})
    metrics.update({f"mrr@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    timings: List[float] = []

    for qid, judgment in judgments.items():
        q_start = time.perf_counter()
        results = retriever_fn(judgment.query, max(k_values))
        timings.append(time.perf_counter() - q_start)

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(results, judgment, k))
            metrics[f"chunk_recall@{k}"].append(chunk_recall_at_k(results, judgment, k))
            metrics[f"mrr@{k}"].append(mrr_at_k(results, judgment, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(results, judgment, k))

    # Агрегируем с bootstrap CI
    aggregated = {}
    for metric_name, values in metrics.items():
        if values:
            arr = np.array(values)
            ci_lower, ci_upper = bootstrap_ci(values)
            aggregated[metric_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            }

    if timings:
        arr = np.array(timings, dtype=np.float64)
        total = float(arr.sum())
        aggregated["performance"] = {
            "query_count": int(arr.size),
            "mean_sec": float(arr.mean()),
            "median_sec": float(np.median(arr)),
            "p95_sec": float(np.percentile(arr, 95)),
            "p99_sec": float(np.percentile(arr, 99)),
            "min_sec": float(arr.min()),
            "max_sec": float(arr.max()),
            "total_sec": total,
            "qps": float(arr.size / total) if total > 0 else 0.0,
        }

    return aggregated


# =========================================================================
# LOAD QA DATASET
# =========================================================================

def load_qa_judgments(qa_csv_path: str) -> Dict[str, RelevanceJudgment]:
    """Загрузка QA датасета."""
    judgments = {}

    with open(qa_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            query_id = f"q{i}"

            doc = row.get("evidence_doc") or row.get("doc") or ""
            page_str = row.get("evidence_page") or row.get("page") or ""
            chunk_id = row.get("chunk_id") or ""

            judgment = RelevanceJudgment(
                query_id=query_id,
                query=row["question"],
                relevant_doc_ids=[doc] if doc else [],
                relevant_pages=[int(page_str)] if page_str else None,
                relevant_chunk_ids=[chunk_id] if chunk_id else None,
            )
            judgments[query_id] = judgment

    return judgments


# =========================================================================
# EXPERIMENT RESULTS
# =========================================================================

@dataclass
class ExperimentResult:
    name: str
    description: str
    metrics: Dict[str, Any]
    timing_sec: float
    category: str = "baseline"  # baseline, fusion, reranker, ablation
    errors: List[str] = field(default_factory=list)


def run_experiment(
    name: str,
    description: str,
    retriever_fn: Callable[[str, int], List[Dict]],
    judgments: Dict[str, RelevanceJudgment],
    category: str = "baseline"
) -> ExperimentResult:
    """Запуск одного эксперимента."""
    logger.info(f"Running {name}...")
    t0 = time.time()
    metrics = evaluate_retrieval(retriever_fn, judgments)
    elapsed = time.time() - t0

    r5 = metrics.get("recall@5", {}).get("mean", 0)
    r10 = metrics.get("recall@10", {}).get("mean", 0)
    logger.info(f"  R@5: {r5:.3f}, R@10: {r10:.3f} (in {elapsed:.1f}s)")

    return ExperimentResult(
        name=name,
        description=description,
        metrics=metrics,
        timing_sec=elapsed,
        category=category,
    )


def run_experiment_safe(
    name: str,
    description: str,
    retriever_fn: Callable[[str, int], List[Dict]],
    judgments: Dict[str, RelevanceJudgment],
    category: str = "baseline",
) -> ExperimentResult:
    """Run experiment with error isolation to keep the pipeline going."""
    try:
        return run_experiment(name, description, retriever_fn, judgments, category=category)
    except Exception as exc:
        logger.error(f"{name} failed: {exc}")
        return ExperimentResult(
            name=name,
            description=description,
            metrics={},
            timing_sec=0.0,
            category=category,
            errors=[str(exc)],
        )


# =========================================================================
# LATEX TABLE GENERATION
# =========================================================================

def generate_latex_main_table(results: List[ExperimentResult]) -> str:
    """Главная таблица результатов."""
    lines = [
        r"% Main Results Table - Extended SOTA Experiments",
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Retrieval Performance Comparison: Baseline, Fusion Methods, and Reranking}",
        r"\label{tab:extended_results}",
        r"\begin{tabular}{ll|ccc|cc|c}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Method} & \textbf{R@1} & \textbf{R@5} & \textbf{R@10} & \textbf{MRR@5} & \textbf{NDCG@5} & \textbf{Time (s)} \\",
        r"\midrule",
    ]

    best_r10 = max(r.metrics.get("recall@10", {}).get("mean", 0) for r in results)

    current_category = None
    for r in results:
        m = r.metrics
        r1 = m.get("recall@1", {}).get("mean", 0)
        r5 = m.get("recall@5", {}).get("mean", 0)
        r10 = m.get("recall@10", {}).get("mean", 0)
        mrr5 = m.get("mrr@5", {}).get("mean", 0)
        ndcg5 = m.get("ndcg@5", {}).get("mean", 0)

        # Разделитель между категориями
        if current_category and current_category != r.category:
            lines.append(r"\midrule")
        current_category = r.category

        cat_display = r.category.capitalize() if r.category != current_category else ""

        # Bold для лучшего результата
        if abs(r10 - best_r10) < 0.001:
            lines.append(
                f"{cat_display} & \\textbf{{{r.name}}} & \\textbf{{{r1:.3f}}} & \\textbf{{{r5:.3f}}} & \\textbf{{{r10:.3f}}} & \\textbf{{{mrr5:.3f}}} & \\textbf{{{ndcg5:.3f}}} & {r.timing_sec:.1f} \\\\"
            )
        else:
            lines.append(
                f"{cat_display} & {r.name} & {r1:.3f} & {r5:.3f} & {r10:.3f} & {mrr5:.3f} & {ndcg5:.3f} & {r.timing_sec:.1f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Cross-lingual QA: 201 Russian questions over English microcontroller datasheets.",
        r"\item R@k = Recall@k, MRR = Mean Reciprocal Rank, NDCG = Normalized DCG.",
        r"\item RRF = Reciprocal Rank Fusion (k=60), CombSUM/MNZ = Fox \& Shaw (1994).",
        r"\end{tablenotes}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


def generate_latex_alpha_table(results: List[ExperimentResult]) -> str:
    """Таблица ablation study по alpha."""
    alpha_results = [r for r in results if r.category == "ablation"]
    if not alpha_results:
        return ""

    lines = [
        r"% Alpha Ablation Study",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of Hybrid Parameter $\alpha$ on Retrieval Performance}",
        r"\label{tab:alpha_ablation}",
        r"\begin{tabular}{c|ccc|cc}",
        r"\toprule",
        r"$\alpha$ & \textbf{R@1} & \textbf{R@5} & \textbf{R@10} & \textbf{MRR@5} & \textbf{NDCG@5} \\",
        r"\midrule",
    ]

    best_r10 = max(r.metrics.get("recall@10", {}).get("mean", 0) for r in alpha_results)

    for r in sorted(alpha_results, key=lambda x: x.name):
        m = r.metrics
        r1 = m.get("recall@1", {}).get("mean", 0)
        r5 = m.get("recall@5", {}).get("mean", 0)
        r10 = m.get("recall@10", {}).get("mean", 0)
        mrr5 = m.get("mrr@5", {}).get("mean", 0)
        ndcg5 = m.get("ndcg@5", {}).get("mean", 0)

        alpha_val = r.name.replace("α=", "")

        if abs(r10 - best_r10) < 0.001:
            lines.append(
                f"\\textbf{{{alpha_val}}} & \\textbf{{{r1:.3f}}} & \\textbf{{{r5:.3f}}} & \\textbf{{{r10:.3f}}} & \\textbf{{{mrr5:.3f}}} & \\textbf{{{ndcg5:.3f}}} \\\\"
            )
        else:
            lines.append(
                f"{alpha_val} & {r1:.3f} & {r5:.3f} & {r10:.3f} & {mrr5:.3f} & {ndcg5:.3f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item $\alpha = 0$: pure BM25, $\alpha = 1$: pure Dense.",
        r"\item Hybrid score: $score = \alpha \cdot dense_{norm} + (1-\alpha) \cdot bm25_{norm}$.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_ci_table(results: List[ExperimentResult]) -> str:
    """Таблица с 95% CI для основных методов."""
    main_results = [r for r in results if r.category in ["baseline", "fusion"]]

    lines = [
        r"% Confidence Intervals Table",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Retrieval Metrics with 95\% Bootstrap Confidence Intervals}",
        r"\label{tab:ci_results}",
        r"\begin{tabular}{l|cc|cc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{R@5 (95\% CI)} & \textbf{R@10 (95\% CI)} & \textbf{MRR@5 (95\% CI)} & \textbf{NDCG@5 (95\% CI)} \\",
        r"\midrule",
    ]

    for r in main_results:
        m = r.metrics

        def fmt_ci(metric_name):
            v = m.get(metric_name, {})
            mean = v.get("mean", 0)
            ci_l = v.get("ci_lower", 0)
            ci_u = v.get("ci_upper", 0)
            return f"{mean:.3f} [{ci_l:.3f}, {ci_u:.3f}]"

        lines.append(
            f"{r.name} & {fmt_ci('recall@5')} & {fmt_ci('recall@10')} & {fmt_ci('mrr@5')} & {fmt_ci('ndcg@5')} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_perf_table(results: List[ExperimentResult]) -> str:
    """Таблица производительности (latency/qps). Adds a lightweight runtime view."""
    lines = [
        r"% Performance Table",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Retrieval Runtime Performance}",
        r"\label{tab:perf_results}",
        r"\begin{tabular}{l|ccc|c}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Mean (s)} & \textbf{P50 (s)} & \textbf{P95 (s)} & \textbf{QPS} \\",
        r"\midrule",
    ]

    for r in results:
        perf = r.metrics.get("performance", {})
        if not perf:
            continue
        mean_s = perf.get("mean_sec", 0.0)
        p50_s = perf.get("median_sec", 0.0)
        p95_s = perf.get("p95_sec", 0.0)
        qps = perf.get("qps", 0.0)
        lines.append(f"{r.name} & {mean_s:.4f} & {p50_s:.4f} & {p95_s:.4f} & {qps:.2f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =========================================================================
# MAIN
# =========================================================================

def main():
    # Пути
    qa_csv = "data/benchmark/qa.csv"
    index_root = "data/index"
    corpus_jsonl = os.path.join(index_root, "corpus.jsonl")
    bm25_index = os.path.join(index_root, "bm25")
    output_dir = Path(os.getenv("OUTPUT_DIR", "results/extended_sota_experiments"))
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dense_models = [
        ("Dense", "multilingual-minilm"),
        ("Dense (E5-base)", "multilingual-e5-base"),
        ("Dense (E5-large)", "multilingual-e5-large"),
        ("Dense (BGE-M3)", "bge-m3"),
    ]
    dense_label_map = {model: label for label, model in baseline_dense_models}
    primary_dense_model = "multilingual-minilm"
    fusion_all_dense = os.getenv("FUSION_ALL_DENSE", "1").lower() not in {"0", "false", "no"}
    embed_device = resolve_embed_device()
    embed_batch_size = resolve_embed_batch_size()
    splade_model_id = os.getenv("SPLADE_MODEL_ID", "naver/splade-v3")
    splade_batch_size = resolve_splade_batch_size()
    splade_max_length = resolve_splade_max_length()
    unicoil_model_id = os.getenv("UNICOIL_MODEL_ID", "castorini/unicoil-msmarco-passage")
    unicoil_batch_size = resolve_unicoil_batch_size()
    unicoil_max_length = resolve_unicoil_max_length()
    unicoil_top_k = resolve_unicoil_top_k()
    rerank_device = resolve_rerank_device()
    rerank_batch_size = resolve_rerank_batch_size()

    logger.info("=" * 70)
    logger.info("EXTENDED SOTA RETRIEVAL EXPERIMENTS")
    logger.info("=" * 70)

    # Загружаем данные
    logger.info(f"Loading QA dataset: {qa_csv}")
    judgments = load_qa_judgments(qa_csv)
    logger.info(f"Loaded {len(judgments)} questions")

    results: List[ExperimentResult] = []

    # =====================================================================
    # PHASE 1: BASELINES
    # =====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: BASELINES")
    logger.info("=" * 50)

    # BM25 Baseline
    results.append(run_experiment_safe(
        "BM25",
        "BM25 with cross-lingual keyword extraction (Whoosh)",
        lambda q, k: search_bm25(bm25_index, q, k),
        judgments,
        category="baseline"
    ))

    # SPLADE (neural sparse)
    try:
        splade_index_dir = ensure_splade_index(
            corpus_jsonl,
            index_root,
            splade_model_id,
            device=embed_device,
            batch_size=splade_batch_size,
            max_length=splade_max_length,
            top_k=128,
        )
        splade_retriever = SpladeRetriever(
            splade_index_dir,
            splade_model_id,
            device=embed_device,
            max_length=splade_max_length,
        )
        results.append(run_experiment_safe(
            "SPLADE",
            f"SPLADE sparse retrieval ({splade_model_id})",
            lambda q, k, sr=splade_retriever: sr.search(q, k),
            judgments,
            category="baseline"
        ))
    except Exception as exc:
        logger.error(f"SPLADE setup failed: {exc}")
        results.append(ExperimentResult(
            name="SPLADE",
            description=f"SPLADE sparse retrieval ({splade_model_id})",
            metrics={},
            timing_sec=0.0,
            category="baseline",
            errors=[str(exc)],
        ))

    # UniCOIL (neural sparse)
    try:
        unicoil_index_dir = ensure_unicoil_index(
            corpus_jsonl,
            index_root,
            unicoil_model_id,
            device=embed_device,
            batch_size=unicoil_batch_size,
            max_length=unicoil_max_length,
            top_k=unicoil_top_k,
        )
        unicoil_retriever = UniCOILRetriever(
            unicoil_index_dir,
            unicoil_model_id,
            device=embed_device,
            max_length=unicoil_max_length,
            top_k=unicoil_top_k,
        )
        results.append(run_experiment_safe(
            "UniCOIL",
            f"UniCOIL sparse retrieval ({unicoil_model_id})",
            lambda q, k, ur=unicoil_retriever: ur.search(q, k),
            judgments,
            category="baseline"
        ))
    except Exception as exc:
        logger.error(f"UniCOIL setup failed: {exc}")
        results.append(ExperimentResult(
            name="UniCOIL",
            description=f"UniCOIL sparse retrieval ({unicoil_model_id})",
            metrics={},
            timing_sec=0.0,
            category="baseline",
            errors=[str(exc)],
        ))

    # Dense Baselines (SOTA alternatives)
    dense_retrievers: Dict[str, DenseRetriever] = {}
    for label, model_name in baseline_dense_models:
        try:
            index_dir = ensure_faiss_index(
                corpus_jsonl,
                index_root,
                model_name,
                device=embed_device,
                batch_size=embed_batch_size,
            )
            dense_retriever = DenseRetriever(index_dir, model_name, device=embed_device)
            dense_retrievers[model_name] = dense_retriever
            model_id = resolve_model_id(model_name)

            results.append(run_experiment_safe(
                label,
                f"Dense retrieval ({model_id})",
                lambda q, k, dr=dense_retriever: dr.search(q, k),
                judgments,
                category="baseline"
            ))
        except Exception as exc:
            logger.error(f"Dense {model_name} failed: {exc}")
            results.append(ExperimentResult(
                name=label,
                description=f"Dense retrieval ({model_name})",
                metrics={},
                timing_sec=0.0,
                category="baseline",
                errors=[str(exc)],
            ))

    dense_retriever = dense_retrievers.get(primary_dense_model)
    if dense_retriever is None and dense_retrievers:
        primary_dense_model = next(iter(dense_retrievers))
        dense_retriever = dense_retrievers[primary_dense_model]
        logger.warning(f"Primary dense model missing; falling back to {primary_dense_model}")

    # =====================================================================
    # PHASE 2: FUSION METHODS
    # =====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: FUSION METHODS")
    logger.info("=" * 50)

    if dense_retrievers:
        fusion_models: List[Tuple[str, DenseRetriever]] = (
            list(dense_retrievers.items())
            if fusion_all_dense
            else [(primary_dense_model, dense_retrievers[primary_dense_model])]
        )
        multi_fusion = len(fusion_models) > 1

        for model_name, fusion_dense in fusion_models:
            label = dense_label_map.get(model_name, model_name)
            suffix = f" ({label})" if multi_fusion else ""

            # Hybrid Weighted (best alpha from previous experiments)
            def hybrid_weighted_fn(q, k, alpha=0.5, dr=fusion_dense):
                bm25_res = search_bm25(bm25_index, q, k * 2)
                dense_res = dr.search(q, k * 2)
                return hybrid_weighted(bm25_res, dense_res, alpha=alpha, k=k)

            results.append(run_experiment_safe(
                f"Hybrid (α=0.5){suffix}",
                f"Weighted Hybrid BM25+{label} with MinMax normalization",
                lambda q, k, dr=fusion_dense: hybrid_weighted_fn(q, k, alpha=0.5, dr=dr),
                judgments,
                category="fusion"
            ))

            # RRF (Reciprocal Rank Fusion)
            def rrf_fn(q, k, dr=fusion_dense):
                bm25_res = search_bm25(bm25_index, q, k * 3)
                dense_res = dr.search(q, k * 3)
                return reciprocal_rank_fusion([bm25_res, dense_res], k_rrf=60, k=k)

            results.append(run_experiment_safe(
                f"RRF (k=60){suffix}",
                "Reciprocal Rank Fusion (Cormack et al. 2009)",
                rrf_fn,
                judgments,
                category="fusion"
            ))

            # CombSUM
            def combsum_fn(q, k, dr=fusion_dense):
                bm25_res = search_bm25(bm25_index, q, k * 2)
                dense_res = dr.search(q, k * 2)
                return combsum_fusion(bm25_res, dense_res, k=k)

            results.append(run_experiment_safe(
                f"CombSUM{suffix}",
                "CombSUM Fusion (Fox & Shaw 1994)",
                combsum_fn,
                judgments,
                category="fusion"
            ))

            # CombMNZ
            def combmnz_fn(q, k, dr=fusion_dense):
                bm25_res = search_bm25(bm25_index, q, k * 2)
                dense_res = dr.search(q, k * 2)
                return combmnz_fusion(bm25_res, dense_res, k=k)

            results.append(run_experiment_safe(
                f"CombMNZ{suffix}",
                "CombMNZ Fusion (Fox & Shaw 1994)",
                combmnz_fn,
                judgments,
                category="fusion"
            ))
    else:
        logger.warning("No dense retrievers available; skipping fusion methods.")

    # =====================================================================
    # PHASE 3: ALPHA ABLATION (fine-grained grid)
    # =====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: ALPHA ABLATION (0.1 - 0.9)")
    logger.info("=" * 50)

    if dense_retriever is not None:
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            def hybrid_alpha_fn(q, k, a=alpha):
                bm25_res = search_bm25(bm25_index, q, k * 2)
                dense_res = dense_retriever.search(q, k * 2)
                return hybrid_weighted(bm25_res, dense_res, alpha=a, k=k)

            results.append(run_experiment_safe(
                f"α={alpha}",
                f"Hybrid with alpha={alpha}",
                hybrid_alpha_fn,
                judgments,
                category="ablation"
            ))
    else:
        logger.warning("No dense retriever available; skipping alpha ablation.")

    # =====================================================================
    # PHASE 4: RERANKERS
    # =====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: RERANKERS")
    logger.info("=" * 50)

    try:
        # ms-marco MiniLM Reranker
        reranker_msmarco = CrossEncoderReranker(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=rerank_device,
            batch_size=rerank_batch_size,
        )

        def bm25_rerank_msmarco(q, k):
            candidates = search_bm25(bm25_index, q, k * 3)
            return reranker_msmarco.rerank(q, candidates, top_k=k)

        results.append(run_experiment_safe(
            "BM25+Rerank (ms-marco)",
            "BM25 with ms-marco-MiniLM-L-6-v2 reranking",
            bm25_rerank_msmarco,
            judgments,
            category="reranker"
        ))

        if dense_retriever is not None:
            def hybrid_rerank_msmarco(q, k):
                bm25_res = search_bm25(bm25_index, q, k * 3)
                dense_res = dense_retriever.search(q, k * 3)
                candidates = hybrid_weighted(bm25_res, dense_res, alpha=0.5, k=k * 3)
                return reranker_msmarco.rerank(q, candidates, top_k=k)

            results.append(run_experiment_safe(
                "Hybrid+Rerank (ms-marco)",
                "Hybrid (α=0.5) with ms-marco-MiniLM-L-6-v2 reranking",
                hybrid_rerank_msmarco,
                judgments,
                category="reranker"
            ))

            def rrf_rerank_msmarco(q, k):
                bm25_res = search_bm25(bm25_index, q, k * 3)
                dense_res = dense_retriever.search(q, k * 3)
                candidates = reciprocal_rank_fusion([bm25_res, dense_res], k_rrf=60, k=k * 3)
                return reranker_msmarco.rerank(q, candidates, top_k=k)

            results.append(run_experiment_safe(
                "RRF+Rerank (ms-marco)",
                "RRF with ms-marco-MiniLM-L-6-v2 reranking",
                rrf_rerank_msmarco,
                judgments,
                category="reranker"
            ))
        else:
            logger.warning("No dense retriever available; skipping hybrid/RRF ms-marco rerank.")

    except Exception as e:
        logger.error(f"ms-marco reranker failed: {e}")

    # BGE Reranker (optional - may not be installed)
    try:
        reranker_bge = CrossEncoderReranker(
            "BAAI/bge-reranker-base",
            device=rerank_device,
            batch_size=rerank_batch_size,
        )

        def bm25_rerank_bge(q, k):
            candidates = search_bm25(bm25_index, q, k * 3)
            return reranker_bge.rerank(q, candidates, top_k=k)

        results.append(run_experiment_safe(
            "BM25+Rerank (BGE)",
            "BM25 with BAAI/bge-reranker-base reranking",
            bm25_rerank_bge,
            judgments,
            category="reranker"
        ))

        if dense_retriever is not None:
            def hybrid_rerank_bge(q, k):
                bm25_res = search_bm25(bm25_index, q, k * 3)
                dense_res = dense_retriever.search(q, k * 3)
                candidates = hybrid_weighted(bm25_res, dense_res, alpha=0.5, k=k * 3)
                return reranker_bge.rerank(q, candidates, top_k=k)

            results.append(run_experiment_safe(
                "Hybrid+Rerank (BGE)",
                "Hybrid (α=0.5) with BAAI/bge-reranker-base reranking",
                hybrid_rerank_bge,
                judgments,
                category="reranker"
            ))
        else:
            logger.warning("No dense retriever available; skipping hybrid BGE rerank.")

    except Exception as e:
        logger.warning(f"BGE reranker not available: {e}")

    # =====================================================================
    # PHASE 5: SAVE RESULTS
    # =====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 5: SAVING RESULTS")
    logger.info("=" * 50)

    # JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {json_path}")

    # Markdown
    md_lines = ["# Extended SOTA Retrieval Experiments\n"]
    md_lines.append(f"QA Dataset: {len(judgments)} questions (RU→EN cross-lingual)\n")

    for category in ["baseline", "fusion", "ablation", "reranker"]:
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue

        md_lines.append(f"\n## {category.upper()}\n")
        md_lines.append("| Method | R@1 | R@5 | R@10 | MRR@5 | NDCG@5 | Time |")
        md_lines.append("|--------|-----|-----|------|-------|--------|------|")

        for r in cat_results:
            m = r.metrics
            md_lines.append(
                f"| {r.name} | {m.get('recall@1',{}).get('mean',0):.3f} | "
                f"{m.get('recall@5',{}).get('mean',0):.3f} | "
                f"{m.get('recall@10',{}).get('mean',0):.3f} | "
                f"{m.get('mrr@5',{}).get('mean',0):.3f} | "
                f"{m.get('ndcg@5',{}).get('mean',0):.3f} | "
                f"{r.timing_sec:.1f}s |"
            )

    # Performance summary
    md_lines.append("\n## PERFORMANCE\n")
    md_lines.append("| Method | Mean (s) | P50 (s) | P95 (s) | QPS | Total (s) |")
    md_lines.append("|--------|----------|---------|---------|-----|-----------|")
    for r in results:
        perf = r.metrics.get("performance", {})
        if not perf:
            continue
        md_lines.append(
            f"| {r.name} | {perf.get('mean_sec', 0):.4f} | "
            f"{perf.get('median_sec', 0):.4f} | "
            f"{perf.get('p95_sec', 0):.4f} | "
            f"{perf.get('qps', 0):.2f} | "
            f"{perf.get('total_sec', 0):.1f} |"
        )

    md_path = output_dir / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved: {md_path}")

    # LaTeX
    tex_path = output_dir / "paper_tables.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(generate_latex_main_table(results))
        f.write("\n\n")
        f.write(generate_latex_alpha_table(results))
        f.write("\n\n")
        f.write(generate_latex_ci_table(results))
        f.write("\n\n")
        f.write(generate_latex_perf_table(results))
    logger.info(f"Saved: {tex_path}")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)

    # Группируем по категориям
    for category in ["baseline", "fusion", "reranker"]:
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue

        logger.info(f"\n{category.upper()}:")
        for r in cat_results:
            m = r.metrics
            r10 = m.get("recall@10", {}).get("mean", 0)
            ci_l = m.get("recall@10", {}).get("ci_lower", 0)
            ci_u = m.get("recall@10", {}).get("ci_upper", 0)
            logger.info(f"  {r.name:25s}: R@10 = {r10:.3f} [{ci_l:.3f}, {ci_u:.3f}]")

    # Лучший alpha
    alpha_results = [r for r in results if r.category == "ablation"]
    if alpha_results:
        best_alpha = max(alpha_results, key=lambda x: x.metrics.get("recall@10", {}).get("mean", 0))
        logger.info(f"\nBest alpha: {best_alpha.name} with R@10 = {best_alpha.metrics.get('recall@10', {}).get('mean', 0):.3f}")

    # Overall best
    overall_best = max(results, key=lambda x: x.metrics.get("recall@10", {}).get("mean", 0))
    logger.info(f"\nOVERALL BEST: {overall_best.name} with R@10 = {overall_best.metrics.get('recall@10', {}).get('mean', 0):.3f}")

    logger.info("\n" + "=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("EXPERIMENTS COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
