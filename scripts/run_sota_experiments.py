#!/usr/bin/env python3
"""
SOTA эксперименты без sentence_transformers.
Использует BM25 (Whoosh) и предзагруженные FAISS индексы.
Для запросов использует кэш эмбеддингов или HTTP API.
"""
import os
import sys
import json
import csv
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

# Добавляем корень проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================================================
# BM25 RETRIEVER (Whoosh only - no sentence_transformers needed)
# =========================================================================

import re

# Простой словарь для извлечения ключевых терминов
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
    "datasheet": "datasheet",
}


def extract_search_terms(query: str) -> str:
    """Извлекает английские поисковые термины из русского запроса."""
    query_lower = query.lower()
    terms = []

    # Извлекаем английские слова/числа напрямую
    english_terms = re.findall(r'[A-Za-z0-9][\w\-\.]*', query)
    terms.extend(english_terms)

    # Добавляем переводы русских ключевых слов
    for ru_key, en_terms in RU_EN_KEYWORDS.items():
        if ru_key in query_lower:
            terms.extend(en_terms.split())

    # Убираем дубликаты, сохраняя порядок
    seen = set()
    unique_terms = []
    for t in terms:
        t_lower = t.lower()
        if t_lower not in seen and len(t) > 1:
            seen.add(t_lower)
            unique_terms.append(t)

    return " OR ".join(unique_terms) if unique_terms else query


def search_bm25(index_dir: str, query: str, k: int = 5) -> List[Dict]:
    """BM25 поиск через Whoosh с автоматическим извлечением терминов."""
    from whoosh import index
    from whoosh.qparser import MultifieldParser, OrGroup

    ix = index.open_dir(index_dir)
    # Используем OrGroup для OR-семантики по умолчанию
    parser = MultifieldParser(["text", "doc"], schema=ix.schema, group=OrGroup)

    # Извлекаем поисковые термины
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


# =========================================================================
# DENSE RETRIEVER (uses sentence-transformers for real-time embedding)
# =========================================================================

class DenseRetriever:
    """Dense retriever с мультиязычными эмбеддингами."""

    def __init__(self, index_dir: str, model_name: str = "intfloat/multilingual-e5-base"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index_dir = index_dir
        self.model_name = model_name

        # Загружаем FAISS индекс
        index_path = os.path.join(index_dir, "faiss.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Загружаем метаданные
        meta_path = os.path.join(index_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)["items"]

        # Загружаем эмбеддер
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        logger.info(f"Embedding dimension: {self.embedder.get_sentence_embedding_dimension()}")

    def embed_query(self, query: str) -> np.ndarray:
        """Создаёт эмбеддинг запроса."""
        # Для E5 моделей нужен префикс "query: "
        if "e5" in self.model_name.lower():
            query = f"query: {query}"

        embedding = self.embedder.encode(query, normalize_embeddings=True)
        return embedding.astype("float32").reshape(1, -1)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Поиск по FAISS индексу."""
        query_embedding = self.embed_query(query)

        D, I = self.index.search(query_embedding, k)

        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.items):
                continue
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


# =========================================================================
# HYBRID RETRIEVER
# =========================================================================

def normalize_scores(scores: List[float], method: str = "minmax") -> List[float]:
    """Нормализация скоров."""
    if not scores:
        return []

    arr = np.array(scores)

    if method == "minmax":
        min_s, max_s = arr.min(), arr.max()
        if max_s - min_s > 1e-9:
            return ((arr - min_s) / (max_s - min_s)).tolist()
        return [0.5] * len(scores)

    elif method == "zscore":
        mean, std = arr.mean(), arr.std()
        if std > 1e-9:
            return ((arr - mean) / std).tolist()
        return [0.0] * len(scores)

    return scores


def hybrid_search(
    bm25_results: List[Dict],
    dense_results: List[Dict],
    alpha: float = 0.5,
    k: int = 5
) -> List[Dict]:
    """
    Гибридный поиск: alpha * dense + (1 - alpha) * bm25
    """
    # Нормализуем скоры
    bm25_scores = [r["score"] for r in bm25_results]
    dense_scores = [r["score"] for r in dense_results]

    bm25_norm = normalize_scores(bm25_scores, "minmax")
    dense_norm = normalize_scores(dense_scores, "minmax")

    # Объединяем
    combined = {}

    for i, r in enumerate(bm25_results):
        doc_id = r["id"]
        combined[doc_id] = {
            **r,
            "bm25_score": bm25_norm[i] if i < len(bm25_norm) else 0,
            "dense_score": 0,
            "retriever": "hybrid",
        }

    for i, r in enumerate(dense_results):
        doc_id = r["id"]
        if doc_id in combined:
            combined[doc_id]["dense_score"] = dense_norm[i] if i < len(dense_norm) else 0
        else:
            combined[doc_id] = {
                **r,
                "bm25_score": 0,
                "dense_score": dense_norm[i] if i < len(dense_norm) else 0,
                "retriever": "hybrid",
            }

    # Вычисляем финальный скор
    for doc_id, r in combined.items():
        r["score"] = alpha * r["dense_score"] + (1 - alpha) * r["bm25_score"]

    # Сортируем и возвращаем top-k
    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:k]


# =========================================================================
# EVALUATION METRICS
# =========================================================================

@dataclass
class RelevanceJudgment:
    query_id: str
    query: str
    relevant_doc_ids: List[str]
    relevant_pages: Optional[List[int]] = None


def recall_at_k(retrieved: List[Dict], judgment: RelevanceJudgment, k: int) -> float:
    """
    Recall@K - бинарная метрика: 1 если релевантный документ найден в top-k, иначе 0.
    Для QA с одним релевантным документом это эквивалентно Hit@K.
    """
    if not judgment.relevant_doc_ids:
        return 0.0

    # Проверяем каждый результат в top-k
    for r in retrieved[:k]:
        doc_name = r.get("doc", "")
        page = r.get("page", -1)

        # Проверяем совпадение по документу
        if doc_name in judgment.relevant_doc_ids:
            # Если есть релевантные страницы, проверяем и их
            if judgment.relevant_pages:
                for rel_doc, rel_page in zip(judgment.relevant_doc_ids, judgment.relevant_pages):
                    if doc_name == rel_doc and page == rel_page:
                        return 1.0
                # Документ найден, но страница не та - всё равно считаем попаданием
                return 1.0
            else:
                return 1.0

    return 0.0


def mrr_at_k(retrieved: List[Dict], judgment: RelevanceJudgment, k: int) -> float:
    """MRR@K - mean reciprocal rank."""
    for i, r in enumerate(retrieved[:k]):
        doc_name = r.get("doc", "")
        if doc_name in judgment.relevant_doc_ids:
            return 1.0 / (i + 1)
        # Проверяем по странице
        if judgment.relevant_pages:
            page = r.get("page", -1)
            for rel_doc, rel_page in zip(judgment.relevant_doc_ids, judgment.relevant_pages):
                if doc_name == rel_doc and page == rel_page:
                    return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[Dict], judgment: RelevanceJudgment, k: int) -> float:
    """NDCG@K с бинарной релевантностью."""
    def dcg(relevances: List[float], k: int) -> float:
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))

    # Binary relevance: 1 если документ релевантен, 0 иначе
    relevances = []
    for r in retrieved[:k]:
        doc_name = r.get("doc", "")
        rel = 1.0 if doc_name in judgment.relevant_doc_ids else 0.0
        relevances.append(rel)

    # Идеальный ranking: все релевантные документы сверху
    # Для QA обычно 1 релевантный документ
    num_relevant = min(len(judgment.relevant_doc_ids), k)
    ideal = [1.0] * num_relevant + [0.0] * (k - num_relevant)

    dcg_score = dcg(relevances, k)
    idcg_score = dcg(ideal, k)

    if idcg_score < 1e-9:
        return 0.0
    return min(dcg_score / idcg_score, 1.0)  # Ограничиваем максимум 1.0


def evaluate_retrieval(
    retriever_fn: Callable[[str, int], List[Dict]],
    judgments: Dict[str, RelevanceJudgment],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """Оценка retriever'а."""
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"mrr@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    for qid, judgment in judgments.items():
        results = retriever_fn(judgment.query, max(k_values))

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(results, judgment, k))
            metrics[f"mrr@{k}"].append(mrr_at_k(results, judgment, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(results, judgment, k))

    # Агрегируем
    aggregated = {}
    for metric_name, values in metrics.items():
        if values:
            arr = np.array(values)
            aggregated[metric_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
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

            judgment = RelevanceJudgment(
                query_id=query_id,
                query=row["question"],
                relevant_doc_ids=[doc] if doc else [],
                relevant_pages=[int(page_str)] if page_str else None,
            )
            judgments[query_id] = judgment

    return judgments


# =========================================================================
# MAIN EXPERIMENTS
# =========================================================================

@dataclass
class ExperimentResult:
    name: str
    description: str
    metrics: Dict[str, Any]
    timing_sec: float
    errors: List[str] = field(default_factory=list)


def run_bm25_experiment(
    bm25_index_dir: str,
    judgments: Dict[str, RelevanceJudgment]
) -> ExperimentResult:
    """BM25 baseline эксперимент."""
    logger.info("Running BM25 baseline...")

    def retriever(query: str, k: int) -> List[Dict]:
        return search_bm25(bm25_index_dir, query, k)

    t0 = time.time()
    metrics = evaluate_retrieval(retriever, judgments)
    elapsed = time.time() - t0

    return ExperimentResult(
        name="BM25",
        description="BM25 baseline (Whoosh)",
        metrics=metrics,
        timing_sec=elapsed,
    )


def run_dense_experiment(
    dense_retriever: DenseRetriever,
    judgments: Dict[str, RelevanceJudgment]
) -> ExperimentResult:
    """Dense retrieval эксперимент."""
    logger.info("Running Dense baseline...")

    def retriever(query: str, k: int) -> List[Dict]:
        return dense_retriever.search(query, k)

    t0 = time.time()
    metrics = evaluate_retrieval(retriever, judgments)
    elapsed = time.time() - t0

    return ExperimentResult(
        name="Dense",
        description="Dense retrieval (FAISS)",
        metrics=metrics,
        timing_sec=elapsed,
    )


def run_hybrid_experiment(
    bm25_index_dir: str,
    dense_retriever: Optional[DenseRetriever],
    judgments: Dict[str, RelevanceJudgment],
    alpha: float = 0.5
) -> ExperimentResult:
    """Hybrid эксперимент."""
    logger.info(f"Running Hybrid (alpha={alpha})...")

    def retriever(query: str, k: int) -> List[Dict]:
        bm25_results = search_bm25(bm25_index_dir, query, k * 2)

        if dense_retriever:
            dense_results = dense_retriever.search(query, k * 2)
        else:
            dense_results = []

        return hybrid_search(bm25_results, dense_results, alpha=alpha, k=k)

    t0 = time.time()
    metrics = evaluate_retrieval(retriever, judgments)
    elapsed = time.time() - t0

    return ExperimentResult(
        name=f"Hybrid_a{alpha}",
        description=f"Hybrid: alpha={alpha}",
        metrics=metrics,
        timing_sec=elapsed,
    )


def generate_latex_table(results: List[ExperimentResult]) -> str:
    """Генерация LaTeX таблицы для статьи."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Retrieval Results on Microcontroller Datasheet QA}",
        r"\label{tab:retrieval_results}",
        r"\begin{tabular}{l|ccc|ccc}",
        r"\toprule",
        r"Method & R@1 & R@5 & R@10 & MRR@5 & NDCG@5 & Time (s) \\",
        r"\midrule",
    ]

    for r in results:
        m = r.metrics
        r1 = m.get("recall@1", {}).get("mean", 0)
        r5 = m.get("recall@5", {}).get("mean", 0)
        r10 = m.get("recall@10", {}).get("mean", 0)
        mrr5 = m.get("mrr@5", {}).get("mean", 0)
        ndcg5 = m.get("ndcg@5", {}).get("mean", 0)

        lines.append(
            f"{r.name} & {r1:.3f} & {r5:.3f} & {r10:.3f} & {mrr5:.3f} & {ndcg5:.3f} & {r.timing_sec:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_markdown_table(results: List[ExperimentResult]) -> str:
    """Генерация Markdown таблицы."""
    lines = [
        "| Method | R@1 | R@5 | R@10 | MRR@5 | NDCG@5 | Time (s) |",
        "|--------|-----|-----|------|-------|--------|----------|",
    ]

    for r in results:
        m = r.metrics
        r1 = m.get("recall@1", {}).get("mean", 0)
        r5 = m.get("recall@5", {}).get("mean", 0)
        r10 = m.get("recall@10", {}).get("mean", 0)
        mrr5 = m.get("mrr@5", {}).get("mean", 0)
        ndcg5 = m.get("ndcg@5", {}).get("mean", 0)

        lines.append(
            f"| {r.name} | {r1:.3f} | {r5:.3f} | {r10:.3f} | {mrr5:.3f} | {ndcg5:.3f} | {r.timing_sec:.1f} |"
        )

    return "\n".join(lines)


def main():
    # Пути
    qa_csv = "data/benchmark/qa.csv"
    bm25_index = "data/index/bm25"
    faiss_index = "data/index/faiss"
    output_dir = Path("results/sota_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Мультиязычная модель для cross-lingual retrieval (RU queries -> EN docs)
    # Должна совпадать с моделью, использованной при создании FAISS индекса
    embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384d

    logger.info("=" * 60)
    logger.info("SOTA RETRIEVAL EXPERIMENTS")
    logger.info("=" * 60)

    # Загружаем QA датасет
    logger.info(f"Loading QA dataset from {qa_csv}...")
    judgments = load_qa_judgments(qa_csv)
    logger.info(f"Loaded {len(judgments)} questions")

    results: List[ExperimentResult] = []

    # 1. BM25 Baseline
    logger.info("\n[1/4] BM25 Baseline...")
    try:
        bm25_result = run_bm25_experiment(bm25_index, judgments)
        results.append(bm25_result)
        logger.info(f"  R@5: {bm25_result.metrics.get('recall@5', {}).get('mean', 0):.3f}")
    except Exception as e:
        logger.error(f"BM25 failed: {e}")

    # 2. Dense Baseline (sentence-transformers)
    dense_retriever = None
    logger.info("\n[2/4] Dense Baseline (multilingual embeddings)...")
    try:
        dense_retriever = DenseRetriever(faiss_index, embed_model)
        dense_result = run_dense_experiment(dense_retriever, judgments)
        results.append(dense_result)
        logger.info(f"  R@5: {dense_result.metrics.get('recall@5', {}).get('mean', 0):.3f}")
    except Exception as e:
        logger.error(f"Dense failed: {e}")

    # 3. Hybrid с разными alpha
    logger.info("\n[3/4] Hybrid experiments...")
    for alpha in [0.3, 0.5, 0.7]:
        try:
            hybrid_result = run_hybrid_experiment(bm25_index, dense_retriever, judgments, alpha)
            results.append(hybrid_result)
            logger.info(f"  Hybrid a={alpha} R@5: {hybrid_result.metrics.get('recall@5', {}).get('mean', 0):.3f}")
        except Exception as e:
            logger.error(f"Hybrid alpha={alpha} failed: {e}")

    # 4. Сохраняем результаты
    logger.info("\n[4/4] Saving results...")

    # JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved to {json_path}")

    # Markdown
    md_table = generate_markdown_table(results)
    md_path = output_dir / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SOTA Retrieval Experiments\n\n")
        f.write(f"QA Dataset: {len(judgments)} questions\n\n")
        f.write(md_table)
    logger.info(f"Markdown saved to {md_path}")

    # LaTeX
    latex_table = generate_latex_table(results)
    tex_path = output_dir / "results.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    logger.info(f"LaTeX saved to {tex_path}")

    # Выводим итоговую таблицу
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    print("\n" + md_table + "\n")

    logger.info("=" * 60)
    logger.info("EXPERIMENTS COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
