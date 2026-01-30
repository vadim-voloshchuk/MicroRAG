"""
Retrieval Evaluation Harness - Комплексная оценка качества retrieval.

Отвечает на требование пункта 8:
- Recall@k, MRR@k, nDCG@k
- Evidence-in-top-k (есть ли правильный документ/секция)
- Выгрузка в CSV + авто-генерация таблиц

Также пункт 10:
- Bootstrap 95% CI для устойчивой статистики
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Literal, Tuple
import csv
import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RelevanceJudgment:
    """Оценка релевантности для одного запроса."""
    query_id: str
    query: str
    relevant_doc_ids: List[str]  # Список релевантных doc IDs
    relevant_pages: Optional[List[int]] = None
    relevant_chunks: Optional[List[str]] = None  # Список релевантных chunk IDs


@dataclass
class RetrievalResult:
    """Результат поиска для одного запроса."""
    query_id: str
    query: str
    retrieved: List[Dict]  # Список найденных документов с полями id, doc, page, score
    latency_ms: float = 0.0


@dataclass
class MetricsResult:
    """Результаты метрик для одного запроса."""
    query_id: str
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    hit_at_k: Dict[int, bool] = field(default_factory=dict)  # Evidence found?

    # Детализация по уровням
    doc_hit: bool = False       # Найден ли правильный документ
    page_hit: bool = False      # Найдена ли правильная страница
    chunk_hit: bool = False     # Найден ли правильный чанк


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Recall@k - доля релевантных документов, найденных в top-k.

    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|
    """
    if not relevant_ids:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    found = len(retrieved_k & relevant_set)

    return found / len(relevant_set)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Precision@k - доля релевантных среди top-k.

    Precision@k = |relevant ∩ retrieved[:k]| / k
    """
    if k == 0:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    found = len(retrieved_k & relevant_set)

    return found / k


def mrr_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Mean Reciprocal Rank@k - обратная позиция первого релевантного.

    MRR = 1 / rank(first_relevant), или 0 если не найден в top-k
    """
    relevant_set = set(relevant_ids)

    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            return 1.0 / i

    return 0.0


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Discounted Cumulative Gain@k.

    DCG@k = Σ (rel_i / log2(i + 1)) for i in 1..k
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        dcg += rel / np.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain@k.

    nDCG@k = DCG@k / IDCG@k

    Где IDCG - идеальный DCG (все релевантные в начале).
    """
    relevant_set = set(relevant_ids)

    # Relevance scores для retrieved (1 если релевантен, 0 иначе)
    relevances = [1.0 if rid in relevant_set else 0.0 for rid in retrieved_ids[:k]]

    dcg = dcg_at_k(relevances, k)

    # Ideal DCG: все релевантные в начале
    ideal_relevances = [1.0] * min(len(relevant_ids), k)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> bool:
    """
    Hit@k - найден ли хотя бы один релевантный в top-k.
    """
    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_k & relevant_set) > 0


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    values: List[float],
    n_samples: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval.

    Returns:
        Tuple[mean, lower_bound, upper_bound]
    """
    if not values:
        return 0.0, 0.0, 0.0

    values = np.array(values)
    n = len(values)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_samples):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(sample.mean())

    bootstrap_means = np.array(bootstrap_means)

    # Confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return values.mean(), lower, upper


# =============================================================================
# EVALUATION HARNESS
# =============================================================================

class RetrievalEvaluator:
    """
    Комплексный evaluator для retrieval.
    """

    def __init__(
        self,
        k_values: List[int] = [1, 3, 5, 10],
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        levels: List[Literal["chunk", "page", "document"]] = ["chunk", "document"]
    ):
        self.k_values = k_values
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.levels = levels

    def evaluate_single(
        self,
        result: RetrievalResult,
        judgment: RelevanceJudgment
    ) -> MetricsResult:
        """
        Оценивает один запрос.
        """
        metrics = MetricsResult(query_id=result.query_id)

        # Определяем, по какому уровню считаем recall/mrr/ndcg (chunk -> page -> doc)
        if judgment.relevant_chunks:
            relevant_ids = judgment.relevant_chunks
            retrieved_ids = [r["id"] for r in result.retrieved if r.get("id")]
        elif judgment.relevant_pages:
            if judgment.relevant_doc_ids and len(judgment.relevant_doc_ids) == len(judgment.relevant_pages):
                pairs = list(zip(judgment.relevant_doc_ids, judgment.relevant_pages))
            else:
                pairs = [
                    (doc_id, page)
                    for doc_id in judgment.relevant_doc_ids
                    for page in judgment.relevant_pages
                ]
            relevant_ids = [f"{doc_id}::p{page}" for doc_id, page in pairs]
            retrieved_ids = [
                f"{r.get('doc')}::p{r.get('page')}"
                for r in result.retrieved
                if r.get("doc") is not None and r.get("page") is not None
            ]
        else:
            relevant_ids = judgment.relevant_doc_ids
            retrieved_ids = [r.get("doc") for r in result.retrieved if r.get("doc")]

        # Chunk-level metrics
        for k in self.k_values:
            metrics.recall_at_k[k] = recall_at_k(retrieved_ids, relevant_ids, k)
            metrics.mrr_at_k[k] = mrr_at_k(retrieved_ids, relevant_ids, k)
            metrics.ndcg_at_k[k] = ndcg_at_k(retrieved_ids, relevant_ids, k)
            metrics.precision_at_k[k] = precision_at_k(retrieved_ids, relevant_ids, k)
            metrics.hit_at_k[k] = hit_at_k(retrieved_ids, relevant_ids, k)

        # Document-level hit
        retrieved_docs = [r.get("doc") for r in result.retrieved]
        metrics.doc_hit = any(doc in judgment.relevant_doc_ids for doc in retrieved_docs if doc)

        # Page-level hit
        if judgment.relevant_pages:
            retrieved_pages = [(r.get("doc"), r.get("page")) for r in result.retrieved]
            for doc_id in judgment.relevant_doc_ids:
                for page in judgment.relevant_pages:
                    if (doc_id, page) in retrieved_pages:
                        metrics.page_hit = True
                        break

        # Chunk-level hit (только если есть chunk labels)
        metrics.chunk_hit = (
            metrics.hit_at_k.get(self.k_values[-1], False)
            if judgment.relevant_chunks
            else False
        )

        return metrics

    def evaluate_batch(
        self,
        results: List[RetrievalResult],
        judgments: Dict[str, RelevanceJudgment]
    ) -> Dict:
        """
        Оценивает батч запросов и агрегирует метрики.
        """
        all_metrics: List[MetricsResult] = []

        for result in results:
            if result.query_id not in judgments:
                logger.warning(f"No judgment for query {result.query_id}")
                continue

            judgment = judgments[result.query_id]
            metrics = self.evaluate_single(result, judgment)
            all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Агрегация метрик
        aggregated = {
            "n_queries": len(all_metrics),
            "metrics": {},
        }

        for k in self.k_values:
            # Recall@k
            recall_values = [m.recall_at_k[k] for m in all_metrics]
            mean, lower, upper = bootstrap_ci(recall_values, self.bootstrap_samples, self.confidence_level)
            aggregated["metrics"][f"recall@{k}"] = {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            }

            # MRR@k
            mrr_values = [m.mrr_at_k[k] for m in all_metrics]
            mean, lower, upper = bootstrap_ci(mrr_values, self.bootstrap_samples, self.confidence_level)
            aggregated["metrics"][f"mrr@{k}"] = {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            }

            # nDCG@k
            ndcg_values = [m.ndcg_at_k[k] for m in all_metrics]
            mean, lower, upper = bootstrap_ci(ndcg_values, self.bootstrap_samples, self.confidence_level)
            aggregated["metrics"][f"ndcg@{k}"] = {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            }

            # Hit@k (success rate)
            hit_values = [float(m.hit_at_k[k]) for m in all_metrics]
            mean, lower, upper = bootstrap_ci(hit_values, self.bootstrap_samples, self.confidence_level)
            aggregated["metrics"][f"hit@{k}"] = {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            }

        # Level-specific hits
        doc_hits = [float(m.doc_hit) for m in all_metrics]
        page_hits = [float(m.page_hit) for m in all_metrics]
        chunk_hits = [float(m.chunk_hit) for m in all_metrics]

        aggregated["metrics"]["doc_hit_rate"] = {
            "mean": np.mean(doc_hits),
            "ci_lower": bootstrap_ci(doc_hits, self.bootstrap_samples, self.confidence_level)[1],
            "ci_upper": bootstrap_ci(doc_hits, self.bootstrap_samples, self.confidence_level)[2],
        }

        aggregated["metrics"]["page_hit_rate"] = {
            "mean": np.mean(page_hits),
            "ci_lower": bootstrap_ci(page_hits, self.bootstrap_samples, self.confidence_level)[1],
            "ci_upper": bootstrap_ci(page_hits, self.bootstrap_samples, self.confidence_level)[2],
        }

        aggregated["metrics"]["chunk_hit_rate"] = {
            "mean": np.mean(chunk_hits),
            "ci_lower": bootstrap_ci(chunk_hits, self.bootstrap_samples, self.confidence_level)[1],
            "ci_upper": bootstrap_ci(chunk_hits, self.bootstrap_samples, self.confidence_level)[2],
        }

        # Детальные результаты для анализа
        aggregated["detailed"] = [
            {
                "query_id": m.query_id,
                **{f"recall@{k}": m.recall_at_k[k] for k in self.k_values},
                **{f"mrr@{k}": m.mrr_at_k[k] for k in self.k_values},
                "doc_hit": m.doc_hit,
                "page_hit": m.page_hit,
                "chunk_hit": m.chunk_hit,
            }
            for m in all_metrics
        ]

        return aggregated


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_results_csv(results: Dict, output_path: str):
    """Экспортирует результаты в CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_path = path.with_suffix(".summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "ci_lower", "ci_upper"])
        for metric_name, values in results["metrics"].items():
            writer.writerow([
                metric_name,
                f"{values['mean']:.4f}",
                f"{values['ci_lower']:.4f}",
                f"{values['ci_upper']:.4f}",
            ])

    # Detailed CSV
    detailed_path = path.with_suffix(".detailed.csv")
    if results.get("detailed"):
        fieldnames = results["detailed"][0].keys()
        with open(detailed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results["detailed"]:
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})

    logger.info(f"Results exported to {summary_path} and {detailed_path}")


def export_results_jsonl(results: Dict, output_path: str):
    """Экспортирует результаты в JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # Summary as first line
        summary_record = {
            "type": "summary",
            "n_queries": results["n_queries"],
            "metrics": results["metrics"],
        }
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

        # Detailed results
        for detail in results.get("detailed", []):
            record = {"type": "query_result", **detail}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Results exported to {path}")


def generate_latex_table(results: Dict, caption: str = "Retrieval Results") -> str:
    """Генерирует LaTeX таблицу для статьи."""
    metrics = results["metrics"]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & Mean & 95\\% CI Lower & 95\\% CI Upper \\\\",
        "\\midrule",
    ]

    for metric_name, values in metrics.items():
        lines.append(
            f"{metric_name} & {values['mean']:.3f} & {values['ci_lower']:.3f} & {values['ci_upper']:.3f} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_qa_judgments(qa_csv_path: str) -> Dict[str, RelevanceJudgment]:
    """
    Загружает judgments из QA CSV.

    Поддерживаемые колонки:
    - question: вопрос
    - evidence_doc или doc: имя документа
    - evidence_page или page: номер страницы
    - chunk_id (опционально): ID релевантного чанка
    """
    judgments = {}

    with open(qa_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            query_id = f"q{i}"

            # Поддержка разных названий колонок
            doc = row.get("evidence_doc") or row.get("doc") or ""
            page_str = row.get("evidence_page") or row.get("page") or ""

            judgment = RelevanceJudgment(
                query_id=query_id,
                query=row["question"],
                relevant_doc_ids=[doc] if doc else [],
                relevant_pages=[int(page_str)] if page_str else None,
                relevant_chunks=[row.get("chunk_id")] if row.get("chunk_id") else None,
            )
            judgments[query_id] = judgment

    return judgments


def run_retrieval_evaluation(
    retriever_fn: Callable,
    qa_csv_path: str,
    k_values: List[int] = [1, 5, 10],
    output_dir: str = "results/retrieval",
    experiment_name: str = "baseline"
) -> Dict:
    """
    Полный pipeline оценки retrieval.

    Args:
        retriever_fn: Функция retrieval (query, k) -> List[Dict]
        qa_csv_path: Путь к CSV с вопросами и ответами
        k_values: Значения k для метрик
        output_dir: Директория для результатов
        experiment_name: Имя эксперимента

    Returns:
        Dict: Агрегированные результаты
    """
    import time

    # Загружаем judgments
    judgments = load_qa_judgments(qa_csv_path)
    logger.info(f"Loaded {len(judgments)} queries from {qa_csv_path}")

    # Прогоняем retrieval
    results: List[RetrievalResult] = []
    max_k = max(k_values)

    for query_id, judgment in judgments.items():
        t0 = time.time()
        retrieved = retriever_fn(judgment.query, k=max_k)
        latency_ms = (time.time() - t0) * 1000

        results.append(RetrievalResult(
            query_id=query_id,
            query=judgment.query,
            retrieved=retrieved,
            latency_ms=latency_ms,
        ))

    # Оцениваем
    evaluator = RetrievalEvaluator(k_values=k_values)
    aggregated = evaluator.evaluate_batch(results, judgments)

    # Добавляем метаданные
    aggregated["experiment"] = experiment_name
    aggregated["k_values"] = k_values
    aggregated["avg_latency_ms"] = np.mean([r.latency_ms for r in results])

    # Экспортируем
    output_path = Path(output_dir) / f"{experiment_name}"
    export_results_csv(aggregated, str(output_path))
    export_results_jsonl(aggregated, str(output_path.with_suffix(".jsonl")))

    # LaTeX таблица
    latex = generate_latex_table(aggregated, caption=f"Retrieval Results: {experiment_name}")
    latex_path = output_path.with_suffix(".tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info(f"LaTeX table saved to {latex_path}")

    return aggregated
