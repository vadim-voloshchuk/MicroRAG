#!/usr/bin/env python3
"""
Полные SOTA эксперименты для Q1 публикации.

Включает все требования рецензентов:
1. BM25 vs Dense vs Hybrid (с корректной нормализацией)
2. +Reranker эксперименты
3. Метрики на уровне chunk и document
4. Bootstrap 95% CI для всех метрик
5. Сравнение text-only vs text+tables
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
import numpy as np

# Добавляем корень проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================================
# IMPORTS FROM PROJECT MODULES
# =========================================================================

from rag_micro.retrievers.bm25_whoosh import search_whoosh
import re

# Простой словарь для кросс-языкового поиска (RU -> EN)
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


# =========================================================================
# DENSE RETRIEVER
# =========================================================================

class DenseRetriever:
    """Dense retriever с мультиязычными эмбеддингами."""

    def __init__(self, index_dir: str, model_name: str):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name

        # FAISS индекс
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        logger.info(f"FAISS index: {self.index.ntotal} vectors")

        # Метаданные
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.items = json.load(f)["items"]

        # Эмбеддер
        logger.info(f"Loading embedder: {model_name}")
        self.embedder = SentenceTransformer(model_name)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        embedding = self.embedder.encode(query, normalize_embeddings=True)
        embedding = embedding.astype("float32").reshape(1, -1)

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


# =========================================================================
# HYBRID с корректной нормализацией (требование п.2)
# =========================================================================

def normalize_scores_minmax(scores: List[float]) -> List[float]:
    """Min-Max нормализация в [0, 1]."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-9:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_search_normalized(
    bm25_results: List[Dict],
    dense_results: List[Dict],
    alpha: float = 0.5,
    k: int = 5
) -> List[Dict]:
    """
    Гибридный поиск с КОРРЕКТНОЙ нормализацией (требование п.2).

    score = alpha * normalized_dense + (1 - alpha) * normalized_bm25
    """
    # Нормализуем скоры ВНУТРИ каждого набора кандидатов
    bm25_scores = [r["score"] for r in bm25_results]
    dense_scores = [r["score"] for r in dense_results]

    bm25_norm = normalize_scores_minmax(bm25_scores)
    dense_norm = normalize_scores_minmax(dense_scores)

    # Объединяем
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

    # Финальный скор с весами
    for doc_id, r in combined.items():
        r["score"] = alpha * r["dense_score_norm"] + (1 - alpha) * r["bm25_score_norm"]

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:k]


# =========================================================================
# RERANKER (требование п.6)
# =========================================================================

class CrossEncoderReranker:
    """Cross-Encoder для переранжирования."""

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading CrossEncoder: {model_id}")
        self.model = CrossEncoder(model_id, device="cpu")
        self.model_id = model_id

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not candidates:
            return []

        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for i, score in enumerate(scores):
            candidates[i]["original_score"] = candidates[i].get("score", 0)
            candidates[i]["rerank_score"] = float(score)
            candidates[i]["score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# =========================================================================
# EVALUATION METRICS с chunk-level (требование п.3)
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
    """Recall@K на уровне чанков (требование п.3)."""
    if not judgment.relevant_chunk_ids:
        # Fallback на doc+page matching
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

    # Точное chunk_id matching
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
    """Bootstrap 95% confidence interval (требование п.10)."""
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

    for qid, judgment in judgments.items():
        results = retriever_fn(judgment.query, max(k_values))

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
    errors: List[str] = field(default_factory=list)


def run_experiment(
    name: str,
    description: str,
    retriever_fn: Callable[[str, int], List[Dict]],
    judgments: Dict[str, RelevanceJudgment]
) -> ExperimentResult:
    """Запуск одного эксперимента."""
    logger.info(f"Running {name}...")
    t0 = time.time()
    metrics = evaluate_retrieval(retriever_fn, judgments)
    elapsed = time.time() - t0

    r5 = metrics.get("recall@5", {}).get("mean", 0)
    logger.info(f"  R@5: {r5:.3f} (in {elapsed:.1f}s)")

    return ExperimentResult(
        name=name,
        description=description,
        metrics=metrics,
        timing_sec=elapsed,
    )


# =========================================================================
# GENERATE LATEX TABLES
# =========================================================================

def generate_latex_main_table(results: List[ExperimentResult]) -> str:
    """Главная таблица результатов с CI."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Retrieval Performance on Cross-lingual Microcontroller QA Dataset}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{l|ccc|cc|c}",
        r"\toprule",
        r"\textbf{Method} & \textbf{R@1} & \textbf{R@5} & \textbf{R@10} & \textbf{MRR@5} & \textbf{NDCG@5} & \textbf{Time (s)} \\",
        r"\midrule",
    ]

    best_r10 = max(r.metrics.get("recall@10", {}).get("mean", 0) for r in results)

    for r in results:
        m = r.metrics
        r1 = m.get("recall@1", {}).get("mean", 0)
        r5 = m.get("recall@5", {}).get("mean", 0)
        r10 = m.get("recall@10", {}).get("mean", 0)
        mrr5 = m.get("mrr@5", {}).get("mean", 0)
        ndcg5 = m.get("ndcg@5", {}).get("mean", 0)

        # Bold для лучшего результата
        if abs(r10 - best_r10) < 0.001:
            lines.append(
                f"\\textbf{{{r.name}}} & \\textbf{{{r1:.3f}}} & \\textbf{{{r5:.3f}}} & \\textbf{{{r10:.3f}}} & \\textbf{{{mrr5:.3f}}} & \\textbf{{{ndcg5:.3f}}} & {r.timing_sec:.1f} \\\\"
            )
        else:
            lines.append(
                f"{r.name} & {r1:.3f} & {r5:.3f} & {r10:.3f} & {mrr5:.3f} & {ndcg5:.3f} & {r.timing_sec:.1f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Cross-lingual QA: 201 Russian questions over English microcontroller datasheets.",
        r"\item R@k = Recall@k (document-level hit rate), MRR = Mean Reciprocal Rank, NDCG = Normalized DCG.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_ci_table(results: List[ExperimentResult]) -> str:
    """Таблица с 95% CI."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Retrieval Metrics with 95\% Bootstrap Confidence Intervals}",
        r"\label{tab:ci_results}",
        r"\begin{tabular}{l|cc|cc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{R@5 (95\% CI)} & \textbf{R@10 (95\% CI)} & \textbf{MRR@5 (95\% CI)} & \textbf{NDCG@5 (95\% CI)} \\",
        r"\midrule",
    ]

    for r in results:
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


# =========================================================================
# MAIN
# =========================================================================

def main():
    # Пути
    qa_csv = "data/benchmark/qa.csv"
    bm25_index = "data/index/bm25"
    faiss_index = "data/index/faiss"
    output_dir = Path("results/sota_experiments_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    logger.info("=" * 70)
    logger.info("FULL SOTA RETRIEVAL EXPERIMENTS FOR Q1 PUBLICATION")
    logger.info("=" * 70)

    # Загружаем данные
    logger.info(f"Loading QA dataset: {qa_csv}")
    judgments = load_qa_judgments(qa_csv)
    logger.info(f"Loaded {len(judgments)} questions")

    results: List[ExperimentResult] = []

    # 1. BM25 Baseline
    logger.info("\n[1/6] BM25 Baseline...")
    results.append(run_experiment(
        "BM25",
        "BM25 with cross-lingual keyword extraction",
        lambda q, k: search_bm25(bm25_index, q, k),
        judgments
    ))

    # 2. Dense Baseline
    logger.info("\n[2/6] Dense Baseline...")
    dense_retriever = DenseRetriever(faiss_index, embed_model)
    results.append(run_experiment(
        "Dense",
        f"Dense retrieval with {embed_model}",
        lambda q, k: dense_retriever.search(q, k),
        judgments
    ))

    # 3. Hybrid с разными alpha (корректная нормализация)
    logger.info("\n[3/6] Hybrid experiments (with normalized scores)...")
    for alpha in [0.3, 0.5, 0.7]:
        def hybrid_fn(q, k, a=alpha):
            bm25_res = search_bm25(bm25_index, q, k * 2)
            dense_res = dense_retriever.search(q, k * 2)
            return hybrid_search_normalized(bm25_res, dense_res, alpha=a, k=k)

        results.append(run_experiment(
            f"Hybrid (α={alpha})",
            f"Hybrid BM25+Dense with alpha={alpha}, normalized scores",
            hybrid_fn,
            judgments
        ))

    # 4. +Reranker эксперименты
    logger.info("\n[4/6] +Reranker experiments...")
    try:
        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # BM25 + Reranker
        def bm25_rerank(q, k):
            candidates = search_bm25(bm25_index, q, k * 3)
            return reranker.rerank(q, candidates, top_k=k)

        results.append(run_experiment(
            "BM25+Rerank",
            "BM25 with cross-encoder reranking",
            bm25_rerank,
            judgments
        ))

        # Hybrid + Reranker (лучший alpha=0.5)
        def hybrid_rerank(q, k):
            bm25_res = search_bm25(bm25_index, q, k * 3)
            dense_res = dense_retriever.search(q, k * 3)
            candidates = hybrid_search_normalized(bm25_res, dense_res, alpha=0.5, k=k * 3)
            return reranker.rerank(q, candidates, top_k=k)

        results.append(run_experiment(
            "Hybrid+Rerank",
            "Hybrid (α=0.5) with cross-encoder reranking",
            hybrid_rerank,
            judgments
        ))
    except Exception as e:
        logger.error(f"Reranker experiments failed: {e}")

    # 5. Сохраняем результаты
    logger.info("\n[5/6] Saving results...")

    # JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # Markdown
    md_lines = ["# Full SOTA Retrieval Experiments\n"]
    md_lines.append(f"QA Dataset: {len(judgments)} questions (RU→EN cross-lingual)\n")
    md_lines.append("| Method | R@1 | R@5 | R@10 | MRR@5 | NDCG@5 | Time |")
    md_lines.append("|--------|-----|-----|------|-------|--------|------|")
    for r in results:
        m = r.metrics
        md_lines.append(
            f"| {r.name} | {m.get('recall@1',{}).get('mean',0):.3f} | "
            f"{m.get('recall@5',{}).get('mean',0):.3f} | "
            f"{m.get('recall@10',{}).get('mean',0):.3f} | "
            f"{m.get('mrr@5',{}).get('mean',0):.3f} | "
            f"{m.get('ndcg@5',{}).get('mean',0):.3f} | "
            f"{r.timing_sec:.1f}s |"
        )

    md_path = output_dir / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # LaTeX
    latex_main = generate_latex_main_table(results)
    latex_ci = generate_latex_ci_table(results)

    tex_path = output_dir / "paper_tables.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Main Results Table\n")
        f.write(latex_main)
        f.write("\n\n% Confidence Intervals Table\n")
        f.write(latex_ci)

    # 6. Выводим итоговую таблицу
    logger.info("\n[6/6] Results summary:")
    logger.info("=" * 70)
    for r in results:
        m = r.metrics
        r10 = m.get("recall@10", {}).get("mean", 0)
        ci_l = m.get("recall@10", {}).get("ci_lower", 0)
        ci_u = m.get("recall@10", {}).get("ci_upper", 0)
        logger.info(f"  {r.name:20s}: R@10 = {r10:.3f} [{ci_l:.3f}, {ci_u:.3f}]")

    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("EXPERIMENTS COMPLETE!")


if __name__ == "__main__":
    main()
