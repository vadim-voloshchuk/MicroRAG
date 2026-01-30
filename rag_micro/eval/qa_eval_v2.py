"""
QA Evaluation v2 - Комплексная оценка end-to-end QA.

Отвечает на требования:
- Пункт 9: end-to-end QA eval (EM/F1), логирование источников
- Пункт 12: confidence/abstain режим
- Пункт 13: hallucination protocol с измеримыми метриками
"""
from __future__ import annotations
import os
import csv
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Tuple
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QAExample:
    """Один пример QA."""
    query_id: str
    question: str
    gold_answer: str
    gold_doc: Optional[str] = None
    gold_page: Optional[int] = None
    gold_section: Optional[str] = None


@dataclass
class QAResult:
    """Результат QA для одного примера."""
    query_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    sources: List[Dict]  # Найденные источники
    confidence_score: float = 0.0
    abstained: bool = False  # Отказался ли отвечать
    latency_ms: float = 0.0


@dataclass
class Citation:
    """Извлечённая цитата из ответа."""
    doc: str
    page: Optional[int]
    text: str  # Текст цитаты


@dataclass
class HallucinationAnalysis:
    """Анализ галлюцинаций для одного ответа."""
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    support_ratio: float
    extracted_citations: List[Citation]
    missing_citations: List[str]  # Утверждения без цитат
    is_hallucination: bool


@dataclass
class QAMetrics:
    """Метрики для одного QA примера."""
    query_id: str
    exact_match: bool
    f1: float
    support_ratio: float
    hallucination_analysis: Optional[HallucinationAnalysis] = None
    confidence_score: float = 0.0
    abstained: bool = False


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def normalize_text(s: str) -> str:
    """Нормализует текст для сравнения."""
    s = s.strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    # Убираем пунктуацию
    s = re.sub(r"[^\w\s]", "", s)
    return s


def extract_first_sentence(text: str) -> str:
    """Извлекает первое предложение/строку ответа."""
    lines = text.strip().split("\n")
    if lines:
        first_line = lines[0].strip()
        # Если первая строка содержит точку, берём до неё
        if "." in first_line:
            return first_line.split(".")[0] + "."
        return first_line
    return text


def word_level_f1(pred: str, gold: str) -> float:
    """
    F1 на уровне слов.
    F1 = 2 * P * R / (P + R)
    """
    pred_words = set(normalize_text(pred).split())
    gold_words = set(normalize_text(gold).split())

    if not pred_words and not gold_words:
        return 1.0
    if not pred_words or not gold_words:
        return 0.0

    intersection = len(pred_words & gold_words)
    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_words)
    recall = intersection / len(gold_words)

    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    """Точное совпадение после нормализации."""
    return normalize_text(pred) == normalize_text(gold)


# =============================================================================
# CONFIDENCE / ABSTAIN
# =============================================================================

class ConfidenceEstimator:
    """
    Оценка уверенности на основе retrieval scores.

    Отвечает на пункт 12: при низкой уверенности не генерировать ответ.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.3,
        min_sources_threshold: int = 1,
        score_aggregation: Literal["max", "mean", "top3_mean"] = "top3_mean"
    ):
        self.min_score_threshold = min_score_threshold
        self.min_sources_threshold = min_sources_threshold
        self.score_aggregation = score_aggregation

    def estimate_confidence(self, sources: List[Dict]) -> Tuple[float, bool]:
        """
        Оценивает уверенность в ответе.

        Returns:
            Tuple[confidence_score, should_abstain]
        """
        if not sources or len(sources) < self.min_sources_threshold:
            return 0.0, True

        scores = [s.get("score", 0) for s in sources]

        # Агрегация scores
        if self.score_aggregation == "max":
            confidence = max(scores)
        elif self.score_aggregation == "mean":
            confidence = np.mean(scores)
        elif self.score_aggregation == "top3_mean":
            confidence = np.mean(sorted(scores, reverse=True)[:3])
        else:
            confidence = max(scores)

        should_abstain = confidence < self.min_score_threshold

        return float(confidence), should_abstain

    def format_abstain_response(self, sources: List[Dict], k: int = 3) -> str:
        """
        Форматирует ответ при отказе.
        """
        response = "Недостаточно информации для уверенного ответа.\n\n"
        response += "Возможно релевантные источники:\n"

        for i, s in enumerate(sources[:k], 1):
            doc = s.get("doc", "?")
            page = s.get("page", "?")
            score = s.get("score", 0)
            snippet = s.get("text", "")[:200]
            response += f"\n{i}. {doc}, стр. {page} (score: {score:.2f})\n"
            response += f"   «{snippet}...»\n"

        response += "\nРекомендуется уточнить запрос или проверить указанные разделы вручную."
        return response


# =============================================================================
# HALLUCINATION DETECTION
# =============================================================================

class HallucinationDetector:
    """
    Детектор галлюцинаций.

    Отвечает на пункт 13:
    - Проверка "утверждения без поддержки в evidence"
    - Обязательные цитаты
    - Метрика hallucination rate
    """

    # Паттерн для извлечения цитат вида (Doc, стр. N) или [Doc, p. N]
    CITATION_PATTERN = re.compile(
        r'\(([^,]+),\s*(?:стр\.?|p\.?|page)\s*(\d+)\)|'
        r'\[([^,]+),\s*(?:стр\.?|p\.?|page)\s*(\d+)\]',
        re.IGNORECASE
    )

    def __init__(
        self,
        min_support_ratio: float = 0.3,
        require_citations: bool = True
    ):
        self.min_support_ratio = min_support_ratio
        self.require_citations = require_citations

    def extract_citations(self, answer: str) -> List[Citation]:
        """Извлекает цитаты из ответа."""
        citations = []
        for match in self.CITATION_PATTERN.finditer(answer):
            doc = match.group(1) or match.group(3)
            page = int(match.group(2) or match.group(4))
            # Текст до цитаты
            start = max(0, match.start() - 100)
            text = answer[start:match.start()].strip()
            citations.append(Citation(doc=doc, page=page, text=text))
        return citations

    def extract_claims(self, answer: str) -> List[str]:
        """
        Извлекает утверждения из ответа.
        Простая эвристика: каждое предложение = утверждение.
        """
        # Разбиваем на предложения
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        # Фильтруем короткие и служебные
        claims = [
            s.strip() for s in sentences
            if len(s.strip()) > 20 and not s.strip().startswith("Источник")
        ]
        return claims

    def check_claim_support(self, claim: str, sources: List[Dict]) -> bool:
        """
        Проверяет, поддерживается ли утверждение источниками.
        """
        claim_words = set(normalize_text(claim).split())
        if not claim_words:
            return True

        # Собираем все слова из источников
        source_text = " ".join(s.get("text", "") for s in sources)
        source_words = set(normalize_text(source_text).split())

        # Считаем перекрытие
        overlap = len(claim_words & source_words)
        ratio = overlap / len(claim_words)

        return ratio >= self.min_support_ratio

    def analyze(self, answer: str, sources: List[Dict]) -> HallucinationAnalysis:
        """
        Полный анализ галлюцинаций.
        """
        claims = self.extract_claims(answer)
        citations = self.extract_citations(answer)

        supported = 0
        unsupported = 0
        missing_citations = []

        for claim in claims:
            if self.check_claim_support(claim, sources):
                supported += 1
            else:
                unsupported += 1
                missing_citations.append(claim)

        total = len(claims) if claims else 1
        support_ratio = supported / total

        # Проверяем наличие цитат
        has_citations = len(citations) > 0 if self.require_citations else True

        is_hallucination = (support_ratio < self.min_support_ratio) or \
                          (self.require_citations and not has_citations)

        return HallucinationAnalysis(
            total_claims=len(claims),
            supported_claims=supported,
            unsupported_claims=unsupported,
            support_ratio=support_ratio,
            extracted_citations=citations,
            missing_citations=missing_citations,
            is_hallucination=is_hallucination,
        )


# =============================================================================
# QA EVALUATOR
# =============================================================================

class QAEvaluator:
    """
    Полный evaluator для end-to-end QA.
    """

    def __init__(
        self,
        confidence_estimator: Optional[ConfidenceEstimator] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95
    ):
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator()
        self.hallucination_detector = hallucination_detector or HallucinationDetector()
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level

    def evaluate_single(self, result: QAResult) -> QAMetrics:
        """Оценивает один пример."""
        # Извлекаем первую строку/предложение для сравнения
        pred_first = extract_first_sentence(result.predicted_answer)

        # EM и F1
        em = exact_match(pred_first, result.gold_answer)
        f1 = word_level_f1(pred_first, result.gold_answer)

        # Support ratio (простая версия)
        answer_words = set(normalize_text(result.predicted_answer).split())
        source_words = set()
        for s in result.sources:
            source_words.update(normalize_text(s.get("text", "")).split())

        if answer_words:
            support_ratio = len(answer_words & source_words) / len(answer_words)
        else:
            support_ratio = 1.0

        # Hallucination analysis
        halluc_analysis = self.hallucination_detector.analyze(
            result.predicted_answer, result.sources
        )

        return QAMetrics(
            query_id=result.query_id,
            exact_match=em,
            f1=f1,
            support_ratio=support_ratio,
            hallucination_analysis=halluc_analysis,
            confidence_score=result.confidence_score,
            abstained=result.abstained,
        )

    def evaluate_batch(self, results: List[QAResult]) -> Dict:
        """Оценивает батч и агрегирует метрики."""
        all_metrics: List[QAMetrics] = []

        for result in results:
            metrics = self.evaluate_single(result)
            all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Агрегация
        n = len(all_metrics)
        em_values = [float(m.exact_match) for m in all_metrics]
        f1_values = [m.f1 for m in all_metrics]
        support_values = [m.support_ratio for m in all_metrics]
        halluc_values = [float(m.hallucination_analysis.is_hallucination) for m in all_metrics if m.hallucination_analysis]
        abstain_values = [float(m.abstained) for m in all_metrics]

        def bootstrap_ci(values):
            from .retrieval_eval import bootstrap_ci as bc
            return bc(values, self.bootstrap_samples, self.confidence_level)

        aggregated = {
            "n_queries": n,
            "n_abstained": sum(abstain_values),
            "metrics": {
                "exact_match": {
                    "mean": np.mean(em_values),
                    "ci_lower": bootstrap_ci(em_values)[1],
                    "ci_upper": bootstrap_ci(em_values)[2],
                },
                "f1": {
                    "mean": np.mean(f1_values),
                    "ci_lower": bootstrap_ci(f1_values)[1],
                    "ci_upper": bootstrap_ci(f1_values)[2],
                },
                "support_ratio": {
                    "mean": np.mean(support_values),
                    "ci_lower": bootstrap_ci(support_values)[1],
                    "ci_upper": bootstrap_ci(support_values)[2],
                },
                "hallucination_rate": {
                    "mean": np.mean(halluc_values) if halluc_values else 0.0,
                    "ci_lower": bootstrap_ci(halluc_values)[1] if halluc_values else 0.0,
                    "ci_upper": bootstrap_ci(halluc_values)[2] if halluc_values else 0.0,
                },
                "abstain_rate": {
                    "mean": np.mean(abstain_values),
                },
            },
            "detailed": [
                {
                    "query_id": m.query_id,
                    "exact_match": m.exact_match,
                    "f1": m.f1,
                    "support_ratio": m.support_ratio,
                    "is_hallucination": m.hallucination_analysis.is_hallucination if m.hallucination_analysis else False,
                    "confidence": m.confidence_score,
                    "abstained": m.abstained,
                }
                for m in all_metrics
            ],
        }

        return aggregated


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_qa_results_jsonl(results: List[QAResult], output_path: str):
    """Сохраняет все результаты в JSONL для воспроизводимости."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            record = {
                "query_id": r.query_id,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "predicted_answer": r.predicted_answer,
                "sources": [
                    {
                        "doc": s.get("doc"),
                        "page": s.get("page"),
                        "score": s.get("score"),
                        "snippet": s.get("text", "")[:200],
                    }
                    for s in r.sources
                ],
                "confidence_score": r.confidence_score,
                "abstained": r.abstained,
                "latency_ms": r.latency_ms,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"QA results saved to {path}")


def find_low_f1_useful_cases(
    results: List[QAResult],
    metrics: List[QAMetrics],
    n_cases: int = 5
) -> List[Dict]:
    """
    Пункт 17: Находит случаи где F1 низкий, но retrieval полезен.

    Такие кейсы показывают ценность системы даже при низких метриках.
    """
    cases = []

    for result, metric in zip(results, metrics):
        # Низкий F1, но есть источники
        if metric.f1 < 0.3 and len(result.sources) > 0:
            # Проверяем, что источники релевантны
            # (упрощённая эвристика: score > 0.5)
            good_sources = [s for s in result.sources if s.get("score", 0) > 0.5]
            if good_sources:
                cases.append({
                    "query_id": result.query_id,
                    "question": result.question,
                    "gold_answer": result.gold_answer,
                    "predicted_answer": result.predicted_answer,
                    "f1": metric.f1,
                    "best_source": {
                        "doc": good_sources[0].get("doc"),
                        "page": good_sources[0].get("page"),
                        "snippet": good_sources[0].get("text", "")[:300],
                        "score": good_sources[0].get("score"),
                    },
                    "why_useful": "Несмотря на низкий F1, система нашла релевантный раздел документации",
                })

    # Сортируем по score источника
    cases.sort(key=lambda x: x["best_source"]["score"], reverse=True)
    return cases[:n_cases]


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def run_qa_evaluation(
    qa_pipeline_fn,  # Функция (question) -> (answer, sources)
    qa_csv_path: str,
    output_dir: str = "results/qa",
    experiment_name: str = "baseline",
    use_confidence: bool = True,
    use_hallucination_check: bool = True,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95
) -> Dict:
    """
    Полный pipeline оценки QA.

    Args:
        qa_pipeline_fn: Функция, принимающая вопрос и возвращающая (ответ, источники)
        qa_csv_path: Путь к CSV с вопросами
        output_dir: Директория для результатов
        experiment_name: Имя эксперимента
        use_confidence: Использовать ли confidence/abstain
        use_hallucination_check: Проверять ли галлюцинации

    Returns:
        Dict: Агрегированные результаты
    """
    import time

    # Инициализация
    confidence_estimator = ConfidenceEstimator() if use_confidence else None
    hallucination_detector = HallucinationDetector() if use_hallucination_check else None
    evaluator = QAEvaluator(
        confidence_estimator,
        hallucination_detector,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )

    # Загружаем примеры
    examples: List[QAExample] = []
    with open(qa_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            examples.append(QAExample(
                query_id=f"q{i}",
                question=row["question"],
                gold_answer=row["answer"],
                gold_doc=row.get("doc"),
                gold_page=int(row["page"]) if row.get("page") else None,
            ))

    logger.info(f"Loaded {len(examples)} QA examples from {qa_csv_path}")

    # Прогоняем QA
    results: List[QAResult] = []

    for example in examples:
        t0 = time.time()

        # Получаем ответ
        answer, sources = qa_pipeline_fn(example.question)
        latency_ms = (time.time() - t0) * 1000

        # Проверяем confidence
        confidence_score = 0.0
        abstained = False
        if confidence_estimator:
            confidence_score, should_abstain = confidence_estimator.estimate_confidence(sources)
            if should_abstain:
                abstained = True
                answer = confidence_estimator.format_abstain_response(sources)

        results.append(QAResult(
            query_id=example.query_id,
            question=example.question,
            gold_answer=example.gold_answer,
            predicted_answer=answer,
            sources=sources,
            confidence_score=confidence_score,
            abstained=abstained,
            latency_ms=latency_ms,
        ))

    # Оцениваем
    aggregated = evaluator.evaluate_batch(results)
    aggregated["experiment"] = experiment_name

    # Находим полезные кейсы с низким F1
    metrics = [evaluator.evaluate_single(r) for r in results]
    useful_cases = find_low_f1_useful_cases(results, metrics)
    aggregated["low_f1_useful_cases"] = useful_cases

    # Экспортируем
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSONL с полными результатами
    export_qa_results_jsonl(results, str(output_path / f"{experiment_name}_results.jsonl"))

    # Summary CSV
    summary_path = output_path / f"{experiment_name}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "ci_lower", "ci_upper"])
        for metric_name, values in aggregated["metrics"].items():
            writer.writerow([
                metric_name,
                f"{values['mean']:.4f}",
                f"{values.get('ci_lower', 0):.4f}",
                f"{values.get('ci_upper', 0):.4f}",
            ])

    # Detailed per-query metrics CSV
    detailed_path = output_path / f"{experiment_name}_detailed.csv"
    if aggregated.get("detailed"):
        fieldnames = list(aggregated["detailed"][0].keys())
        with open(detailed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregated["detailed"]:
                writer.writerow(row)

    # Useful cases JSON
    cases_path = output_path / f"{experiment_name}_useful_cases.json"
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(useful_cases, f, ensure_ascii=False, indent=2)

    logger.info(f"QA evaluation results saved to {output_path}")

    return aggregated
