"""
Experiment Runner - Автоматический раннер экспериментов.

Отвечает на требования:
- Пункт 14: Абляции chunking (chunk_size × overlap × strategy)
- Пункт 15: Сравнение baseline-ов (BM25 vs Dense vs Hybrid vs +Reranker)
- Пункт 16: Тест табличного вклада (text-only vs text+tables)
"""
from __future__ import annotations
import os
import json
import csv
import time
import itertools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any
import logging
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Конфигурация одного эксперимента."""
    name: str
    description: str = ""

    # Chunking params
    chunk_strategy: str = "sentence_aware"
    chunk_size: int = 512
    chunk_overlap: int = 128

    # Embedding params (мультиязычный по умолчанию для RU вопросов + EN документов)
    embed_model: str = "multilingual-minilm"

    # Retrieval params
    retrieval_mode: str = "hybrid"  # bm25 | dense | hybrid
    hybrid_alpha: float = 0.5
    use_reranker: bool = False
    reranker_model: str = "ms-marco-mini"

    # Data params
    include_tables: bool = True

    # Other
    seed: int = 42


@dataclass
class ExperimentResult:
    """Результат эксперимента."""
    config: ExperimentConfig
    retrieval_metrics: Dict = field(default_factory=dict)
    qa_metrics: Dict = field(default_factory=dict)
    timing: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# ABLATION GRIDS
# =============================================================================

def generate_chunking_ablation_grid() -> List[ExperimentConfig]:
    """
    Пункт 14: Сетка chunk_size × overlap × strategy.
    """
    strategies = ["fixed_tokens", "sentence_aware", "section_aware"]
    chunk_sizes = [256, 512, 768, 1024]
    overlaps = [0, 64, 128, 256]

    configs = []
    for strategy, size, overlap in itertools.product(strategies, chunk_sizes, overlaps):
        # Overlap не должен превышать половину chunk_size
        if overlap >= size / 2:
            continue

        configs.append(ExperimentConfig(
            name=f"chunk_{strategy}_{size}_{overlap}",
            description=f"Chunking ablation: {strategy}, size={size}, overlap={overlap}",
            chunk_strategy=strategy,
            chunk_size=size,
            chunk_overlap=overlap,
        ))

    return configs


def generate_retrieval_baseline_grid() -> List[ExperimentConfig]:
    """
    Пункт 15: BM25 vs Dense vs Hybrid vs +Reranker.
    Используем мультиязычные эмбеддеры для RU вопросов + EN документов.
    """
    # Мультиязычные модели для cross-lingual retrieval
    embed_models = ["multilingual-minilm", "multilingual-e5-base", "labse"]

    configs = []

    # BM25 only
    configs.append(ExperimentConfig(
        name="bm25_only",
        description="BM25 baseline (lexical only)",
        retrieval_mode="bm25",
    ))

    # Dense only (разные мультиязычные эмбеддинги)
    for model in embed_models:
        configs.append(ExperimentConfig(
            name=f"dense_{model}",
            description=f"Dense retrieval with {model}",
            retrieval_mode="dense",
            embed_model=model,
        ))

    # Hybrid (разные alpha)
    for alpha in [0.3, 0.5, 0.7]:
        for model in embed_models:
            configs.append(ExperimentConfig(
                name=f"hybrid_{model}_alpha{alpha}",
                description=f"Hybrid retrieval: alpha={alpha}, embed={model}",
                retrieval_mode="hybrid",
                embed_model=model,
                hybrid_alpha=alpha,
            ))

    # +Reranker
    for model in embed_models:
        configs.append(ExperimentConfig(
            name=f"hybrid_{model}_rerank",
            description=f"Hybrid + Reranker: embed={model}",
            retrieval_mode="hybrid",
            embed_model=model,
            use_reranker=True,
        ))

    return configs


def generate_table_ablation_grid() -> List[ExperimentConfig]:
    """
    Пункт 16: text-only vs text+tables.
    """
    return [
        ExperimentConfig(
            name="text_only",
            description="Text-only (tables excluded)",
            include_tables=False,
        ),
        ExperimentConfig(
            name="text_and_tables",
            description="Text + Tables",
            include_tables=True,
        ),
    ]


def generate_embedder_comparison_grid() -> List[ExperimentConfig]:
    """
    Пункт 5: Сравнение эмбеддеров.
    Мультиязычные модели для RU вопросов + EN документов.
    """
    # Мультиязычные модели разного размера и качества
    models = [
        "multilingual-minilm",      # 384d, быстрая
        "multilingual-e5-base",     # 768d, хороший баланс
        "multilingual-e5-large",    # 1024d, лучшее качество
        "labse",                    # 768d, Google LaBSE
        "bge-m3",                   # 1024d, SOTA мультиязычная
    ]

    return [
        ExperimentConfig(
            name=f"embed_{model}",
            description=f"Embedding comparison: {model}",
            embed_model=model,
            retrieval_mode="dense",
        )
        for model in models
    ]


def generate_reranker_comparison_grid() -> List[ExperimentConfig]:
    """
    Сравнение reranker моделей.
    """
    rerankers = ["ms-marco-tiny", "ms-marco-mini", "ms-marco-medium", "bge-reranker-base"]

    return [
        ExperimentConfig(
            name=f"rerank_{reranker}",
            description=f"Reranker comparison: {reranker}",
            retrieval_mode="hybrid",
            use_reranker=True,
            reranker_model=reranker,
        )
        for reranker in rerankers
    ]


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Автоматический раннер экспериментов.
    """

    def __init__(
        self,
        base_config_path: str = "config.yaml",
        output_dir: str = "results/experiments",
        qa_csv_path: str = "data/benchmark/qa.csv"
    ):
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.qa_csv_path = qa_csv_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Загружаем базовый конфиг
        with open(base_config_path, "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        retrieval_eval_fn: Callable,
        qa_eval_fn: Optional[Callable] = None
    ) -> ExperimentResult:
        """
        Запускает один эксперимент.
        """
        logger.info(f"Running experiment: {config.name}")
        result = ExperimentResult(config=config)

        try:
            # Timing
            t0 = time.time()

            # Retrieval evaluation
            retrieval_results = retrieval_eval_fn(config)
            result.retrieval_metrics = retrieval_results

            t_retrieval = time.time() - t0

            # QA evaluation (если нужно)
            if qa_eval_fn:
                t1 = time.time()
                qa_results = qa_eval_fn(config)
                result.qa_metrics = qa_results
                t_qa = time.time() - t1
            else:
                t_qa = 0

            result.timing = {
                "retrieval_sec": t_retrieval,
                "qa_sec": t_qa,
                "total_sec": time.time() - t0,
            }

        except Exception as e:
            logger.error(f"Experiment {config.name} failed: {e}")
            result.errors.append(str(e))

        return result

    def run_ablation(
        self,
        configs: List[ExperimentConfig],
        retrieval_eval_fn: Callable,
        qa_eval_fn: Optional[Callable] = None,
        ablation_name: str = "ablation"
    ) -> List[ExperimentResult]:
        """
        Запускает серию экспериментов (абляцию).
        """
        logger.info(f"Running ablation '{ablation_name}' with {len(configs)} configs")

        results = []
        for i, config in enumerate(configs, 1):
            logger.info(f"[{i}/{len(configs)}] {config.name}")
            result = self.run_single_experiment(config, retrieval_eval_fn, qa_eval_fn)
            results.append(result)

        # Сохраняем результаты
        self._save_ablation_results(results, ablation_name)

        return results

    def _save_ablation_results(self, results: List[ExperimentResult], ablation_name: str):
        """Сохраняет результаты абляции."""
        output_path = self.output_dir / ablation_name

        # CSV summary
        csv_path = output_path.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            # Определяем колонки из первого результата
            if not results:
                return

            fieldnames = ["name", "description"]

            # Config fields
            config_fields = [
                "chunk_strategy", "chunk_size", "chunk_overlap",
                "embed_model", "retrieval_mode", "hybrid_alpha",
                "use_reranker", "include_tables"
            ]
            fieldnames.extend(config_fields)

            # Metric fields (из первого результата)
            metric_fields = []
            if results[0].retrieval_metrics.get("metrics"):
                for k in results[0].retrieval_metrics["metrics"].keys():
                    metric_fields.append(f"ret_{k}")
            if results[0].qa_metrics.get("metrics"):
                for k in results[0].qa_metrics["metrics"].keys():
                    metric_fields.append(f"qa_{k}")
            fieldnames.extend(metric_fields)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "name": result.config.name,
                    "description": result.config.description,
                }

                # Config values
                for field in config_fields:
                    row[field] = getattr(result.config, field, "")

                # Retrieval metrics
                for k, v in result.retrieval_metrics.get("metrics", {}).items():
                    row[f"ret_{k}"] = f"{v.get('mean', 0):.4f}"

                # QA metrics
                for k, v in result.qa_metrics.get("metrics", {}).items():
                    row[f"qa_{k}"] = f"{v.get('mean', 0):.4f}"

                writer.writerow(row)

        logger.info(f"Ablation results saved to {csv_path}")

        # JSON detailed
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "config": asdict(r.config),
                        "retrieval_metrics": r.retrieval_metrics,
                        "qa_metrics": r.qa_metrics,
                        "timing": r.timing,
                        "errors": r.errors,
                    }
                    for r in results
                ],
                f,
                ensure_ascii=False,
                indent=2
            )

        logger.info(f"Detailed results saved to {json_path}")


# =============================================================================
# MAIN ABLATION RUNNERS
# =============================================================================

def run_chunking_ablation(
    runner: ExperimentRunner,
    retrieval_eval_fn: Callable,
    qa_eval_fn: Optional[Callable] = None
) -> List[ExperimentResult]:
    """Пункт 14: Абляция chunking."""
    configs = generate_chunking_ablation_grid()
    return runner.run_ablation(configs, retrieval_eval_fn, qa_eval_fn, "chunking_ablation")


def run_baseline_comparison(
    runner: ExperimentRunner,
    retrieval_eval_fn: Callable,
    qa_eval_fn: Optional[Callable] = None
) -> List[ExperimentResult]:
    """Пункт 15: Сравнение baseline-ов."""
    configs = generate_retrieval_baseline_grid()
    return runner.run_ablation(configs, retrieval_eval_fn, qa_eval_fn, "baseline_comparison")


def run_table_ablation(
    runner: ExperimentRunner,
    retrieval_eval_fn: Callable,
    qa_eval_fn: Optional[Callable] = None
) -> List[ExperimentResult]:
    """Пункт 16: Тест табличного вклада."""
    configs = generate_table_ablation_grid()
    return runner.run_ablation(configs, retrieval_eval_fn, qa_eval_fn, "table_ablation")


def run_embedder_comparison(
    runner: ExperimentRunner,
    retrieval_eval_fn: Callable
) -> List[ExperimentResult]:
    """Пункт 5: Сравнение эмбеддеров."""
    configs = generate_embedder_comparison_grid()
    return runner.run_ablation(configs, retrieval_eval_fn, None, "embedder_comparison")


def run_reranker_comparison(
    runner: ExperimentRunner,
    retrieval_eval_fn: Callable
) -> List[ExperimentResult]:
    """Сравнение reranker'ов."""
    configs = generate_reranker_comparison_grid()
    return runner.run_ablation(configs, retrieval_eval_fn, None, "reranker_comparison")


# =============================================================================
# RESULT ANALYSIS
# =============================================================================

def find_best_config(results: List[ExperimentResult], metric: str = "recall@5") -> ExperimentConfig:
    """Находит лучшую конфигурацию по метрике."""
    best = None
    best_score = -1

    for result in results:
        if result.errors:
            continue

        score = result.retrieval_metrics.get("metrics", {}).get(metric, {}).get("mean", 0)
        if score > best_score:
            best_score = score
            best = result.config

    return best


def generate_comparison_table(
    results: List[ExperimentResult],
    metrics: List[str] = ["recall@5", "mrr@5", "ndcg@5"]
) -> str:
    """Генерирует таблицу сравнения для статьи."""
    lines = [
        "| Method | " + " | ".join(metrics) + " |",
        "|--------|" + "|".join(["--------"] * len(metrics)) + "|",
    ]

    for result in results:
        if result.errors:
            continue

        row = [result.config.name]
        for m in metrics:
            val = result.retrieval_metrics.get("metrics", {}).get(m, {})
            mean = val.get("mean", 0)
            ci_l = val.get("ci_lower", 0)
            ci_u = val.get("ci_upper", 0)
            row.append(f"{mean:.3f} ({ci_l:.3f}-{ci_u:.3f})")

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
