#!/usr/bin/env python3
"""
CLI для запуска экспериментов RAG.

Использование:
    python -m rag_micro.run_experiments chunking --qa-csv data/benchmark/qa.csv
    python -m rag_micro.run_experiments baselines --qa-csv data/benchmark/qa.csv
    python -m rag_micro.run_experiments tables --qa-csv data/benchmark/qa.csv
    python -m rag_micro.run_experiments all --qa-csv data/benchmark/qa.csv
"""
import typer
import os
import json
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="RAG Experiments CLI")


def resolve_embed_device() -> str:
    env_device = os.getenv("EMBED_DEVICE")
    if env_device:
        return env_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_embed_batch_size() -> Optional[int]:
    value = os.getenv("EMBED_BATCH_SIZE")
    if not value:
        return None
    try:
        batch_size = int(value)
    except ValueError:
        return None
    return batch_size if batch_size > 0 else None


def get_retrieval_fn(index_root: str, embed_model: str, mode: str, alpha: float = 0.5):
    """Фабрика для создания retrieval функций."""
    from .retrievers.hybrid_v2 import search_hybrid_v2, HybridConfig
    from .retrievers.bm25_whoosh import search_whoosh
    from .retrievers.faiss_dense import search_faiss
    from .retrievers.embedders import get_embedder

    embedder = get_embedder(embed_model)
    embed_model_id = embedder.info.model_id

    def retrieve(query: str, k: int = 5):
        if mode == "bm25":
            return search_whoosh(f"{index_root}/bm25", query, k=k)
        elif mode == "dense":
            return search_faiss(f"{index_root}/faiss", query, k=k, embed_model=embed_model_id)
        else:
            config = HybridConfig(alpha=alpha)
            return search_hybrid_v2(index_root, query, k=k, embed_model=embed_model_id, config=config)

    return retrieve


def build_index_for_config(
    config,
    index_root: str,
    ablation_name: str,
    raw_pdf_dir: str,
    processed_txt_dir: str,
    embed_device: str,
    embed_batch_size: Optional[int],
    table_format: str = "markdown",
    table_context_sentences: int = 2,
    force_rebuild: bool = False,
) -> str:
    from .ingest.build_corpus_v2 import ingest_v2, get_corpus_stats
    from .retrievers.bm25_whoosh import build_whoosh_index
    from .retrievers.faiss_dense import build_faiss_index
    from .retrievers.embedders import get_embedder

    index_dir = Path(index_root) / ablation_name / config.name
    index_dir.mkdir(parents=True, exist_ok=True)

    meta_path = index_dir / "index_metadata.json"
    faiss_path = index_dir / "faiss" / "faiss.index"
    bm25_dir = index_dir / "bm25"
    corpus_path = index_dir / "corpus.jsonl"

    expected_meta = {
        "chunk_strategy": config.chunk_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "include_tables": config.include_tables,
        "embed_model": config.embed_model,
        "table_format": table_format,
        "table_context_sentences": table_context_sentences,
    }

    if not force_rebuild and meta_path.exists() and faiss_path.exists() and bm25_dir.exists() and corpus_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("config") == expected_meta:
                return str(index_dir)
        except Exception:
            pass

    logger.info(
        "Rebuilding index for %s (strategy=%s, size=%s, overlap=%s, tables=%s)",
        config.name,
        config.chunk_strategy,
        config.chunk_size,
        config.chunk_overlap,
        config.include_tables,
    )

    corpus_jsonl, total_chunks = ingest_v2(
        from_pdf=raw_pdf_dir,
        from_txt=processed_txt_dir,
        out_root=str(index_dir),
        chunk_strategy=config.chunk_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        include_tables=config.include_tables,
        table_format=table_format,
        table_context_sentences=table_context_sentences,
    )

    stats = get_corpus_stats(corpus_jsonl)
    build_whoosh_index(corpus_jsonl, str(bm25_dir))

    embedder = get_embedder(config.embed_model, device=embed_device)
    build_faiss_index(
        corpus_jsonl,
        str(index_dir / "faiss"),
        config.embed_model,
        device=embed_device,
        batch_size=embed_batch_size,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": expected_meta,
                "embed_model_id": embedder.info.model_id,
                "embed_dim": embedder.info.dim,
                "total_chunks": total_chunks,
                "corpus_stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(index_dir)


def get_qa_fn(retriever_fn, llm_backend: str = "openai", llm_config: Optional[dict] = None):
    """Фабрика для создания QA функций."""
    from .config import settings
    from .llm.openai_backend import answer_openai
    from .llm.ollama_backend import answer_ollama
    from .llm.llama_cpp_backend import answer_llama
    from .llm.stub import answer_stub

    cfg = llm_config or {}
    backend = llm_backend or cfg.get("backend") or "openai"

    def qa_pipeline(question: str):
        sources = retriever_fn(question, k=5)

        if backend == "openai":
            openai_cfg = cfg.get("openai", {})
            model = openai_cfg.get("model", settings.openai_model)
            temperature = openai_cfg.get("temperature", 0.1)
            max_tokens = openai_cfg.get("max_tokens", 300)
            system_prompt = cfg.get("system_prompt")
            answer = answer_openai(
                sources,
                question,
                model,
                settings.openai_api_key,
                settings.openai_base_url,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif backend == "ollama":
            model = cfg.get("ollama", {}).get("model", settings.ollama_model)
            answer = answer_ollama(sources, question, model, settings.ollama_host)
        elif backend == "llama_cpp":
            model_path = settings.llama_model_path
            if not model_path:
                raise RuntimeError("LLAMA_MODEL_PATH не задан.")
            answer = answer_llama(sources, question, model_path, settings.llama_ctx, settings.llama_threads)
        else:
            answer = answer_stub(sources, question)

        return answer, sources

    return qa_pipeline


@app.command()
def chunking(
    qa_csv: str = typer.Option("data/benchmark/qa.csv", help="Path to QA CSV"),
    index_root: str = typer.Option("data/index", help="Index root"),
    output_dir: str = typer.Option("results/chunking_ablation", help="Output directory"),
    raw_pdf_dir: str = typer.Option("data/raw", help="Path to raw PDF directory"),
    processed_txt_dir: str = typer.Option("data/processed/text", help="Path to processed TXT directory"),
    table_format: str = typer.Option("markdown", help="Table format for extraction"),
    table_context_sentences: int = typer.Option(2, help="Context sentences around tables"),
    force_rebuild: bool = typer.Option(False, help="Force reindexing even if cache exists"),
):
    """
    Пункт 14: Абляция параметров чанкинга.
    Прогоняет сетку chunk_size × overlap × strategy.
    """
    from .experiments.runner import ExperimentRunner, run_chunking_ablation

    typer.echo("Running chunking ablation...")

    runner = ExperimentRunner(
        output_dir=output_dir,
        qa_csv_path=qa_csv,
    )

    embed_device = resolve_embed_device()
    embed_batch_size = resolve_embed_batch_size()

    def retrieval_eval_fn(config):
        from .eval.retrieval_eval import run_retrieval_evaluation

        config.embed_model = config.embed_model or "multilingual-minilm"
        config.retrieval_mode = config.retrieval_mode or "hybrid"

        config_index_root = build_index_for_config(
            config,
            index_root=index_root,
            ablation_name="chunking_ablation",
            raw_pdf_dir=raw_pdf_dir,
            processed_txt_dir=processed_txt_dir,
            embed_device=embed_device,
            embed_batch_size=embed_batch_size,
            table_format=table_format,
            table_context_sentences=table_context_sentences,
            force_rebuild=force_rebuild,
        )

        retriever_fn = get_retrieval_fn(
            config_index_root,
            config.embed_model,
            config.retrieval_mode,
            config.hybrid_alpha,
        )
        return run_retrieval_evaluation(
            retriever_fn,
            qa_csv,
            output_dir=output_dir,
            experiment_name=config.name,
        )

    results = run_chunking_ablation(runner, retrieval_eval_fn)

    typer.echo(f"Completed {len(results)} experiments")
    typer.echo(f"Results saved to {output_dir}")


@app.command()
def baselines(
    qa_csv: str = typer.Option("data/benchmark/qa.csv", help="Path to QA CSV"),
    index_root: str = typer.Option("data/index", help="Index root"),
    output_dir: str = typer.Option("results/baseline_comparison", help="Output directory"),
):
    """
    Пункт 15: Сравнение baseline-ов.
    BM25 vs Dense vs Hybrid vs +Reranker.
    """
    from .experiments.runner import ExperimentRunner, run_baseline_comparison

    typer.echo("Running baseline comparison...")

    runner = ExperimentRunner(
        output_dir=output_dir,
        qa_csv_path=qa_csv,
    )

    def retrieval_eval_fn(config):
        from .eval.retrieval_eval import run_retrieval_evaluation

        retriever_fn = get_retrieval_fn(
            index_root,
            config.embed_model,
            config.retrieval_mode,
            config.hybrid_alpha
        )

        if config.use_reranker:
            from .retrievers.reranker import get_reranker, retrieve_and_rerank
            reranker = get_reranker(config.reranker_model)
            original_fn = retriever_fn

            def reranked_retriever(query, k=5):
                return retrieve_and_rerank(
                    lambda q, k: original_fn(q, k=20),
                    reranker, query, retrieve_k=20, final_k=k
                )

            retriever_fn = reranked_retriever

        return run_retrieval_evaluation(
            retriever_fn, qa_csv,
            output_dir=output_dir,
            experiment_name=config.name
        )

    results = run_baseline_comparison(runner, retrieval_eval_fn)

    typer.echo(f"Completed {len(results)} experiments")
    typer.echo(f"Results saved to {output_dir}")


@app.command()
def tables(
    qa_csv: str = typer.Option("data/benchmark/qa.csv", help="Path to QA CSV"),
    index_root: str = typer.Option("data/index", help="Index root"),
    output_dir: str = typer.Option("results/table_ablation", help="Output directory"),
    raw_pdf_dir: str = typer.Option("data/raw", help="Path to raw PDF directory"),
    processed_txt_dir: str = typer.Option("data/processed/text", help="Path to processed TXT directory"),
    table_format: str = typer.Option("markdown", help="Table format for extraction"),
    table_context_sentences: int = typer.Option(2, help="Context sentences around tables"),
    force_rebuild: bool = typer.Option(False, help="Force reindexing even if cache exists"),
):
    """
    Пункт 16: Тест табличного вклада.
    text-only vs text+tables.
    """
    from .experiments.runner import ExperimentRunner, run_table_ablation

    typer.echo("Running table ablation...")

    runner = ExperimentRunner(
        output_dir=output_dir,
        qa_csv_path=qa_csv,
    )

    embed_device = resolve_embed_device()
    embed_batch_size = resolve_embed_batch_size()

    def retrieval_eval_fn(config):
        from .eval.retrieval_eval import run_retrieval_evaluation

        config.embed_model = config.embed_model or "multilingual-minilm"
        config.retrieval_mode = config.retrieval_mode or "hybrid"

        config_index_root = build_index_for_config(
            config,
            index_root=index_root,
            ablation_name="table_ablation",
            raw_pdf_dir=raw_pdf_dir,
            processed_txt_dir=processed_txt_dir,
            embed_device=embed_device,
            embed_batch_size=embed_batch_size,
            table_format=table_format,
            table_context_sentences=table_context_sentences,
            force_rebuild=force_rebuild,
        )

        retriever_fn = get_retrieval_fn(
            config_index_root,
            config.embed_model,
            config.retrieval_mode,
            config.hybrid_alpha,
        )
        return run_retrieval_evaluation(
            retriever_fn,
            qa_csv,
            output_dir=output_dir,
            experiment_name=config.name,
        )

    results = run_table_ablation(runner, retrieval_eval_fn)

    typer.echo(f"Completed {len(results)} experiments")


@app.command()
def embedders(
    qa_csv: str = typer.Option("data/benchmark/qa.csv", help="Path to QA CSV"),
    index_root: str = typer.Option("data/index", help="Index root"),
    output_dir: str = typer.Option("results/embedder_comparison", help="Output directory"),
):
    """
    Пункт 5: Сравнение эмбеддеров.
    """
    from .experiments.runner import ExperimentRunner, run_embedder_comparison

    typer.echo("Running embedder comparison...")

    runner = ExperimentRunner(
        output_dir=output_dir,
        qa_csv_path=qa_csv,
    )

    def retrieval_eval_fn(config):
        from .eval.retrieval_eval import run_retrieval_evaluation

        retriever_fn = get_retrieval_fn(index_root, config.embed_model, "dense")
        return run_retrieval_evaluation(
            retriever_fn, qa_csv,
            output_dir=output_dir,
            experiment_name=config.name
        )

    results = run_embedder_comparison(runner, retrieval_eval_fn)

    typer.echo(f"Completed {len(results)} experiments")


@app.command("qa-eval")
def qa_evaluation(
    qa_csv: str = typer.Option("data/benchmark/qa.csv", help="Path to QA CSV"),
    index_root: str = typer.Option("data/index", help="Index root"),
    output_dir: str = typer.Option("results/qa", help="Output directory"),
    mode: str = typer.Option("hybrid", help="Retrieval mode"),
    use_confidence: bool = typer.Option(True, help="Use confidence/abstain"),
    config_path: str = typer.Option("config.yaml", help="Config YAML path"),
    llm_backend: Optional[str] = typer.Option(None, help="Override LLM backend"),
    llm_model: Optional[str] = typer.Option(None, help="Override LLM model"),
    bootstrap_samples: int = typer.Option(1000, help="Bootstrap samples for CI"),
    confidence_level: float = typer.Option(0.95, help="Confidence level for CI"),
):
    """
    Пункт 9: End-to-end QA evaluation.
    """
    from .eval.qa_eval_v2 import run_qa_evaluation

    typer.echo("Running QA evaluation...")

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    llm_cfg = cfg.get("llm", {})
    if llm_backend:
        llm_cfg["backend"] = llm_backend
    if llm_model:
        backend = llm_cfg.get("backend", "openai")
        if backend == "ollama":
            llm_cfg.setdefault("ollama", {})
            llm_cfg["ollama"]["model"] = llm_model
        else:
            llm_cfg.setdefault("openai", {})
            llm_cfg["openai"]["model"] = llm_model

    retriever_fn = get_retrieval_fn(index_root, "multilingual-minilm", mode)
    qa_fn = get_qa_fn(retriever_fn, llm_backend=llm_cfg.get("backend", "openai"), llm_config=llm_cfg)

    results = run_qa_evaluation(
        qa_fn, qa_csv,
        output_dir=output_dir,
        experiment_name=f"qa_{mode}",
        use_confidence=use_confidence,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )

    typer.echo("\n=== QA RESULTS ===")
    for metric, values in results.get("metrics", {}).items():
        mean = values.get("mean", 0)
        typer.echo(f"{metric}: {mean:.3f}")


@app.command("noise")
def noise_analysis(
    clean_dir: str = typer.Option("data/processed/text", help="Clean text directory"),
    noisy_dir: str = typer.Option("data/corpus_noisy", help="Noisy text directory"),
    output_dir: str = typer.Option("results/noise", help="Output directory"),
    config_path: str = typer.Option("config.yaml", help="Config YAML path"),
    noise_level: float = typer.Option(0.1, help="Target noise level for generation"),
    seed: int = typer.Option(42, help="Random seed for noise generation"),
    generate: bool = typer.Option(True, help="Generate noisy corpus before evaluation"),
    overwrite: bool = typer.Option(False, help="Overwrite noisy files if present"),
):
    """
    Шум: генерация воспроизводимого noisy-corpus + CER/WER метрики.
    """
    import yaml
    from .ingest.noise import generate_noisy_corpus
    from .eval.noise_eval import evaluate_corpus

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    buckets = cfg.get("noise", {}).get("buckets", [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.0)])
    buckets = [tuple(map(float, b)) for b in buckets]

    if generate:
        typer.echo("Generating noisy corpus...")
        count = generate_noisy_corpus(clean_dir, noisy_dir, noise_level, seed=seed, overwrite=overwrite)
        typer.echo(f"Noisy files generated: {count}")

    typer.echo("Evaluating noise metrics (CER/WER)...")
    summary = evaluate_corpus(clean_dir, noisy_dir, buckets, output_dir)
    typer.echo(f"Summary saved to {output_dir}")
    typer.echo(f"CER mean={summary['cer']['mean']:.4f}, weighted={summary['cer']['weighted']:.4f}")
    typer.echo(f"WER mean={summary['wer']['mean']:.4f}, weighted={summary['wer']['weighted']:.4f}")


@app.command("significance")
def significance_test(
    file_a: str = typer.Argument(..., help="Detailed CSV for system A"),
    file_b: str = typer.Argument(..., help="Detailed CSV for system B"),
    metric: str = typer.Option("recall@5", help="Metric column to compare"),
    n_samples: int = typer.Option(10000, help="Bootstrap samples"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """
    Парный bootstrap-тест значимости по метрике.
    """
    from .eval.significance import load_metric_map, align_metric_values, paired_bootstrap_pvalue

    a_map = load_metric_map(file_a, metric)
    b_map = load_metric_map(file_b, metric)
    a_vals, b_vals = align_metric_values(a_map, b_map)

    if not a_vals:
        raise typer.Exit(code=1)

    stats = paired_bootstrap_pvalue(a_vals, b_vals, n_samples=n_samples, seed=seed)
    typer.echo(f"diff={stats['diff']:.6f}, p-value={stats['p_value']:.6f} (n={len(a_vals)})")


@app.command("all")
def run_all(
    qa_csv: str = typer.Option("data/benchmark/qa.csv", help="Path to QA CSV"),
    index_root: str = typer.Option("data/index", help="Index root"),
    output_dir: str = typer.Option("results", help="Output directory"),
):
    """
    Запускает все эксперименты.
    """
    typer.echo("Running all experiments...")

    # 1. Baseline comparison
    typer.echo("\n[1/4] Baseline comparison...")
    baselines(qa_csv, index_root, f"{output_dir}/baseline_comparison")

    # 2. Embedder comparison
    typer.echo("\n[2/4] Embedder comparison...")
    embedders(qa_csv, index_root, f"{output_dir}/embedder_comparison")

    # 3. Table ablation
    typer.echo("\n[3/4] Table ablation...")
    tables(qa_csv, index_root, f"{output_dir}/table_ablation")

    # 4. QA evaluation
    typer.echo("\n[4/4] QA evaluation...")
    qa_evaluation(qa_csv, index_root, f"{output_dir}/qa", "hybrid", True)

    typer.echo("\n=== ALL EXPERIMENTS COMPLETE ===")
    typer.echo(f"Results saved to {output_dir}")


if __name__ == "__main__":
    app()
