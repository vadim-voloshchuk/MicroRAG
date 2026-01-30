from __future__ import annotations
import os, json
from enum import Enum
from typing import Optional
import typer
from tabulate import tabulate
import time


from .config import settings
from .ingest.build_corpus import ingest
from .retrievers.bm25_whoosh import build_whoosh_index, search_whoosh
from .retrievers.faiss_dense import build_faiss_index, search_faiss
from .retrievers.hybrid import search_hybrid
from .llm.stub import answer_stub
from .llm.llama_cpp_backend import answer_llama
from .llm.openai_backend import answer_openai
from .llm.ollama_backend import answer_ollama


class Mode(str, Enum):
    bm25 = "bm25"
    dense = "dense"
    hybrid = "hybrid"

class What(str, Enum):
    qa = "qa"

app = typer.Typer(add_completion=False, help="RAG CLI (BM25 + Dense + Hybrid)")

@app.command("ingest")
def ingest_cmd(
    from_pdf: Optional[str] = typer.Option(None, help="Папка с PDF"),
    from_txt: Optional[str] = typer.Option(None, help="Папка с TXT"),
    out: str = typer.Option("data/index", help="Куда писать corpus.jsonl"),
    chunk_chars: int = typer.Option(1000, help="Размер чанка"),
    overlap: int = typer.Option(200, help="Перекрытие"),
):
    out_jsonl, n = ingest(from_pdf, from_txt, out, chunk_chars, overlap)
    typer.echo(f"Corpus: {out_jsonl}, chunks: {n}")

@app.command("index")
def index_cmd(
    build: bool = typer.Option(True, help="Собрать индексы"),
    index_root: str = typer.Option("data/index"),
    faiss: bool = typer.Option(True, help="Строить FAISS"),
    bm25: bool = typer.Option(True, help="Строить BM25"),
    embed_model: str = typer.Option(settings.embed_model, help="Модель эмбеддингов"),
):
    corpus_jsonl = os.path.join(index_root, "corpus.jsonl")
    if not os.path.exists(corpus_jsonl):
        raise SystemExit(f"Не найден {corpus_jsonl}. Сначала запустите ingest.")
    if bm25:
        build_whoosh_index(corpus_jsonl, os.path.join(index_root, "bm25"))
    if faiss:
        build_faiss_index(corpus_jsonl, os.path.join(index_root, "faiss"), embed_model)
    typer.echo("Индексы готовы.")

@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Вопрос"),
    mode: Mode = typer.Option(Mode.hybrid, help="Режим ретривера"),
    k: int = typer.Option(5, help="Сколько фрагментов в контекст"),
    index_root: str = typer.Option("data/index"),
    embed_model: str = typer.Option(settings.embed_model),
):
    t0 = time.time()
    if mode == Mode.bm25:
        chunks = search_whoosh(os.path.join(index_root, "bm25"), question, k=k)
    elif mode == Mode.dense:
        chunks = search_faiss(os.path.join(index_root, "faiss"), question, k=k, embed_model=embed_model)
    else:
        chunks = search_hybrid(index_root, question, k=k, embed_model=embed_model)


    t_retrieve = time.time() - t0

    # Вывод превью найденных
    headers = ["#", "doc", "page", "score", "retriever"]
    rows = [[i+1, c["doc"], c["page"], f"{c['score']:.3f}", c["retriever"]] for i, c in enumerate(chunks)]
    typer.echo(tabulate(rows, headers=headers, tablefmt="github"))

    # Выбор LLM
    t1 = time.time()
    if settings.llm_backend == "stub":
        ans = answer_stub(chunks, question)
    elif settings.llm_backend == "llama_cpp":
        if not settings.llama_model_path:
            raise SystemExit("LLM_BACKEND=llama_cpp, но не задан LLAMA_MODEL_PATH")
        ans = answer_llama(chunks, question, settings.llama_model_path, settings.llama_ctx, settings.llama_threads)  # type: ignore
    elif settings.llm_backend == "openai":
        ans = answer_openai(chunks, question, settings.openai_model, settings.openai_api_key, settings.openai_base_url)  # type: ignore
    elif settings.llm_backend == "ollama":
        ans = answer_ollama(chunks, question, settings.ollama_model, settings.ollama_host)

    else:
        raise SystemExit(f"Неизвестный LLM_BACKEND={settings.llm_backend}")

    t_generate = time.time() - t1
    typer.echo("\n--- Ответ ---\n")
    typer.echo(ans)
    typer.echo("\n--- Источники ---")
    typer.echo(f"\n[timings] retrieve={t_retrieve:.3f}s, generate={t_generate:.3f}s, total={t_retrieve+t_generate:.3f}s")
    for c in chunks:
        typer.echo(f"- {c['doc']}, стр.{c['page']}")

# ====== Оценка ======
from .eval.qa_eval import eval_qa

@app.command("eval")
def eval_cmd(
    what: What = typer.Argument(What.qa),
    index_root: str = typer.Option("data/index"),
    mode: Mode = typer.Option(Mode.hybrid),
    embed_model: str = typer.Option(settings.embed_model),
    qa_csv: str = typer.Option("data/benchmark/qa.csv", help="CSV с колонками: question,answer"),
    limit: int = typer.Option(0, help="Ограничение числа вопросов (0=все)"),
):
    if what != What.qa:
        raise SystemExit("Поддерживается только eval qa")
    eval_qa(index_root, mode.value, embed_model, qa_csv, limit)


if __name__ == "__main__":
    app()
