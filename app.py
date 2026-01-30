from __future__ import annotations
import os
import time
import shutil
import logging
from logging.handlers import RotatingFileHandler
from typing import List

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_micro.config import settings
from rag_micro.ingest.build_corpus import ingest  # сигнатура: ingest(from_pdf, from_txt, out, chunk_chars, overlap)
from rag_micro.retrievers.bm25_whoosh import build_whoosh_index, search_whoosh
from rag_micro.retrievers.faiss_dense import build_faiss_index, search_faiss
from rag_micro.retrievers.hybrid import search_hybrid
from rag_micro.llm.stub import answer_stub
from rag_micro.llm.llama_cpp_backend import answer_llama
from rag_micro.llm.openai_backend import answer_openai
from rag_micro.llm.ollama_backend import answer_ollama

# ----------------------------
# CONFIG / PATHS / LOGGING
# ----------------------------
INDEX_ROOT = os.environ.get("INDEX_ROOT", "data/index")
DOCS_ROOT = os.environ.get("DOCS_ROOT", "data/docs")
os.makedirs(INDEX_ROOT, exist_ok=True)
os.makedirs(DOCS_ROOT, exist_ok=True)

ANSWER_TIMEOUT_S = float(os.environ.get("RAG_ANSWER_TIMEOUT_S", "120"))
RETRIEVE_TIMEOUT_S = float(os.environ.get("RAG_RETRIEVE_TIMEOUT_S", "60"))

LOG_PATH = os.environ.get("RAG_LOG_PATH", "micro_rag.log")
logger = logging.getLogger("micro_rag")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

# ----------------------------
# FASTAPI
# ----------------------------
app = FastAPI(title="MicroRAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        dur = (time.time() - t0) * 1000
        code = response.status_code if "response" in locals() else "-"
        logger.info(f"{request.method} {request.url.path} {code} {dur:.1f}ms")

class AskRequest(BaseModel):
    question: str
    mode: str = "hybrid"        # bm25|faiss|hybrid
    top_k: int = 5
    embed_model: str = settings.embed_model
    provider: str = "auto"      # auto|openai|ollama|llama.cpp
    temperature: float = 0.1

# ----------------------------
# HELPERS
# ----------------------------
def _ensure_indexes(embed_model: str):
    bm25_dir  = os.path.join(INDEX_ROOT, "bm25")
    faiss_dir = os.path.join(INDEX_ROOT, "faiss")
    corpus_jsonl = os.path.join(INDEX_ROOT, "corpus.jsonl")

    # 1) сборка корпуса
    if not os.path.exists(corpus_jsonl):
        logger.info(f"[ingest] from {DOCS_ROOT} -> {corpus_jsonl}")
        # ВАЖНО: позиционные аргументы, как в CLI
        ingest(DOCS_ROOT, DOCS_ROOT, INDEX_ROOT, 1000, 200)
        logger.info("[ingest] corpus ready")

    # 2) индексы
    if not os.path.exists(bm25_dir):
        logger.info("[index] build Whoosh (BM25)")
        build_whoosh_index(corpus_jsonl, bm25_dir)
        logger.info("[index] Whoosh ready")

    if not os.path.exists(faiss_dir):
        logger.info(f"[index] build FAISS (embed={embed_model})")
        build_faiss_index(corpus_jsonl, faiss_dir, embed_model)
        logger.info("[index] FAISS ready")

def _search(q: str, mode: str, k: int, embed_model: str):
    t0 = time.time()
    _ensure_indexes(embed_model)
    try:
        if mode == "bm25":
            res = search_whoosh(os.path.join(INDEX_ROOT, "bm25"), q, k=k)
        elif mode in ("faiss", "dense"):
            res = search_faiss(os.path.join(INDEX_ROOT, "faiss"), q, k=k, embed_model=embed_model)
        else:
            res = search_hybrid(INDEX_ROOT, q, k=k, embed_model=embed_model)
        logger.info(f"[retrieve] mode={mode} k={k} q='{q[:80]}' -> {len(res)} hits in {(time.time()-t0)*1000:.1f}ms")
        return res
    except Exception as e:
        logger.exception(f"[retrieve] failed: {e}")
        raise

def _answer(provider: str, question: str, passages: list, temperature: float):
    t0 = time.time()
    logger.info(f"[answer] provider={provider} chunks={len(passages)} q='{question[:80]}'")
    try:
        # сигнатуры как в CLI: chunks первым аргументом
        if provider == "openai":
            ans = answer_openai(passages, question, settings.openai_model, settings.openai_api_key, settings.openai_base_url)  # type: ignore
        elif provider == "ollama":
            ans = answer_ollama(passages, question, settings.ollama_model, settings.ollama_host)
        elif provider == "llama.cpp":
            if not settings.llama_model_path:
                ans = answer_stub(passages, question)
            else:
                ans = answer_llama(passages, question, settings.llama_model_path, settings.llama_ctx, settings.llama_threads)  # type: ignore
        else:
            if settings.openai_api_key or settings.openai_base_url:
                ans = answer_openai(passages, question, settings.openai_model, settings.openai_api_key, settings.openai_base_url)  # type: ignore
            else:
                try:
                    ans = answer_ollama(passages, question, settings.ollama_model, settings.ollama_host)
                except Exception:
                    ans = answer_stub(passages, question)

        dur = time.time() - t0
        logger.info(f"[answer] completed in {dur:.2f}s")
        if dur > ANSWER_TIMEOUT_S:
            ans = (ans or "") + f"\n\n[warn] generation took {dur:.1f}s (> {ANSWER_TIMEOUT_S}s)."
        return ans
    except Exception as e:
        logger.exception(f"[answer] failed: {e}")
        return f"[error] answer failed: {e}"

# ----------------------------
# ENDPOINTS
# ----------------------------
@app.get("/api/stats")
def stats():
    docs = len([e for e in os.scandir(DOCS_ROOT) if e.is_file()]) if os.path.exists(DOCS_ROOT) else 0
    last_build = time.strftime("%Y-%m-%d %H:%M")
    return {"docs": docs, "chunks": None, "last_build": last_build}

@app.post("/api/ingest")
async def api_ingest(files: List[UploadFile] = File(...)):
    logger.info(f"[ingest] upload {len(files)} file(s)")
    for f in files:
        dst = os.path.join(DOCS_ROOT, f.filename)
        with open(dst, "wb") as out:
            out.write(await f.read())
        logger.info(f"[ingest] saved {dst}")

    ingest(DOCS_ROOT, DOCS_ROOT, INDEX_ROOT, 1000, 200)
    corpus_jsonl = os.path.join(INDEX_ROOT, "corpus.jsonl")
    build_whoosh_index(corpus_jsonl, os.path.join(INDEX_ROOT, "bm25"))
    build_faiss_index(corpus_jsonl, os.path.join(INDEX_ROOT, "faiss"), settings.embed_model)
    logger.info("[ingest] indexes rebuilt")
    return {"message": f"Indexed {len(files)} file(s)."}

@app.get("/api/search")
def api_search(
    q: str = Query(..., description="query"),
    mode: str = Query("hybrid"),
    k: int = Query(5),
    embed_model: str = Query(settings.embed_model),
    raw: int = Query(0, description="вернуть сырые хиты (1)"),
):
    try:
        # Гарантируем, что есть корпус и индексы (и залогируем это)
        logger.info(f"/api/search q='{q[:120]}' mode={mode} k={k} embed={embed_model}")
        _ensure_indexes(embed_model)

        # Сам поиск
        hits = _search(q, mode, k, embed_model)  # ключи: doc, page, text, score, snippet?
        logger.info(f"/api/search hits={len(hits)}")

        # Нормализуем для UI
        results = [{
            "title": h.get("doc"),
            "page": h.get("page"),
            "score": h.get("score"),
            "snippet": h.get("snippet") or h.get("text"),
            "doc": h.get("doc"),
            "text": h.get("text"),
        } for h in (hits or [])]

        # при raw=1 вернём ещё и сырые хиты для проверки
        return {"results": results, **({"raw": hits} if raw else {})}

    except Exception as e:
        logger.exception(f"/api/search failed: {e}")
        # Не роняем — пусть фронт покажет ошибку, но 200 заменим на 500 только если хочешь
        # здесь вернём пусто и сообщение об ошибке
        return {"results": [], "error": str(e)}


@app.post("/api/ask")
def api_ask(req: AskRequest):
    logger.info(f"/api/ask q='{req.question[:120]}' mode={req.mode} k={req.top_k} provider={req.provider}")
    hits = _search(req.question, req.mode, req.top_k, req.embed_model)
    answer = _answer(req.provider, req.question, hits, req.temperature)
    sources = []
    for c in hits:
        sources.append({
            "title": c.get("doc"),
            "page": c.get("page"),
            "score": c.get("score"),
            "snippet": c.get("snippet") or c.get("text"),
            "doc": c.get("doc"),
            "text": c.get("text"),
        })
    return {"answer": answer, "sources": sources}

@app.post("/api/clear")
def api_clear():
    def rm(path: str):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            try: os.remove(path)
            except FileNotFoundError: pass
    rm(os.path.join(INDEX_ROOT, "bm25"))
    rm(os.path.join(INDEX_ROOT, "faiss"))
    rm(os.path.join(INDEX_ROOT, "corpus.jsonl"))
    os.makedirs(os.path.join(INDEX_ROOT, "bm25"), exist_ok=True)
    os.makedirs(os.path.join(INDEX_ROOT, "faiss"), exist_ok=True)
    logger.info("[clear] indexes removed")
    return {"ok": True}
