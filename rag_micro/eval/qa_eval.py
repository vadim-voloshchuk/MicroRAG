from __future__ import annotations
import os, csv, re
from typing import Literal, List, Dict, Any, Tuple

from ..retrievers.bm25_whoosh import search_whoosh
from ..retrievers.faiss_dense import search_faiss
from ..retrievers.hybrid import search_hybrid
from ..llm.stub import answer_stub
from ..llm.llama_cpp_backend import answer_llama
from ..llm.openai_backend import answer_openai
from ..config import settings

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    return s

def f1_word_level(pred: str, gold: str) -> float:
    ps = normalize(pred).split()
    gs = normalize(gold).split()
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    inter = len(set(ps) & set(gs))
    if inter == 0:
        return 0.0
    prec = inter / len(set(ps))
    rec = inter / len(set(gs))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def retrieve(index_root: str, mode: Literal["bm25","dense","hybrid"], embed_model: str, q: str, k: int = 5):
    if mode == "bm25":
        return search_whoosh(os.path.join(index_root, "bm25"), q, k=k)
    elif mode == "dense":
        return search_faiss(os.path.join(index_root, "faiss"), q, k=k, embed_model=embed_model)
    else:
        return search_hybrid(index_root, q, k=k, embed_model=embed_model)

def generate(chunks, q: str) -> str:
    if settings.llm_backend == "stub":
        return answer_stub(chunks, q)
    elif settings.llm_backend == "llama_cpp":
        return answer_llama(chunks, q, settings.llama_model_path, settings.llama_ctx, settings.llama_threads)  # type: ignore
    elif settings.llm_backend == "openai":
        return answer_openai(chunks, q, settings.openai_model, settings.openai_api_key, settings.openai_base_url)  # type: ignore
    elif settings.llm_backend == "ollama":
        return answer_ollama(chunks, q, settings.ollama_model, settings.ollama_host)
    else:
        raise RuntimeError("Unknown backend")

def support_ratio(answer: str, chunks: List[Dict]) -> float:
    """Оценка доли токенов ответа, покрытых текстом источников (очень грубо)."""
    ctx = " ".join([c["text"] for c in chunks])
    a = set(normalize(answer).split())
    c = set(normalize(ctx).split())
    if not a:
        return 1.0
    inter = len(a & c)
    return inter / max(1, len(a))

def eval_qa(index_root: str, mode: str, embed_model: str, qa_csv: str, limit: int = 0):
    # Читаем вопросы
    rows = []
    with open(qa_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    if limit > 0:
        rows = rows[:limit]

    em_cnt, f1_sum = 0, 0.0
    sup_sum, halluc_cnt = 0.0, 0
    # для Retrieval-метрик
    r5_hits = 0
    mrr_sum = 0.0
    rN = 5

    for i, r in enumerate(rows, 1):
        q, gold = r["question"], r["answer"]
        gold_doc = r.get("doc")  # опционально
        gold_page = r.get("page")

        chunks = retrieve(index_root, mode, embed_model, q, k=rN)
        pred = generate(chunks, q)
        first_line = pred.splitlines()[0].strip()

        # EM/F1
        em = int(normalize(first_line) == normalize(gold))
        # F1 по словам
        f1 = f1_word_level(first_line, gold)
        em_cnt += em
        f1_sum += f1

        # простая проверка поддержки и "галлюцинаций"
        sr = support_ratio(first_line, chunks)
        sup_sum += sr
        if sr < 0.3:
            halluc_cnt += 1

        # Recall@5 / MRR если есть эталонный документ
        if gold_doc:
            # найдём позицию первого фрагмента из нужного документа (и, если есть, совп. страницы)
            rank = None
            for idx, c in enumerate(chunks, 1):
                if c["doc"] == gold_doc and (not gold_page or str(c["page"]) == str(gold_page)):
                    rank = idx
                    break
            if rank is not None:
                r5_hits += 1
                mrr_sum += 1.0 / rank

        print(f"[{i}/{len(rows)}] EM={em} F1={f1:.3f} :: {q}")
        print(f"pred: {first_line}")
        print(f"gold: {gold}")
        if gold_doc:
            print(f"gold source: {gold_doc}#{gold_page}")
        print()

    n = len(rows) if rows else 1

    print("=== SUMMARY ===")
    print(f"Exact Match: {100*em_cnt/n:.1f}%")
    print(f"F1: {100*(f1_sum/n):.1f}%")
    print(f"Support ratio (avg): {100*(sup_sum/n):.1f}%")
    print(f"Hallucinations (heuristic, <30% support): {100*(halluc_cnt/n):.1f}%")

    if any(r.get('doc') for r in rows):
        print(f"Recall@{rN}: {100*(r5_hits/n):.1f}%")
        print(f"MRR: {(mrr_sum/n):.3f}")
