from __future__ import annotations
import os, json
from typing import List, Dict, Tuple
import numpy as np
import faiss  # type: ignore
from .embedders import get_embedder

_EMBEDDER_CACHE: dict[tuple[str, str], object] = {}
_INDEX_CACHE: dict[str, faiss.Index] = {}

def _load_corpus(corpus_jsonl: str) -> Tuple[List[Dict], List[str]]:
    items, texts = [], []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            items.append(rec)
            texts.append(rec["text"])
    return items, texts

def build_faiss_index(
    corpus_jsonl: str,
    index_dir: str,
    embed_model: str,
    device: str = "cpu",
    batch_size: int | None = None,
):
    os.makedirs(index_dir, exist_ok=True)
    items, texts = _load_corpus(corpus_jsonl)
    embedder = get_embedder(embed_model, device=device)
    embs = embedder.encode_documents(
        texts,
        show_progress=True,
        batch_size=batch_size,
    )
    embs = np.asarray(embs).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if normalized
    index.add(embs)

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    # сохраняем метаданные
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "items": items,
                "embed_model": embed_model,
                "embed_model_id": embedder.info.model_id,
                "embed_dim": embedder.info.dim,
            },
            f,
            ensure_ascii=False,
        )
    return index_dir

def search_faiss(
    index_dir: str,
    query: str,
    k: int,
    embed_model: str,
    device: str = "cpu",
) -> List[Dict]:
    cache_key = (embed_model, device)
    embedder = _EMBEDDER_CACHE.get(cache_key)
    if embedder is None:
        embedder = get_embedder(embed_model, device=device)
        _EMBEDDER_CACHE[cache_key] = embedder
    q = embedder.encode_query(query)
    q = np.asarray(q, dtype="float32").reshape(1, -1)
    index = _INDEX_CACHE.get(index_dir)
    if index is None:
        index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        _INDEX_CACHE[index_dir] = index
    D, I = index.search(q, k)
    with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
        items = json.load(f)["items"]
    out = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        rec = items[idx]
        out.append({
            "id": rec["id"],
            "doc": rec["doc"],
            "page": int(rec["page"] or 0),
            "text": rec["text"],
            "score": float(score),
            "retriever": "dense",
        })
    return out
