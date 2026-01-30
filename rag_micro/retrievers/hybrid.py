from __future__ import annotations
from typing import List, Dict
from .bm25_whoosh import search_whoosh
from .faiss_dense import search_faiss

def dedup(results: List[Dict], max_items: int) -> List[Dict]:
    seen = set()
    out = []
    # сортируем по score по убыванию
    sorted_res = sorted(results, key=lambda x: x["score"], reverse=True)
    for r in sorted_res:
        if r["id"] in seen:
            continue
        out.append(r)
        seen.add(r["id"])
        if len(out) >= max_items:
            break
    return out

def search_hybrid(index_root: str, query: str, k: int, embed_model: str) -> List[Dict]:
    bm = search_whoosh(f"{index_root}/bm25", query, k=k*2)
    dn = search_faiss(f"{index_root}/faiss", query, k=k*2, embed_model=embed_model)
    mix = bm + dn
    return dedup(mix, max_items=k)
