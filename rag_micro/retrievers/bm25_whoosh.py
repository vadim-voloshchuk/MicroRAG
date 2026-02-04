from __future__ import annotations
import os, json
from typing import List, Dict
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.qparser import MultifieldParser
from whoosh.analysis import StemmingAnalyzer
from whoosh import qparser
from tqdm import tqdm

_INDEX_CACHE: dict[str, index.Index] = {}

def build_whoosh_index(corpus_jsonl: str, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    schema = Schema(
        id=ID(stored=True, unique=True),
        doc=TEXT(stored=True),
        page=NUMERIC(stored=True, numtype=int, signed=False),
        text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )
    # recreate index dir
    for f in os.listdir(index_dir):
        try:
            os.remove(os.path.join(index_dir, f))
        except IsADirectoryError:
            pass
    ix = index.create_in(index_dir, schema)
    writer = ix.writer(limitmb=512, procs=1, multisegment=True)
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Whoosh add"):
            rec = json.loads(line)
            writer.add_document(
                id=rec["id"],
                doc=rec["doc"],
                page=int(rec["page"] or 0),
                text=rec["text"],
            )
    writer.commit()
    return index_dir

def search_whoosh(index_dir: str, query: str, k: int = 5) -> List[Dict]:
    ix = _INDEX_CACHE.get(index_dir)
    if ix is None:
        ix = index.open_dir(index_dir)
        _INDEX_CACHE[index_dir] = ix
    parser = MultifieldParser(["text", "doc"], schema=ix.schema, group=qparser.OrGroup)
    q = parser.parse(query)
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
