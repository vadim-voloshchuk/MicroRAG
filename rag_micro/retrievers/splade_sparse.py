from __future__ import annotations
import os
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy import sparse


def _load_corpus(corpus_jsonl: str) -> Tuple[List[Dict], List[str]]:
    items, texts = [], []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            items.append(rec)
            texts.append(rec["text"])
    return items, texts


def _splade_weights(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int,
) -> torch.Tensor:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        relu = torch.relu(logits)
        weights = torch.log1p(relu)
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(-1)
            weights = weights * mask
        weights = torch.max(weights, dim=1).values
    return weights


def build_splade_index(
    corpus_jsonl: str,
    index_dir: str,
    model_id: str = "naver/splade-v3",
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 256,
    top_k: int = 128,
) -> str:
    os.makedirs(index_dir, exist_ok=True)
    items, texts = _load_corpus(corpus_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()
    model.to(device)

    vocab_size = tokenizer.vocab_size
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        weights = _splade_weights(model, tokenizer, batch, device, max_length)
        values, indices = torch.topk(weights, k=min(top_k, weights.shape[1]), dim=1)
        values = values.cpu().numpy()
        indices = indices.cpu().numpy()

        for i, (vals, inds) in enumerate(zip(values, indices)):
            doc_id = start + i
            for val, idx in zip(vals, inds):
                if val <= 0:
                    continue
                rows.append(doc_id)
                cols.append(int(idx))
                data.append(float(val))

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(items), vocab_size), dtype=np.float32)
    sparse.save_npz(os.path.join(index_dir, "splade.npz"), matrix)

    with open(os.path.join(index_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)

    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "vocab_size": vocab_size,
                "top_k": top_k,
                "max_length": max_length,
            },
            f,
            ensure_ascii=False,
        )

    return index_dir


class SpladeRetriever:
    def __init__(
        self,
        index_dir: str,
        model_id: str,
        device: str = "cpu",
        max_length: int = 256,
    ):
        self.index_dir = index_dir
        self.device = device
        self.max_length = max_length
        self.model_id = model_id

        self.matrix = sparse.load_npz(os.path.join(index_dir, "splade.npz"))
        with open(os.path.join(index_dir, "items.json"), "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        self.model.eval()
        self.model.to(device)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        weights = _splade_weights(self.model, self.tokenizer, [query], self.device, self.max_length)
        q_vec = weights.cpu().numpy().reshape(-1)
        scores = self.matrix.dot(q_vec)
        if sparse.issparse(scores):
            scores = scores.toarray().reshape(-1)
        else:
            scores = np.asarray(scores).reshape(-1)

        if len(scores) == 0:
            return []

        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        out = []
        for idx in top_idx:
            rec = self.items[idx]
            out.append(
                {
                    "id": rec["id"],
                    "doc": rec["doc"],
                    "page": int(rec["page"] or 0),
                    "text": rec["text"],
                    "score": float(scores[idx]),
                    "retriever": "splade",
                }
            )
        return out
