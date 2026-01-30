from __future__ import annotations
import os
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, BertModel
from scipy import sparse


def _load_corpus(corpus_jsonl: str) -> Tuple[List[Dict], List[str]]:
    items, texts = [], []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            items.append(rec)
            texts.append(rec["text"])
    return items, texts


def _load_unicoil_components(model_id: str, device: str):
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    state = torch.hub.load_state_dict_from_url(
        f"https://huggingface.co/{model_id}/resolve/main/pytorch_model.bin",
        map_location="cpu",
    )

    bert = BertModel(config)
    tok_proj = torch.nn.Linear(config.hidden_size, 1)

    bert_state = {
        k.replace("coil_encoder.bert.", ""): v
        for k, v in state.items()
        if k.startswith("coil_encoder.bert.")
    }
    bert.load_state_dict(bert_state, strict=False)
    tok_proj.load_state_dict(
        {
            "weight": state["coil_encoder.tok_proj.weight"],
            "bias": state["coil_encoder.tok_proj.bias"],
        }
    )

    bert.to(device)
    tok_proj.to(device)
    bert.eval()
    tok_proj.eval()

    return tokenizer, bert, tok_proj


def _aggregate_token_weights(
    input_ids: np.ndarray,
    weights: np.ndarray,
    pad_id: int,
    top_k: int,
) -> List[Dict[int, float]]:
    output: List[Dict[int, float]] = []
    for ids, wts in zip(input_ids, weights):
        bucket: Dict[int, float] = {}
        for token_id, weight in zip(ids, wts):
            if token_id == pad_id:
                continue
            if weight <= 0:
                continue
            prev = bucket.get(int(token_id))
            if prev is None or weight > prev:
                bucket[int(token_id)] = float(weight)

        if top_k and len(bucket) > top_k:
            top_items = sorted(bucket.items(), key=lambda x: x[1], reverse=True)[:top_k]
            bucket = dict(top_items)

        output.append(bucket)

    return output


def _encode_texts(
    tokenizer,
    bert,
    tok_proj,
    texts: List[str],
    device: str,
    max_length: int,
    top_k: int,
) -> List[Dict[int, float]]:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
        token_scores = tok_proj(outputs.last_hidden_state).squeeze(-1)
        token_scores = torch.relu(token_scores)
        token_scores = token_scores * attention_mask

    token_scores = token_scores.cpu().numpy()
    input_ids = input_ids.cpu().numpy()
    return _aggregate_token_weights(input_ids, token_scores, tokenizer.pad_token_id or 0, top_k)


def build_unicoil_index(
    corpus_jsonl: str,
    index_dir: str,
    model_id: str = "castorini/unicoil-msmarco-passage",
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 256,
    top_k: int = 128,
) -> str:
    os.makedirs(index_dir, exist_ok=True)
    items, texts = _load_corpus(corpus_jsonl)

    tokenizer, bert, tok_proj = _load_unicoil_components(model_id, device)
    vocab_size = tokenizer.vocab_size

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        weights = _encode_texts(
            tokenizer,
            bert,
            tok_proj,
            batch,
            device,
            max_length,
            top_k,
        )
        for i, token_map in enumerate(weights):
            doc_id = start + i
            for token_id, weight in token_map.items():
                rows.append(doc_id)
                cols.append(token_id)
                data.append(weight)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(items), vocab_size), dtype=np.float32)
    sparse.save_npz(os.path.join(index_dir, "unicoil.npz"), matrix)

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


class UniCOILRetriever:
    def __init__(
        self,
        index_dir: str,
        model_id: str,
        device: str = "cpu",
        max_length: int = 256,
        top_k: int = 128,
    ):
        self.index_dir = index_dir
        self.device = device
        self.max_length = max_length
        self.top_k = top_k
        self.model_id = model_id

        self.matrix = sparse.load_npz(os.path.join(index_dir, "unicoil.npz"))
        with open(os.path.join(index_dir, "items.json"), "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.tokenizer, self.bert, self.tok_proj = _load_unicoil_components(model_id, device)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        token_map = _encode_texts(
            self.tokenizer,
            self.bert,
            self.tok_proj,
            [query],
            self.device,
            self.max_length,
            self.top_k,
        )[0]

        if not token_map:
            return []

        q_vec = np.zeros(self.tokenizer.vocab_size, dtype=np.float32)
        for token_id, weight in token_map.items():
            q_vec[token_id] = weight

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
                    "retriever": "unicoil",
                }
            )
        return out
