from __future__ import annotations
import csv
import json
import os
from typing import Dict, List, Tuple

from ..ingest.chunker_v2 import iter_pages_from_txt


def _levenshtein_distance(a: List[str], b: List[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def compute_cer(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    if not reference:
        return (0.0, 0, 0) if not hypothesis else (1.0, len(hypothesis), 0)
    edits = _levenshtein_distance(list(reference), list(hypothesis))
    return edits / len(reference), edits, len(reference)


def compute_wer(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return (0.0, 0, 0) if not hyp_tokens else (1.0, len(hyp_tokens), 0)
    edits = _levenshtein_distance(ref_tokens, hyp_tokens)
    return edits / len(ref_tokens), edits, len(ref_tokens)


def _parse_pages(text: str) -> Dict[int, str]:
    pages = list(iter_pages_from_txt(text))
    if not pages:
        return {0: text}
    return {p["page"]: p.get("text", "") for p in pages}


def evaluate_pair(clean_text: str, noisy_text: str) -> List[Dict]:
    clean_pages = _parse_pages(clean_text)
    noisy_pages = _parse_pages(noisy_text)

    metrics = []
    for page, clean_page in clean_pages.items():
        noisy_page = noisy_pages.get(page, "")
        cer, cer_edits, cer_len = compute_cer(clean_page, noisy_page)
        wer, wer_edits, wer_len = compute_wer(clean_page, noisy_page)
        metrics.append(
            {
                "page": page,
                "cer": cer,
                "wer": wer,
                "cer_edits": cer_edits,
                "cer_len": cer_len,
                "wer_edits": wer_edits,
                "wer_len": wer_len,
            }
        )
    return metrics


def _bucketize(values: List[float], buckets: List[Tuple[float, float]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for low, high in buckets:
        key = f"{low:.2f}-{high:.2f}"
        counts[key] = 0

    for val in values:
        for low, high in buckets:
            if low <= val < high or (val == high and high == buckets[-1][1]):
                key = f"{low:.2f}-{high:.2f}"
                counts[key] += 1
                break
    return counts


def evaluate_corpus(
    clean_dir: str,
    noisy_dir: str,
    buckets: List[Tuple[float, float]],
    output_dir: str,
) -> Dict:
    clean_dir = os.path.abspath(clean_dir)
    noisy_dir = os.path.abspath(noisy_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    page_rows = []
    cer_values = []
    wer_values = []
    total_cer_edits = 0
    total_cer_len = 0
    total_wer_edits = 0
    total_wer_len = 0

    for root, _, files in os.walk(clean_dir):
        for name in files:
            if not name.endswith(".txt"):
                continue
            rel_path = os.path.relpath(os.path.join(root, name), clean_dir)
            clean_path = os.path.join(clean_dir, rel_path)
            noisy_path = os.path.join(noisy_dir, rel_path)
            if not os.path.exists(noisy_path):
                continue

            with open(clean_path, "r", encoding="utf-8", errors="ignore") as f:
                clean_text = f.read()
            with open(noisy_path, "r", encoding="utf-8", errors="ignore") as f:
                noisy_text = f.read()

            metrics = evaluate_pair(clean_text, noisy_text)
            for m in metrics:
                m["doc"] = rel_path
                page_rows.append(m)
                cer_values.append(m["cer"])
                wer_values.append(m["wer"])
                total_cer_edits += m["cer_edits"]
                total_cer_len += m["cer_len"]
                total_wer_edits += m["wer_edits"]
                total_wer_len += m["wer_len"]

    summary = {
        "cer": {
            "mean": sum(cer_values) / len(cer_values) if cer_values else 0.0,
            "weighted": (total_cer_edits / total_cer_len) if total_cer_len else 0.0,
            "buckets": _bucketize(cer_values, buckets),
        },
        "wer": {
            "mean": sum(wer_values) / len(wer_values) if wer_values else 0.0,
            "weighted": (total_wer_edits / total_wer_len) if total_wer_len else 0.0,
            "buckets": _bucketize(wer_values, buckets),
        },
        "pages": len(page_rows),
    }

    json_path = os.path.join(output_dir, "noise_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pages": page_rows}, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "noise_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc",
                "page",
                "cer",
                "wer",
                "cer_edits",
                "cer_len",
                "wer_edits",
                "wer_len",
            ],
        )
        writer.writeheader()
        for row in page_rows:
            writer.writerow(row)

    return summary
