from __future__ import annotations
import csv
from typing import Dict, List, Tuple
import numpy as np


def load_metric_map(csv_path: str, metric: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("query_id")
            if not query_id:
                continue
            raw = row.get(metric)
            if raw is None:
                continue
            try:
                values[query_id] = float(raw)
            except ValueError:
                continue
    return values


def align_metric_values(
    a: Dict[str, float],
    b: Dict[str, float],
) -> Tuple[List[float], List[float]]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    return [a[k] for k in keys], [b[k] for k in keys]


def paired_bootstrap_pvalue(
    a: List[float],
    b: List[float],
    n_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    if len(a) != len(b):
        raise ValueError("Input lengths must match for paired bootstrap.")
    if not a:
        return {"diff": 0.0, "p_value": 1.0}

    rng = np.random.default_rng(seed)
    a_arr = np.array(a)
    b_arr = np.array(b)
    n = len(a_arr)
    observed = float(a_arr.mean() - b_arr.mean())

    diffs = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(a_arr[idx].mean() - b_arr[idx].mean())

    if observed >= 0:
        p_value = float((diffs <= 0).mean())
    else:
        p_value = float((diffs >= 0).mean())

    return {"diff": observed, "p_value": p_value}
