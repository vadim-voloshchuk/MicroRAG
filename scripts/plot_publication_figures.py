#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("results/figures_publication")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_latest(pattern: str) -> Optional[Path]:
    files = list(Path("results").glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(metric: Dict[str, Any]) -> Optional[float]:
    if not metric:
        return None
    return metric.get("mean")


def plot_bar_with_ci(
    labels: List[str],
    means: List[float],
    ci_low: List[float],
    ci_high: List[float],
    title: str,
    ylabel: str,
    filename: str,
    colors: Optional[List[str]] = None,
):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(labels))
    err_low = np.array(means) - np.array(ci_low)
    err_high = np.array(ci_high) - np.array(means)
    err = [err_low, err_high]

    ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor="black", capsize=3, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{filename}.{ext}", dpi=300)
    plt.close(fig)


def plot_alpha_ablation(alpha_vals: List[float], metric_vals: List[float], filename: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alpha_vals, metric_vals, marker="o", linewidth=1.5)
    ax.set_title("Alpha Ablation (R@10)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("R@10")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{filename}.{ext}", dpi=300)
    plt.close(fig)


def plot_time_vs_metric(labels: List[str], times: List[float], metric_vals: List[float], filename: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(times, metric_vals, s=40)
    for label, x, y in zip(labels, times, metric_vals):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("R@10")
    ax.set_title("Quality vs Runtime")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{filename}.{ext}", dpi=300)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, strategy: str, metric: str, filename: str):
    subset = df[df["chunk_strategy"] == strategy]
    if subset.empty:
        return
    pivot = subset.pivot(index="chunk_overlap", columns="chunk_size", values=metric)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist())
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Overlap")
    ax.set_title(f"{strategy}: {metric}")

    # annotate values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if math.isnan(val):
                continue
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{filename}.{ext}", dpi=300)
    plt.close(fig)


def plot_best_strategy_bar(df: pd.DataFrame, metric: str, filename: str):
    rows = []
    for strategy in sorted(df["chunk_strategy"].dropna().unique().tolist()):
        sub = df[df["chunk_strategy"] == strategy]
        if sub.empty:
            continue
        best = sub.loc[sub[metric].idxmax()]
        rows.append((strategy, best[metric]))
    if not rows:
        return
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color="#4c78a8", edgecolor="black", linewidth=0.5)
    ax.set_title(f"Best {metric} by Strategy")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{filename}.{ext}", dpi=300)
    plt.close(fig)


def plot_table_ablation(df: pd.DataFrame, metric: str, filename: str):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = df["name"].tolist()
    vals = df[metric].tolist()
    ax.bar(labels, vals, color="#59a14f", edgecolor="black", linewidth=0.5)
    ax.set_title(f"Tables Ablation: {metric}")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{filename}.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    matplotlib.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # SOTA results
    sota_path = None
    for candidate in [
        Path("results/extended_sota_experiments/results.json"),
        Path("results/sota_experiments_full/results.json"),
        Path("results/sota_experiments/results.json"),
    ]:
        if candidate.exists():
            sota_path = candidate
            break

    if sota_path:
        data = load_json(sota_path)
        rows = []
        for item in data:
            metrics = item.get("metrics", {}) or {}
            r10 = metrics.get("recall@10", {})
            if not r10:
                continue
            rows.append({
                "name": item.get("name"),
                "category": item.get("category"),
                "r10_mean": r10.get("mean", 0.0),
                "r10_lo": r10.get("ci_lower", r10.get("mean", 0.0)),
                "r10_hi": r10.get("ci_upper", r10.get("mean", 0.0)),
                "r5_mean": metrics.get("recall@5", {}).get("mean", 0.0),
                "r5_lo": metrics.get("recall@5", {}).get("ci_lower", 0.0),
                "r5_hi": metrics.get("recall@5", {}).get("ci_upper", 0.0),
                "mrr5_mean": metrics.get("mrr@5", {}).get("mean", 0.0),
                "ndcg5_mean": metrics.get("ndcg@5", {}).get("mean", 0.0),
                "time": item.get("timing_sec", 0.0),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["category", "r10_mean"], ascending=[True, False])
            color_map = {
                "baseline": "#4c78a8",
                "fusion": "#f58518",
                "reranker": "#54a24b",
                "ablation": "#b279a2",
            }
            colors = [color_map.get(c, "#888888") for c in df["category"].tolist()]

            plot_bar_with_ci(
                df["name"].tolist(),
                df["r10_mean"].tolist(),
                df["r10_lo"].tolist(),
                df["r10_hi"].tolist(),
                "R@10 by Method",
                "R@10",
                "sota_r10_by_method",
                colors,
            )

            plot_bar_with_ci(
                df["name"].tolist(),
                df["r5_mean"].tolist(),
                df["r5_lo"].tolist(),
                df["r5_hi"].tolist(),
                "R@5 by Method",
                "R@5",
                "sota_r5_by_method",
                colors,
            )

            plot_time_vs_metric(
                df["name"].tolist(),
                df["time"].tolist(),
                df["r10_mean"].tolist(),
                "sota_time_vs_r10",
            )

            # Alpha ablation
            alpha_rows = []
            for item in data:
                name = item.get("name", "")
                if name.startswith("α="):
                    try:
                        alpha = float(name.replace("α=", ""))
                    except ValueError:
                        continue
                    r10 = item.get("metrics", {}).get("recall@10", {}).get("mean")
                    if r10 is None:
                        continue
                    alpha_rows.append((alpha, r10))
            if alpha_rows:
                alpha_rows.sort(key=lambda x: x[0])
                plot_alpha_ablation(
                    [a for a, _ in alpha_rows],
                    [v for _, v in alpha_rows],
                    "sota_alpha_ablation",
                )

    # Chunking ablation
    chunk_path = pick_latest("**/chunking_ablation.csv")
    if chunk_path:
        df_chunk = pd.read_csv(chunk_path)
        metric = "ret_doc_hit_rate"
        for strategy in ["fixed_tokens", "sentence_aware", "section_aware"]:
            plot_heatmap(
                df_chunk,
                strategy,
                metric,
                f"chunking_heatmap_{strategy}_{metric}",
            )
        plot_best_strategy_bar(df_chunk, metric, "chunking_best_strategy_doc_hit_rate")

    # Tables ablation
    table_path = pick_latest("**/table_ablation.csv")
    if table_path:
        df_tab = pd.read_csv(table_path)
        plot_table_ablation(df_tab, "ret_doc_hit_rate", "tables_doc_hit_rate")

    print(f"Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
