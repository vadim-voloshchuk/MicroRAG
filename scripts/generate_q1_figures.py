#!/usr/bin/env python3
"""
Q1 Article Figures Generator
Generates beautiful publication-ready visualizations for SOTA RAG results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import pandas as pd
from collections import Counter

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - modern and professional
COLORS = {
    'bm25': '#3498db',      # Blue
    'dense': '#e74c3c',     # Red
    'hybrid_03': '#9b59b6', # Purple
    'hybrid_05': '#2ecc71', # Green (SOTA highlight)
    'hybrid_07': '#f39c12', # Orange
    'accent': '#1abc9c',    # Teal
    'dark': '#2c3e50',      # Dark gray
    'light': '#ecf0f1',     # Light gray
}

# Gradient colors for bars
GRADIENT_COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']

def load_sota_results():
    """Load SOTA experiment results."""
    results_path = Path(__file__).parent.parent / 'results' / 'sota_experiments' / 'results.json'
    with open(results_path, 'r') as f:
        return json.load(f)

def load_qa_data():
    """Load QA benchmark data."""
    qa_path = Path(__file__).parent.parent / 'data' / 'benchmark' / 'qa.csv'
    return pd.read_csv(qa_path)

def create_output_dir():
    """Create output directory for figures."""
    output_dir = Path(__file__).parent.parent / 'figures' / 'q1_article'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# ============================================================
# Figure 1: Main SOTA Comparison Bar Chart
# ============================================================
def fig1_sota_comparison(results, output_dir):
    """Create main SOTA comparison bar chart with Recall@5."""
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = ['BM25\n(Lexical)', 'Dense\n(Semantic)', 'Hybrid\n(α=0.3)', 'Hybrid\n(α=0.5)', 'Hybrid\n(α=0.7)']
    colors = [COLORS['bm25'], COLORS['dense'], COLORS['hybrid_03'], COLORS['hybrid_05'], COLORS['hybrid_07']]

    recall5 = [r['metrics']['recall@5']['mean'] * 100 for r in results]
    std5 = [r['metrics']['recall@5']['std'] * 100 for r in results]

    # Create bars with gradient effect
    bars = ax.bar(methods, recall5, color=colors, edgecolor='white', linewidth=2, width=0.7)

    # Add error bars
    ax.errorbar(methods, recall5, yerr=std5, fmt='none', color='#2c3e50', capsize=5, capthick=2, linewidth=2)

    # Highlight SOTA (Hybrid α=0.5)
    bars[3].set_edgecolor('#27ae60')
    bars[3].set_linewidth(4)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, recall5, std5)):
        height = bar.get_height()
        label = f'{val:.1f}%'
        if i == 3:  # SOTA
            label = f'★ {val:.1f}%'
            ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', fontsize=14, fontweight='bold',
                       color='#27ae60')
        else:
            ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', va='bottom', fontsize=12, fontweight='medium')

    # Styling
    ax.set_ylabel('Recall@5 (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Retrieval Method', fontsize=14, fontweight='bold')
    ax.set_title('SOTA Retrieval Performance Comparison\nRAG Micro v2.0 on Microcontroller Datasheets',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.axhline(y=85.6, color='#27ae60', linestyle='--', alpha=0.7, linewidth=1.5, label='SOTA: 85.6%')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['hybrid_05'], edgecolor='#27ae60', linewidth=2, label='SOTA: Hybrid (α=0.5)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, shadow=True)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_sota_comparison.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig1_sota_comparison.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 1: SOTA Comparison saved")

# ============================================================
# Figure 2: Recall@K Performance Curves
# ============================================================
def fig2_recall_at_k(results, output_dir):
    """Create Recall@K performance curves for all methods."""
    fig, ax = plt.subplots(figsize=(11, 7))

    k_values = [1, 3, 5, 10]
    method_names = ['BM25', 'Dense', 'Hybrid (α=0.3)', 'Hybrid (α=0.5)', 'Hybrid (α=0.7)']
    colors = [COLORS['bm25'], COLORS['dense'], COLORS['hybrid_03'], COLORS['hybrid_05'], COLORS['hybrid_07']]
    markers = ['o', 's', '^', 'D', 'v']
    linewidths = [2, 2, 2, 3.5, 2]

    for i, (result, name, color, marker, lw) in enumerate(zip(results, method_names, colors, markers, linewidths)):
        recalls = [result['metrics'][f'recall@{k}']['mean'] * 100 for k in k_values]

        # Special styling for SOTA
        if i == 3:  # Hybrid α=0.5
            ax.plot(k_values, recalls, color=color, marker=marker, markersize=12,
                   linewidth=lw, label=f'★ {name} (SOTA)', linestyle='-',
                   markeredgecolor='white', markeredgewidth=2, zorder=10)
        else:
            ax.plot(k_values, recalls, color=color, marker=marker, markersize=9,
                   linewidth=lw, label=name, linestyle='-', alpha=0.85,
                   markeredgecolor='white', markeredgewidth=1.5)

    # Styling
    ax.set_xlabel('Top-K Results', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=14, fontweight='bold')
    ax.set_title('Retrieval Performance vs. Top-K\nDocument Retrieval Effectiveness at Different Cutoffs',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(k_values)
    ax.set_ylim(40, 100)
    ax.set_xlim(0.5, 10.5)

    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add annotation for K=5 sweet spot
    ax.axvline(x=5, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('K=5 Sweet Spot', xy=(5, 42), fontsize=10, ha='center', color='gray')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_recall_at_k.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig2_recall_at_k.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 2: Recall@K Curves saved")

# ============================================================
# Figure 3: Alpha Parameter Sensitivity Analysis
# ============================================================
def fig3_alpha_sensitivity(results, output_dir):
    """Create alpha parameter sensitivity analysis chart."""
    fig, ax = plt.subplots(figsize=(10, 7))

    alphas = [0.3, 0.5, 0.7]
    metrics_names = ['Recall@5', 'MRR@5', 'nDCG@5']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    markers = ['o', 's', '^']

    # Extract hybrid results
    hybrid_results = results[2:5]  # Hybrid α=0.3, 0.5, 0.7

    for metric_key, name, color, marker in zip(['recall@5', 'mrr@5', 'ndcg@5'],
                                                metrics_names, colors, markers):
        values = [r['metrics'][metric_key]['mean'] * 100 for r in hybrid_results]
        ax.plot(alphas, values, color=color, marker=marker, markersize=12,
               linewidth=3, label=name, markeredgecolor='white', markeredgewidth=2)

    # Highlight optimal alpha
    ax.axvline(x=0.5, color='#27ae60', linestyle='--', alpha=0.8, linewidth=2, label='Optimal α')
    ax.fill_between([0.45, 0.55], 0, 100, alpha=0.1, color='#27ae60')

    # Styling
    ax.set_xlabel('Alpha (α) - BM25 vs Dense Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Hybrid Search Alpha Parameter Sensitivity\nBalancing Lexical (BM25) and Semantic (Dense) Retrieval',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(alphas)
    ax.set_xticklabels(['0.3\n(Dense bias)', '0.5\n(Balanced)', '0.7\n(BM25 bias)'])
    ax.set_ylim(50, 95)
    ax.set_xlim(0.2, 0.8)

    # Legend
    ax.legend(loc='lower center', frameon=True, fancybox=True, shadow=True, ncol=4)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add annotation
    ax.annotate('OPTIMAL', xy=(0.5, 86), fontsize=11, ha='center',
               fontweight='bold', color='#27ae60')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_alpha_sensitivity.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig3_alpha_sensitivity.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 3: Alpha Sensitivity saved")

# ============================================================
# Figure 4: Comprehensive Metrics Heatmap
# ============================================================
def fig4_metrics_heatmap(results, output_dir):
    """Create comprehensive metrics heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))

    methods = ['BM25', 'Dense', 'Hybrid\n(α=0.3)', 'Hybrid\n(α=0.5)', 'Hybrid\n(α=0.7)']
    metrics = ['recall@1', 'recall@3', 'recall@5', 'recall@10',
               'mrr@5', 'mrr@10', 'ndcg@5', 'ndcg@10']
    metric_labels = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
                    'MRR@5', 'MRR@10', 'nDCG@5', 'nDCG@10']

    # Build data matrix
    data = np.array([[r['metrics'][m]['mean'] * 100 for m in metrics] for r in results])

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Score (%)', rotation=-90, va="bottom", fontsize=12, fontweight='bold')

    # Set ticks
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(methods)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics)):
            value = data[i, j]
            text_color = 'white' if value < 60 or value > 85 else 'black'
            fontweight = 'bold' if i == 3 else 'normal'  # Highlight SOTA row
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   color=text_color, fontsize=10, fontweight=fontweight)

    # Highlight SOTA row
    ax.add_patch(plt.Rectangle((-0.5, 3-0.5), len(metrics), 1, fill=False,
                               edgecolor='#27ae60', linewidth=4))

    ax.set_title('Comprehensive Retrieval Metrics Comparison\nAll Methods × All Evaluation Metrics',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Retrieval Method', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_metrics_heatmap.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig4_metrics_heatmap.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 4: Metrics Heatmap saved")

# ============================================================
# Figure 5: Radar/Spider Chart
# ============================================================
def fig5_radar_chart(results, output_dir):
    """Create radar chart comparing methods across metrics."""
    # Metrics for radar
    metrics = ['recall@5', 'recall@10', 'mrr@5', 'ndcg@5', 'ndcg@10']
    metric_labels = ['Recall@5', 'Recall@10', 'MRR@5', 'nDCG@5', 'nDCG@10']

    # Number of variables
    N = len(metrics)

    # Compute angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    method_names = ['BM25', 'Dense', 'Hybrid (α=0.5)']
    colors = [COLORS['bm25'], COLORS['dense'], COLORS['hybrid_05']]
    indices = [0, 1, 3]  # BM25, Dense, Hybrid α=0.5

    for idx, name, color in zip(indices, method_names, colors):
        values = [results[idx]['metrics'][m]['mean'] * 100 for m in metrics]
        values += values[:1]  # Close the loop

        # Plot
        linewidth = 4 if idx == 3 else 2.5
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=name, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15 if idx != 3 else 0.25, color=color)

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')

    # Set y-axis
    ax.set_ylim(40, 100)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_yticklabels(['50%', '60%', '70%', '80%', '90%', '100%'], fontsize=9)

    # Title and legend
    ax.set_title('Multi-Metric Performance Comparison\nRadar View of Key Retrieval Metrics',
                fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True,
             fancybox=True, shadow=True, fontsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_radar_chart.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig5_radar_chart.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 5: Radar Chart saved")

# ============================================================
# Figure 6: Dataset Statistics - Category Distribution
# ============================================================
def fig6_dataset_categories(qa_data, output_dir):
    """Create dataset category distribution pie/donut chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Category distribution
    category_counts = qa_data['category'].value_counts()

    # Colors for categories
    category_colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))

    # Donut chart for categories
    wedges1, texts1, autotexts1 = ax1.pie(category_counts.values,
                                          labels=category_counts.index.str.upper(),
                                          autopct='%1.1f%%',
                                          colors=category_colors,
                                          pctdistance=0.75,
                                          wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
                                          textprops={'fontsize': 9, 'fontweight': 'bold'})

    # Make percentage text smaller
    for autotext in autotexts1:
        autotext.set_fontsize(8)
        autotext.set_fontweight('medium')

    # Center text
    ax1.text(0, 0, f'{len(qa_data)}\nQ&A Pairs', ha='center', va='center',
            fontsize=18, fontweight='bold', color=COLORS['dark'])
    ax1.set_title('Question Categories Distribution\nTechnical Documentation Topics',
                 fontsize=14, fontweight='bold', pad=10)

    # Difficulty distribution
    difficulty_counts = qa_data['difficulty'].value_counts()
    difficulty_colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    colors = [difficulty_colors[d] for d in difficulty_counts.index]

    wedges2, texts2, autotexts2 = ax2.pie(difficulty_counts.values,
                                          labels=[d.upper() for d in difficulty_counts.index],
                                          autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(qa_data))})',
                                          colors=colors,
                                          pctdistance=0.65,
                                          wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
                                          textprops={'fontsize': 12, 'fontweight': 'bold'},
                                          explode=[0.02, 0.02, 0.05])

    for autotext in autotexts2:
        autotext.set_fontsize(10)
        autotext.set_fontweight('medium')

    ax2.text(0, 0, 'Difficulty\nLevels', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax2.set_title('Question Difficulty Distribution\nBenchmark Challenge Levels',
                 fontsize=14, fontweight='bold', pad=10)

    plt.suptitle('RAG Micro Benchmark Dataset Statistics\n201 Expert-Curated Q&A Pairs from 18 Microcontroller Datasheets',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_dataset_statistics.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig6_dataset_statistics.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 6: Dataset Statistics saved")

# ============================================================
# Figure 7: Performance vs Latency Trade-off
# ============================================================
def fig7_performance_latency(results, output_dir):
    """Create performance vs latency scatter plot."""
    fig, ax = plt.subplots(figsize=(11, 8))

    method_names = ['BM25', 'Dense', 'Hybrid (α=0.3)', 'Hybrid (α=0.5)', 'Hybrid (α=0.7)']
    colors = [COLORS['bm25'], COLORS['dense'], COLORS['hybrid_03'], COLORS['hybrid_05'], COLORS['hybrid_07']]

    for i, (result, name, color) in enumerate(zip(results, method_names, colors)):
        recall = result['metrics']['recall@5']['mean'] * 100
        latency = result['timing_sec']

        # Size based on whether it's SOTA
        size = 400 if i == 3 else 200
        marker = '*' if i == 3 else 'o'

        ax.scatter(latency, recall, c=color, s=size, marker=marker,
                  label=name, edgecolors='white', linewidths=2, zorder=5)

        # Add labels
        offset = (0.5, 2) if i != 3 else (0.5, 3)
        fontweight = 'bold' if i == 3 else 'normal'
        ax.annotate(name, (latency, recall), textcoords='offset points',
                   xytext=offset, fontsize=11, fontweight=fontweight)

    # Pareto frontier line (approximate)
    pareto_x = [11.3, 24.2]
    pareto_y = [77.1, 85.6]
    ax.plot(pareto_x, pareto_y, 'g--', alpha=0.5, linewidth=2, label='Pareto Frontier')

    # Styling
    ax.set_xlabel('Latency (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall@5 (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance vs Latency Trade-off\nFinding the Optimal Balance',
                fontsize=16, fontweight='bold', pad=20)

    # Add quadrant annotations
    ax.axvline(x=20, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=75, color='gray', linestyle=':', alpha=0.3)

    ax.annotate('Fast & Less Accurate', xy=(12, 50), fontsize=10, color='gray', style='italic')
    ax.annotate('★ Optimal Trade-off', xy=(25, 88), fontsize=11, color='#27ae60',
               fontweight='bold')

    ax.set_xlim(5, 40)
    ax.set_ylim(45, 100)

    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig7_performance_latency.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig7_performance_latency.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 7: Performance vs Latency saved")

# ============================================================
# Figure 8: Improvement Over Baselines Bar Chart
# ============================================================
def fig8_improvement_chart(results, output_dir):
    """Create improvement over baselines grouped bar chart."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Compare SOTA vs baselines
    bm25_recall5 = results[0]['metrics']['recall@5']['mean'] * 100
    dense_recall5 = results[1]['metrics']['recall@5']['mean'] * 100
    sota_recall5 = results[3]['metrics']['recall@5']['mean'] * 100  # Hybrid α=0.5

    # Improvements
    improvement_over_bm25 = sota_recall5 - bm25_recall5
    improvement_over_dense = sota_recall5 - dense_recall5

    metrics = ['Recall@5', 'Recall@10', 'MRR@5', 'nDCG@5']
    x = np.arange(len(metrics))
    width = 0.35

    bm25_values = [results[0]['metrics'][m.lower().replace('@', '@')]['mean'] * 100
                   for m in ['recall@5', 'recall@10', 'mrr@5', 'ndcg@5']]
    sota_values = [results[3]['metrics'][m.lower().replace('@', '@')]['mean'] * 100
                   for m in ['recall@5', 'recall@10', 'mrr@5', 'ndcg@5']]

    bars1 = ax.bar(x - width/2, bm25_values, width, label='BM25 Baseline',
                   color=COLORS['bm25'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, sota_values, width, label='Hybrid SOTA (α=0.5)',
                   color=COLORS['hybrid_05'], edgecolor='white', linewidth=1.5)

    # Add improvement annotations
    for i, (bm25_val, sota_val) in enumerate(zip(bm25_values, sota_values)):
        improvement = sota_val - bm25_val
        ax.annotate(f'+{improvement:.1f}%',
                   xy=(i + width/2, sota_val),
                   xytext=(0, 8), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold', color='#27ae60')

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Metric', fontsize=14, fontweight='bold')
    ax.set_title('SOTA Improvement Over BM25 Baseline\nHybrid Search Consistently Outperforms Lexical Retrieval',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 110)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig8_improvement_chart.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig8_improvement_chart.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 8: Improvement Chart saved")

# ============================================================
# Figure 9: Document Coverage by Source
# ============================================================
def fig9_document_sources(qa_data, output_dir):
    """Create document sources bar chart."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Count questions per document
    doc_counts = qa_data['evidence_doc'].value_counts()

    # Shorten document names
    short_names = {
        'esp32-wroom-32_datasheet_en': 'ESP32-WROOM-32',
        'stm32f407vg': 'STM32F407VG',
        'rm0008-stm32f101xx-stm32f102xx-stm32f103xx-stm32f105xx-and-stm32f107xx-advanced-armbased-32bit-mcus-stmicroelectronics': 'STM32F10x (RM0008)',
        'LPC1769_68_67_66_65_64_63': 'LPC1769',
        'Rockchip_RK3588_Datasheet_V1.6-20231016': 'RK3588',
    }

    names = [short_names.get(n, n[:20]) for n in doc_counts.index]

    # Color by vendor
    vendor_colors = {
        'ESP32': '#e74c3c',
        'STM32': '#3498db',
        'LPC': '#2ecc71',
        'RK': '#9b59b6',
    }

    colors = []
    for name in names:
        if 'ESP32' in name:
            colors.append(vendor_colors['ESP32'])
        elif 'STM32' in name:
            colors.append(vendor_colors['STM32'])
        elif 'LPC' in name:
            colors.append(vendor_colors['LPC'])
        elif 'RK' in name:
            colors.append(vendor_colors['RK'])
        else:
            colors.append('#95a5a6')

    bars = ax.barh(range(len(names)), doc_counts.values, color=colors,
                   edgecolor='white', linewidth=1.5, height=0.7)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, doc_counts.values)):
        ax.text(val + 0.5, i, f'{val}', va='center', fontsize=10, fontweight='medium')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()

    ax.set_xlabel('Number of Q&A Pairs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Document Source', fontsize=14, fontweight='bold')
    ax.set_title('Benchmark Coverage by Document Source\nQ&A Distribution Across Microcontroller Datasheets',
                fontsize=16, fontweight='bold', pad=20)

    # Legend for vendors
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='Espressif (ESP32)'),
        mpatches.Patch(facecolor='#3498db', label='STMicroelectronics'),
        mpatches.Patch(facecolor='#2ecc71', label='NXP (LPC)'),
        mpatches.Patch(facecolor='#9b59b6', label='Rockchip'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
             fancybox=True, shadow=True)

    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig9_document_sources.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig9_document_sources.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 9: Document Sources saved")

# ============================================================
# Figure 10: Summary Results Table (as figure)
# ============================================================
def fig10_summary_table(results, output_dir):
    """Create a beautiful summary results table figure."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Table data
    methods = ['BM25', 'Dense', 'Hybrid (α=0.3)', 'Hybrid (α=0.5) ★', 'Hybrid (α=0.7)']
    columns = ['Method', 'Recall@5', 'Recall@10', 'MRR@5', 'nDCG@5', 'Latency (s)']

    table_data = []
    for i, (result, method) in enumerate(zip(results, methods)):
        row = [
            method,
            f"{result['metrics']['recall@5']['mean']*100:.1f}%",
            f"{result['metrics']['recall@10']['mean']*100:.1f}%",
            f"{result['metrics']['mrr@5']['mean']*100:.1f}%",
            f"{result['metrics']['ndcg@5']['mean']*100:.1f}%",
            f"{result['timing_sec']:.1f}s"
        ]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['#3498db']*6)

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Header styling
    for j in range(len(columns)):
        table[(0, j)].set_text_props(fontweight='bold', color='white')
        table[(0, j)].set_facecolor('#2c3e50')

    # Highlight SOTA row
    for j in range(len(columns)):
        table[(4, j)].set_facecolor('#d5f5e3')
        table[(4, j)].set_text_props(fontweight='bold')

    # Alternate row colors
    for i in range(1, len(methods)+1):
        if i != 4:  # Skip SOTA row
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')

    ax.set_title('SOTA Retrieval Results Summary\nRAG Micro v2.0 Benchmark on Microcontroller Documentation',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig10_summary_table.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'fig10_summary_table.svg', format='svg', facecolor='white')
    plt.close()
    print("✓ Figure 10: Summary Table saved")

# ============================================================
# Main execution
# ============================================================
def main():
    print("\n" + "="*60)
    print("🎨 Q1 Article Figures Generator")
    print("="*60 + "\n")

    # Load data
    print("Loading data...")
    results = load_sota_results()
    qa_data = load_qa_data()
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}\n")

    # Generate all figures
    print("Generating figures...\n")

    fig1_sota_comparison(results, output_dir)
    fig2_recall_at_k(results, output_dir)
    fig3_alpha_sensitivity(results, output_dir)
    fig4_metrics_heatmap(results, output_dir)
    fig5_radar_chart(results, output_dir)
    fig6_dataset_categories(qa_data, output_dir)
    fig7_performance_latency(results, output_dir)
    fig8_improvement_chart(results, output_dir)
    fig9_document_sources(qa_data, output_dir)
    fig10_summary_table(results, output_dir)

    print("\n" + "="*60)
    print("✅ All figures generated successfully!")
    print(f"📁 Output: {output_dir}")
    print("="*60 + "\n")

    # List generated files
    print("Generated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
