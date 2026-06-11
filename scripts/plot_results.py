#!/usr/bin/env python3
"""
plot_results.py — 读取 CSV 并绘制 GEMM 性能基准测试可视化图表

用法：
    python plot_results.py
    python plot_results.py --input results.csv --output plots/

依赖：
    pip install matplotlib pandas
"""

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # 无头环境，避免 $DISPLAY 报错
    import matplotlib.pyplot as plt
except ImportError:
    print("[Error] matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


ALGORITHM_ORDER = [
    'CPU-Baseline',
    'GPU-Naive',
    'GPU-Tiled',
    'GPU-Tiled-Padded',
    'GPU-Tiled-Coarse',
    'GPU-cuBLAS',
]

ALGORITHM_COLORS = {
    'CPU-Baseline':      '#888888',
    'GPU-Naive': '#e74c3c',
    'GPU-Tiled': '#3498db',
    'GPU-Tiled-Padded':  '#2ecc71',
    'GPU-Tiled-Coarse':  '#9b59b6',
    'GPU-cuBLAS':        '#f39c12',
}


def load_csv(path):
    """加载 CSV 文件为 records列表"""
    records = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 类型转换
            row['M'] = int(row['M'])
            row['N'] = int(row['N'])
            row['K'] = int(row['K'])
            row['time_ms'] = float(row['time_ms']) if row['time_ms'] else None
            row['gflops'] = float(row['gflops']) if row['gflops'] else None
            row['speedup'] = float(row['speedup']) if row['speedup'] else None
            records.append(row)
    return records


def size_label(r):
    """生成横轴标签：'1024³' 或 '4096×4096×128'"""
    if r['M'] == r['N'] == r['K']:
        return f"{r['M']}³"
    return f"{r['M']}×{r['N']}×{r['K']}"


def pivot_by_size(records, field):
    """按矩阵大小分组，返回 {size_label: {algorithm: value}}"""
    pivot = {}
    for r in records:
        key = size_label(r)
        algo = r['algorithm']
        val = r.get(field)
        if key not in pivot:
            pivot[key] = {}
        pivot[key][algo] = val
    return pivot


def sorted_sizes(pivot):
    """按数值大小排序矩阵大小标签"""
    def sort_key(k):
        #提取第一个数字用于排序
        m = k.split('×')[0].split('³')[0]
        return int(m)
    return sorted(pivot.keys(), key=sort_key)


# ─────────────────────────────────────────────────────────────────────────────
# 图表 1：GFLOPS 分组柱状图
# ─────────────────────────────────────────────────────────────────────────────
def plot_gflops_bar(records, out_dir):
    pivot = pivot_by_size(records, 'gflops')
    sizes = sorted_sizes(pivot)
    algorithms = ALGORITHM_ORDER

    n_sizes = len(sizes)
    n_algos = len(algorithms)
    bar_width = 0.12
    x = range(n_sizes)

    fig, ax = plt.subplots(figsize=(max(10, n_sizes * 1.8), 6))
    for i, algo in enumerate(algorithms):
        values = [pivot[s].get(algo) or 0 for s in sizes]
        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar([xi + offset for xi in x], values, bar_width *0.9,
                      label=algo, color=ALGORITHM_COLORS.get(algo, None))
        # 标注数值
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7, rotation=30)

    ax.set_xlabel('Matrix Size (M×N×K)')
    ax.set_ylabel('GFLOPS')
    ax.set_title('GEMM Performance — GFLOPS by Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, 'benchmark_gflops_bar.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 图表 2：执行时间分组柱状图
# ─────────────────────────────────────────────────────────────────────────────
def plot_time_bar(records, out_dir):
    pivot = pivot_by_size(records, 'time_ms')
    sizes = sorted_sizes(pivot)
    algorithms = ALGORITHM_ORDER

    n_sizes = len(sizes)
    n_algos = len(algorithms)
    bar_width = 0.12
    x = range(n_sizes)

    fig, ax = plt.subplots(figsize=(max(10, n_sizes * 1.8), 6))
    for i, algo in enumerate(algorithms):
        values = [pivot[s].get(algo) or 0 for s in sizes]
        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar([xi + offset for xi in x], values, bar_width * 0.9,
                      label=algo, color=ALGORITHM_COLORS.get(algo, None))
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=30)

    ax.set_xlabel('Matrix Size (M×N×K)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('GEMM Performance — Execution Time by Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, 'benchmark_time_bar.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 图表 3：GFLOPS 折线图
# ─────────────────────────────────────────────────────────────────────────────
def plot_gflops_line(records, out_dir):
    pivot = pivot_by_size(records, 'gflops')
    sizes = sorted_sizes(pivot)
    algorithms = [a for a in ALGORITHM_ORDER if a != 'CPU-Baseline']  # GPU only

    fig, ax = plt.subplots(figsize=(max(8, n_sizes := len(sizes)) * 1.2, 5))
    for algo in algorithms:
        xs = []
        ys = []
        for s in sizes:
            val = pivot[s].get(algo)
            if val is not None and val > 0:
                xs.append(s)
                ys.append(val)
        if xs:
            ax.plot(xs, ys, 'o-', label=algo, color=ALGORITHM_COLORS.get(algo, None),
                    markersize=7, linewidth=2)

    ax.set_xlabel('Matrix Size (M×N×K)')
    ax.set_ylabel('GFLOPS')
    ax.set_title('GEMM Performance — GFLOPS Trend (GPU Algorithms)')
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    plt.xticks(rotation=15)
    plt.tight_layout()
    out = os.path.join(out_dir, 'benchmark_gflops_line.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 图表 4：加速比折线图
# ─────────────────────────────────────────────────────────────────────────────
def plot_speedup_line(records, out_dir):
    pivot = pivot_by_size(records, 'speedup')
    sizes = sorted_sizes(pivot)
    algorithms = [a for a in ALGORITHM_ORDER if a not in ('CPU-Baseline', 'GPU-cuBLAS')]

    fig, ax = plt.subplots(figsize=(max(8, n_sizes := len(sizes)) * 1.2, 5))
    for algo in algorithms:
        xs = []
        ys = []
        for s in sizes:
            val = pivot[s].get(algo)
            if val is not None and val > 0:
                xs.append(s)
                ys.append(val)
        if xs:
            ax.plot(xs, ys, 'o-', label=algo, color=ALGORITHM_COLORS.get(algo, None),
                    markersize=7, linewidth=2)

    ax.set_xlabel('Matrix Size (M×N×K)')
    ax.set_ylabel('Speedup vs CPU (×)')
    ax.set_title('GEMM Performance — Speedup vs CPU (GPU Algorithms)')
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    plt.xticks(rotation=15)
    plt.tight_layout()
    out = os.path.join(out_dir, 'benchmark_speedup_line.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Plot GEMM benchmark results')
    parser.add_argument('--input', '-i', type=str, default='../results/benchmark_results.csv',
                        help='Input CSV file (default: ../results/benchmark_results.csv)')
    parser.add_argument('--output', '-o', type=str, default='../results/',
                        help='Output directory for plots (default: ../results/)')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"[Error] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    records = load_csv(args.input)
    print(f"[OK] Loaded {len(records)} records from {args.input}", file=sys.stderr)

    plot_gflops_bar(records, output_dir)
    plot_time_bar(records, output_dir)
    plot_gflops_line(records, output_dir)
    plot_speedup_line(records, output_dir)

    print(f"[Done] All plots saved to: {output_dir}/", file=sys.stderr)


if __name__ == '__main__':
    main()