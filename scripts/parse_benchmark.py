#!/usr/bin/env python3
"""
parse_benchmark.py — 解析 gemm_acc 的 Markdown 表格输出，保存为 CSV

用法：
    ./gemm_acc | python parse_benchmark.py
    python parse_benchmark.py < output.txt
    python parse_benchmark.py --input output.txt --output results.csv
"""

import sys
import csv
import re
import argparse
import os
from pathlib import Path


def parse_size(size_str):
    """解析 '1024×1024×1024' 格式，返回 (M, N, K)"""
    parts = size_str.split('×')
    if len(parts) != 3:
        raise ValueError(f"Invalid size format: {size_str}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def parse_speedup(speedup_str):
    """解析 '709.03x' 或 '-'，返回 float 或空字符串"""
    s = speedup_str.strip()
    if s == '-' or s == '':
        return ''
    m = re.match(r'([\d.]+)x?', s)
    if m:
        return float(m.group(1))
    return ''


def is_data_row(line):
    """判断是否为有效的数据行（跳过表头、分隔线、注释、Compare 行等）"""
    line = line.strip()
    if not line:
        return False
    # Markdown 表格分隔线
    if line.startswith('|---|'):
        return False
    # 表头
    if line.startswith('| 矩阵大小'):
        return False
    # 注释行
    if line.startswith('//'):
        return False
    # Compare 调试行
    if line.startswith('[Compare]'):
        return False
    # 必须是 |开头且有足够的列
    if not line.startswith('|'):
        return False
    cols = [c.strip() for c in line.split('|')]
    # cols[0] 为空（左侧 |），cols[-1] 为空（右侧 |）
    return len(cols) >= 8


def parse_row(line):
    """解析一行数据，返回字段字典"""
    cols = [c.strip() for c in line.split('|')]
    # cols[0] = '', cols[1] = 矩阵大小, cols[2] = 描述, cols[3] = 算法类型,
    # cols[4] = 时间, cols[5] = GFLOPS, cols[6] = 加速比, cols[7] = 校验, cols[8] = ''

    size_str = cols[1]
    M, N, K = parse_size(size_str)

    time_str = cols[4]
    gflops_str = cols[5]
    speedup_str = cols[6]
    verified_str = cols[7]

    time_ms = float(time_str) if time_str not in ('N/A', '-', '') else ''
    gflops = float(gflops_str) if gflops_str not in ('N/A', '-', '') else ''
    speedup = parse_speedup(speedup_str)
    verified = verified_str

    return {
        'M': M,
        'N': N,
        'K': K,
        'description': cols[2],
        'algorithm': cols[3],
        'time_ms': time_ms,
        'gflops': gflops,
        'speedup': speedup,
        'verified': verified,
    }


def parse_stream(input_stream):
    """从输入流解析所有数据行"""
    results = []
    for line in input_stream:
        if is_data_row(line):
            try:
                row = parse_row(line)
                results.append(row)
            except Exception as e:
                print(f"[Warning] Failed to parse line: {line[:60]}... — {e}", file=sys.stderr)
    return results


def write_csv(results, output_file):
    """将解析结果写入 CSV"""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    fieldnames = ['M', 'N', 'K', 'description', 'algorithm', 'time_ms', 'gflops', 'speedup', 'verified']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description='Parse gemm_acc benchmark output to CSV')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Input file (default: stdin)')
    parser.add_argument('--output', '-o', type=str, default='../results/benchmark_results.csv',
                        help='Output CSV file (default: ../results/benchmark_results.csv)')
    args = parser.parse_args()

    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            results = parse_stream(f)
    else:
        results = parse_stream(sys.stdin)

    if not results:
        print("[Error] No data rows parsed. Check input format.", file=sys.stderr)
        sys.exit(1)

    write_csv(results, args.output)
    print(f"[OK] Wrote {len(results)} rows to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()