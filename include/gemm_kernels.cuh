#ifndef GEMM_KERNELS_CUH
#define GEMM_KERNELS_CUH

#include <cuda_runtime.h>

/**
 * @brief CPU 端串行 GEMM 实现：C = A × B
 *        A: M × K,  B: K × N,  C: M × N
 *        计算采用最经典的三重循环，作为性能对比的基线 (Baseline)。
 * @param A 输入矩阵 (行优先), 尺寸 M × K
 * @param B 输入矩阵 (行优先), 尺寸 K × N
 * @param C 输出矩阵 (行优先), 尺寸 M × N
 * @param M 矩阵 A 的行数 / C 的行数
 * @param N 矩阵 B 的列数 / C 的列数
 * @param K 矩阵 A 的列数 / B 的行数 (内积长度)
 */
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * @brief GPU 端朴素 GEMM 实现 (Global Memory 直接访问)
 *        每个线程负责计算 C 矩阵中的一个元素。
 *        直接从 Global Memory 读取 A 的行和 B 的列，没有任何缓存优化。
 *        用于展示最直观的并行思路，但性能受限于显存带宽。
 * @param A 设备端输入矩阵 (行优先), 尺寸 M × K
 * @param B 设备端输入矩阵 (行优先), 尺寸 K × N
 * @param C 设备端输出矩阵 (行优先), 尺寸 M × N
 * @param M 矩阵 A 的行数 / C 的行数
 * @param N 矩阵 B 的列数 / C 的列数
 * @param K 矩阵 A 的列数 / B 的行数
 */
void gemm_gpu_naive(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * @brief GPU 端共享内存分块 (Tiled) GEMM 优化实现
 *        利用 CUDA Shared Memory 将 Global Memory 上的数据分块 (Tile)
 *        加载到快速的片上缓存中，大幅减少重复的显存访问，提升计算强度。
 *        同时结合寄存器累加与同步原语，最大化 SM 利用率。
 * @param A 设备端输入矩阵 (行优先), 尺寸 M × K
 * @param B 设备端输入矩阵 (行优先), 尺寸 K × N
 * @param C 设备端输出矩阵 (行优先), 尺寸 M × N
 * @param M 矩阵 A 的行数 / C 的行数
 * @param N 矩阵 B 的列数 / C 的列数
 * @param K 矩阵 A 的列数 / B 的行数
 */
void gemm_gpu_tiled(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * @brief GPU 端共享内存分块 + 线程粗化 GEMM 实现
 *        每个线程负责计算输出矩阵中同行相邻的 2 个元素，
 *        通过增加寄存器累加器数量来更好地掩盖访存延迟。
 * @param A 设备端输入矩阵 (行优先), 尺寸 M × K
 * @param B 设备端输入矩阵 (行优先), 尺寸 K × N
 * @param C 设备端输出矩阵 (行优先), 尺寸 M × N
 */
void gemm_gpu_tiled_coarse(const float* A, const float* B, float* C, int M, int N, int K);

#endif // GEMM_KERNELS_CUH
