#include "gemm_kernels.cuh"
#include "utils.h"
#include <cuda_runtime.h>

// ---------------------------------------------------------
// 朴素 Kernel 的线程块配置
// 每个 Block 含 16×16 = 256 个线程，是一个 Warp 的整数倍，
// 有利于 Warp Scheduler 的调度，但与 Tiled 版本不同，
// 这里不利用 Shared Memory，仅作为性能对比的基准。
// ---------------------------------------------------------
#define NAIVE_BLOCK_SIZE 16

/**
 * @brief CUDA Kernel: 朴素矩阵乘法 (Naive GEMM)
 *
 * 核心思路：
 * 每个线程负责计算输出矩阵 C 中的一个元素 C[row][col]。
 * 具体地，线程 (blockIdx.x, blockIdx.y) 内的 (threadIdx.x, threadIdx.y)
 * 计算全局坐标 (row, col) = (blockIdx.y * BLOCK_SIZE + threadIdx.y,
 *                              blockIdx.x * BLOCK_SIZE + threadIdx.x)
 * 处的元素。
 *
 * 内存访问特征（性能瓶颈所在）：
 * 1. 读取 A[row][k]：沿 A 的第 row 行连续读取，满足合并访问条件，带宽利用率尚可。
 * 2. 读取 B[k][col]：沿 B 的第 col 列读取，行优先存储下列内地址跨度为 N，
 *    同一 Warp 内相邻线程的内存地址相距甚远（跨步访问，Strided Access），
 *    严重不满足 Global Memory 的合并访问要求，导致带宽急剧下降。
 * 3. 写入 C[row][col]：每个线程独立写回一个元素，若地址连续则合并，通常无大问题。
 *
 * 总结：朴素实现虽然逻辑最简单，但由于对 B 矩阵的列访问模式极差，
 * 且所有数据均来自高延迟的 Global Memory（无 Shared Memory 缓存），
 * 实际性能往往远低于理论峰值，通常只有峰值性能的 1%~5%。
 *
 * @param A 设备端输入矩阵 (行优先), 尺寸 M × K
 * @param B 设备端输入矩阵 (行优先), 尺寸 K × N
 * @param C 设备端输出矩阵 (行优先), 尺寸 M × N
 * @param M 矩阵 A 的行数 / C 的行数
 * @param N 矩阵 B 的列数 / C 的列数
 * @param K 矩阵 A 的列数 / B 的行数
 */
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    // 计算当前线程负责的 C 元素的全局坐标
    int row = blockIdx.y * NAIVE_BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * NAIVE_BLOCK_SIZE + threadIdx.x;

    // 边界检查：防止网格覆盖超出矩阵实际尺寸的区域（当 M/N 不是块大小整数倍时）
    if (row < M && col < N) {
        float sum = 0.0f;

        // 内积计算：C[row][col] = Σ(A[row][k] * B[k][col])
        // 注意：B[k][col] 的访问模式是列优先式的跨步访问，这是主要性能瓶颈
        for (int k = 0; k < K; ++k) {
            // A[row][k] 的线性索引：row * K + k (沿行连续)
            // B[k][col] 的线性索引：k * N + col (沿列跨步，步长为 N)
            sum += A[row * K + k] * B[k * N + col];
        }

        // 将结果写回 Global Memory
        C[row * N + col] = sum;
    }
}

/**
 * @brief Host 端封装函数：启动朴素 GEMM Kernel
 *
 * 配置线程网格：
 * - Block 维度：NAIVE_BLOCK_SIZE × NAIVE_BLOCK_SIZE (16×16)
 * - Grid 维度：向上取整覆盖 M 行和 N 列
 */
void gemm_gpu_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(NAIVE_BLOCK_SIZE, NAIVE_BLOCK_SIZE);
    dim3 grid((N + NAIVE_BLOCK_SIZE - 1) / NAIVE_BLOCK_SIZE,
              (M + NAIVE_BLOCK_SIZE - 1) / NAIVE_BLOCK_SIZE);

    gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);

    // 同步设备，确保 Kernel 执行完毕
    cudaDeviceSynchronize();
}
