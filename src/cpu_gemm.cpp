#include "gemm_kernels.cuh"
#include <cstring>  // 用于 memset

/**
 * @brief CPU 端串行 GEMM 实现：C = A × B
 *
 * 算法说明：
 * 采用经典的三重循环，但循环顺序为 i → k → j，而非直观的 i → j → k。
 * 原因（缓存友好性）：
 * - 矩阵按行优先 (Row-Major) 存储。
 * - 在 k 为外层的内层 j 循环中：
 *   · B[k][j] 的访问是沿 B 的第 k 行连续扫描，空间局部性极佳，利于 CPU 缓存命中。
 *   · C[i][j] 的访问是沿 C 的第 i 行连续写入，同样具有良好的空间局部性。
 *   · A[i][k] 在当前内层循环中为常量，可从寄存器读取，无需重复访存。
 * - 若采用 i → j → k 顺序，则 B[k][j] 的访问将变成跨行跳跃（步长为 N），
 *   严重破坏缓存局部性，性能会下降数倍。
 *
 * 复杂度：O(M × N × K)，单线程串行执行，作为 GPU 加速的对比基线 (Baseline)。
 *
 * @param A 输入矩阵 (行优先), 尺寸 M × K
 * @param B 输入矩阵 (行优先), 尺寸 K × N
 * @param C 输出矩阵 (行优先), 尺寸 M × N，函数内部会先清零
 * @param M 矩阵 A 的行数 / C 的行数
 * @param N 矩阵 B 的列数 / C 的列数
 * @param K 矩阵 A 的列数 / B 的行数 (内积长度)
 */
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // 先将输出矩阵 C 清零，因为后续使用累加模式 (C[i][j] += ...)
    std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(float));

    // 外层循环：遍历 C 的每一行 (对应 A 的每一行)
    for (int i = 0; i < M; ++i) {
        // 中层循环：遍历 A 的列 / B 的行 (内积维度 K)
        for (int k = 0; k < K; ++k) {
            // 将 A[i][k] 提取到局部变量，避免重复索引计算和重复访存
            float a_ik = A[i * K + k];

            // 内层循环：遍历 C 的每一列 (对应 B 的每一列)
            // 此时 B[k][j] 和 C[i][j] 均为行优先连续访问
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}
