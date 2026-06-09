#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>

// ---------------------------------------------------------
// 浮点数误差容忍度：用于矩阵结果比对
// ---------------------------------------------------------
#define EPSILON 1e-4f

/**
 * @brief 使用 CPU 随机数生成器初始化矩阵，数值范围 [-1.0, 1.0]
 *        使用固定种子 (42) 保证实验结果可复现。
 * @param mat  指向矩阵数据的指针 (行优先存储)
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 */
void random_matrix_init(float* mat, int rows, int cols);

/**
 * @brief 逐元素比对两个矩阵，考虑浮点计算带来的数值误差。
 *        同时检查绝对误差与相对误差，取二者较大值进行判断。
 * @param A       参考矩阵 (通常来自 CPU 串行实现)
 * @param B       待验证矩阵 (通常来自 GPU 实现)
 * @param rows    矩阵行数
 * @param cols    矩阵列数
 * @param epsilon 误差容忍阈值，默认为 1e-4
 * @return        若所有元素误差均在容忍范围内返回 true，否则返回 false
 */
bool compare_matrices(const float* A, const float* B, int rows, int cols, float epsilon = EPSILON);

/**
 * @brief 计算 GEMM 操作的理论 GFLOPS
 *        单次乘加 (MAC) 计为 2 次浮点运算 (乘 + 加)。
 *        公式: GFLOPS = (2.0 * M * N * K) / (执行时间 ms * 1e6)
 * @param M  矩阵 C 的行数 (A 的行数)
 * @param N  矩阵 C 的列数 (B 的列数)
 * @param K  矩阵 A 的列数 / B 的行数 (内积维度)
 * @param ms 执行时间，单位毫秒
 * @return   理论峰值 GFLOPS
 */
inline double calc_gflops(int M, int N, int K, double ms) {
    return (2.0 * static_cast<double>(M) * N * K) / (ms * 1e6);
}

// ---------------------------------------------------------
// CPU 计时器：基于 std::chrono::high_resolution_clock
// 用于测量 Host 端代码 (如 CPU GEMM、内存分配) 的耗时
// ---------------------------------------------------------
class CpuTimer {
public:
    CpuTimer();
    void start();
    void stop();
    double elapsed_ms() const;  // 返回已耗费的毫秒数

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_;
};

// ---------------------------------------------------------
// GPU 计时器：基于 CUDA Event (cudaEventRecord / cudaEventSynchronize)
// 用于精确测量 Kernel 在 GPU 上的实际执行耗时，排除 Host 调度开销
// ---------------------------------------------------------
class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();
    void start();
    void stop();
    float elapsed_ms() const;   // 返回已耗费的毫秒数

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    bool started_;
};

#endif // UTILS_H
