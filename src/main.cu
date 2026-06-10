#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "utils.h"
#include "gemm_kernels.cuh"

// ---------------------------------------------------------
// 测试矩阵维度配置：方阵 + 非方阵（模拟注意力机制实际特征）
// ---------------------------------------------------------
struct TestCase {
    int M, N, K;
    const char* desc;
};

const TestCase TEST_CASES[] = {
    {1024, 1024, 1024, "Square"},
    {2048, 2048, 2048, "Square"},
    {4096, 4096, 4096, "Square"},
    {4096, 4096, 128,  "Long-seq x head-dim"},
    {4096, 128,  128,  "Flat projection"},
    {8192, 8192, 64,   "Ultra-long seq"},
};
const int NUM_TESTS = sizeof(TEST_CASES) / sizeof(TestCase);

/**
 * @brief 单次实验结果结构体
 * 用于收集某一维度下某一算法的性能数据，便于统一格式化输出
 */
struct Result {
    int M, N, K;          // 矩阵维度 (支持非方阵)
    const char* desc;     // 场景描述
    const char* algo;     // 算法名称字符串
    double time_ms;       // 执行时间 (毫秒)
    double gflops;        // 理论峰值 GFLOPS
    double speedup;       // 相对于 CPU 基线的加速比 (-1 表示 CPU 未运行)
    bool verified;        // 结果校验是否通过
};

/**
 * @brief 打印 Markdown 表格表头
 */
void print_header() {
    printf("| 矩阵大小 | 描述 | 算法类型 | 执行时间(ms) | GFLOPS | 加速比(vs CPU) | 结果校验 |\n");
    printf("|---|---|---|---|---|---|---|\n");
}

/**
 * @brief 打印单行结果到 Markdown 表格
 * @param r 结果结构体引用
 */
void print_result(const Result& r) {
    printf("| %d×%d×%d | %s | %s | ", r.M, r.N, r.K, r.desc, r.algo);
    if (r.time_ms < 0.0) {
        printf("N/A | N/A | N/A | %s |\n", r.verified ? "PASS" : "FAIL");
        return;
    }
    printf("%.3f | %.2f | ", r.time_ms, r.gflops);
    if (r.speedup > 0.0) {
        printf("%.2fx", r.speedup);
    } else {
        printf("-");
    }
    printf(" | %s |\n", r.verified ? "PASS" : "FAIL");
}

/**
 * @brief 主函数：驱动完整的基准测试流程
 *
 * 流程说明：
 * 1. 对每个测试用例 (M,N,K)，在 Host 端生成随机矩阵 A(M×K) 和 B(K×N)。
 * 2. 将 A、B 拷贝到 Device 端（一次拷贝，供所有 GPU 版本复用）。
 * 3. 依次执行 CPU 基线、GPU Naive、GPU Tiled、GPU Tiled-Padded、
 *    GPU Tiled-Coarse、GPU cuBLAS。
 * 4. 使用 utils.h 中的 CpuTimer / GpuTimer 进行精确计时。
 * 5. 将 GPU 结果拷贝回 Host，调用 compare_matrices() 与 CPU 黄金标准比对。
 * 6. 计算 GFLOPS 与加速比，以 Markdown 表格形式输出。
 * 7. 释放所有 Host / Device 内存。
 */
int main() {
    print_header();

    for (int t = 0; t < NUM_TESTS; ++t) {
        int M = TEST_CASES[t].M;
        int N = TEST_CASES[t].N;
        int K = TEST_CASES[t].K;
        const char* desc = TEST_CASES[t].desc;

        size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
        size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
        size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

        // ---------------------------------------------------------
        // Host 内存分配
        // h_C_cpu 作为黄金标准；其余 h_C_* 接收各 GPU 版本结果
        // ---------------------------------------------------------
        float* h_A  = (float*)malloc(bytes_A);
        float* h_B  = (float*)malloc(bytes_B);
        float* h_C_cpu    = (float*)malloc(bytes_C);
        float* h_C_naive  = (float*)malloc(bytes_C);
        float* h_C_tiled  = (float*)malloc(bytes_C);
        float* h_C_padded = (float*)malloc(bytes_C);
        float* h_C_coarse = (float*)malloc(bytes_C);
        float* h_C_cublas = (float*)malloc(bytes_C);

        if (!h_A || !h_B || !h_C_cpu || !h_C_naive || !h_C_tiled ||
            !h_C_padded || !h_C_coarse || !h_C_cublas) {
            fprintf(stderr, "[Error] Host 内存分配失败 (%d×%d×%d)\n", M, N, K);
            return EXIT_FAILURE;
        }

        // 初始化输入矩阵，数值范围 [-1.0, 1.0]
        random_matrix_init(h_A, M, K);
        random_matrix_init(h_B, K, N);

        // ---------------------------------------------------------
        // Device 内存分配与数据拷贝
        // 输入矩阵只需 H2D 一次，所有 GPU Kernel 共享 d_A 和 d_B
        // ---------------------------------------------------------
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        cudaMalloc((void**)&d_A, bytes_A);
        cudaMalloc((void**)&d_B, bytes_B);
        cudaMalloc((void**)&d_C, bytes_C);

        cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

        // 用于记录当前维度下 CPU 的执行时间，供后续计算加速比
        double cpu_time_ms = 0.0;
        // CPU 基线跳过阈值：总计算量 (MAC) 超过 100 亿则跳过
        bool run_cpu = (static_cast<long long>(M) * N * K <= 10000000000LL);

        // ---------------------------------------------------------
        // 1. CPU 串行基线 (可选)
        // ---------------------------------------------------------
        if (run_cpu) {
            CpuTimer cpu_timer;
            cpu_timer.start();
            gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
            cpu_timer.stop();
            cpu_time_ms = cpu_timer.elapsed_ms();

            Result res_cpu;
            res_cpu.M = M; res_cpu.N = N; res_cpu.K = K;
            res_cpu.desc = desc;
            res_cpu.algo = "CPU-Baseline";
            res_cpu.time_ms = cpu_time_ms;
            res_cpu.gflops = calc_gflops(M, N, K, cpu_time_ms);
            res_cpu.speedup = 1.0;
            res_cpu.verified = true;
            print_result(res_cpu);
        } else {
            printf("// [Note] %d×%d×%d 的 CPU 基线执行时间过长，已跳过。\n", M, N, K);
        }

        // ---------------------------------------------------------
        // 2. GPU Naive (Global Memory 直接访问)
        // ---------------------------------------------------------
        GpuTimer gpu_timer;

        // Warm-up
        gemm_gpu_naive(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        gpu_timer.start();
        gemm_gpu_naive(d_A, d_B, d_C, M, N, K);
        gpu_timer.stop();
        float naive_ms = gpu_timer.elapsed_ms();

        cudaMemcpy(h_C_naive, d_C, bytes_C, cudaMemcpyDeviceToHost);

        bool naive_pass = true;
        if (run_cpu) {
            naive_pass = compare_matrices(h_C_cpu, h_C_naive, M, N, EPSILON);
        }

        Result res_naive;
        res_naive.M = M; res_naive.N = N; res_naive.K = K;
        res_naive.desc = desc;
        res_naive.algo = "GPU-Naive";
        res_naive.time_ms = naive_ms;
        res_naive.gflops = calc_gflops(M, N, K, naive_ms);
        res_naive.speedup = run_cpu ? (cpu_time_ms / naive_ms) : -1.0;
        res_naive.verified = naive_pass;
        print_result(res_naive);

        // ---------------------------------------------------------
        // 3. GPU Tiled (Shared Memory 分块优化)
        // ---------------------------------------------------------
        gemm_gpu_tiled(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        gpu_timer.start();
        gemm_gpu_tiled(d_A, d_B, d_C, M, N, K);
        gpu_timer.stop();
        float tiled_ms = gpu_timer.elapsed_ms();

        cudaMemcpy(h_C_tiled, d_C, bytes_C, cudaMemcpyDeviceToHost);

        bool tiled_pass = true;
        if (run_cpu) {
            tiled_pass = compare_matrices(h_C_cpu, h_C_tiled, M, N, EPSILON);
        } else {
            tiled_pass = compare_matrices(h_C_naive, h_C_tiled, M, N, EPSILON);
        }

        Result res_tiled;
        res_tiled.M = M; res_tiled.N = N; res_tiled.K = K;
        res_tiled.desc = desc;
        res_tiled.algo = "GPU-Tiled";
        res_tiled.time_ms = tiled_ms;
        res_tiled.gflops = calc_gflops(M, N, K, tiled_ms);
        res_tiled.speedup = run_cpu ? (cpu_time_ms / tiled_ms) : -1.0;
        res_tiled.verified = tiled_pass;
        print_result(res_tiled);

        // ---------------------------------------------------------
        // 4. GPU Tiled + Padded (预填充对齐，Kernel 无分支)
        // ---------------------------------------------------------
        #define BLOCK_SIZE 16
        int padM = ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        int padN = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        int padK = ((K + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        #undef BLOCK_SIZE

        gemm_gpu_tiled_padded(d_A, d_B, d_C, M, N, K, padM, padN, padK);
        cudaDeviceSynchronize();

        gpu_timer.start();
        gemm_gpu_tiled_padded(d_A, d_B, d_C, M, N, K, padM, padN, padK);
        gpu_timer.stop();
        float padded_ms = gpu_timer.elapsed_ms();

        cudaMemcpy(h_C_padded, d_C, bytes_C, cudaMemcpyDeviceToHost);

        bool padded_pass = true;
        if (run_cpu) {
            padded_pass = compare_matrices(h_C_cpu, h_C_padded, M, N, EPSILON);
        } else {
            padded_pass = compare_matrices(h_C_naive, h_C_padded, M, N, EPSILON);
        }

        Result res_padded;
        res_padded.M = M; res_padded.N = N; res_padded.K = K;
        res_padded.desc = desc;
        res_padded.algo = "GPU-Tiled-Padded";
        res_padded.time_ms = padded_ms;
        res_padded.gflops = calc_gflops(M, N, K, padded_ms);
        res_padded.speedup = run_cpu ? (cpu_time_ms / padded_ms) : -1.0;
        res_padded.verified = padded_pass;
        print_result(res_padded);

        // ---------------------------------------------------------
        // 5. GPU Tiled + Coarse (线程粗化，每个线程计算 2 个元素)
        // ---------------------------------------------------------
        gemm_gpu_tiled_coarse(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        gpu_timer.start();
        gemm_gpu_tiled_coarse(d_A, d_B, d_C, M, N, K);
        gpu_timer.stop();
        float coarse_ms = gpu_timer.elapsed_ms();

        cudaMemcpy(h_C_coarse, d_C, bytes_C, cudaMemcpyDeviceToHost);

        bool coarse_pass = true;
        if (run_cpu) {
            coarse_pass = compare_matrices(h_C_cpu, h_C_coarse, M, N, EPSILON);
        } else {
            coarse_pass = compare_matrices(h_C_naive, h_C_coarse, M, N, EPSILON);
        }

        Result res_coarse;
        res_coarse.M = M; res_coarse.N = N; res_coarse.K = K;
        res_coarse.desc = desc;
        res_coarse.algo = "GPU-Tiled-Coarse";
        res_coarse.time_ms = coarse_ms;
        res_coarse.gflops = calc_gflops(M, N, K, coarse_ms);
        res_coarse.speedup = run_cpu ? (cpu_time_ms / coarse_ms) : -1.0;
        res_coarse.verified = coarse_pass;
        print_result(res_coarse);

        // ---------------------------------------------------------
        // 6. GPU cuBLAS (官方库，性能上限参考)
        // ---------------------------------------------------------
        gemm_gpu_cublas(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        gpu_timer.start();
        gemm_gpu_cublas(d_A, d_B, d_C, M, N, K);
        gpu_timer.stop();
        float cublas_ms = gpu_timer.elapsed_ms();

        cudaMemcpy(h_C_cublas, d_C, bytes_C, cudaMemcpyDeviceToHost);

        // 检测 cuBLAS 是否初始化失败：若 C 被填充为 NaN，则标记为不可用
        bool cublas_available = std::isfinite(h_C_cublas[0]);

        Result res_cublas;
        res_cublas.M = M; res_cublas.N = N; res_cublas.K = K;
        res_cublas.desc = desc;
        if (cublas_available) {
            bool cublas_pass = true;
            if (run_cpu) {
                cublas_pass = compare_matrices(h_C_cpu, h_C_cublas, M, N, EPSILON);
            } else {
                cublas_pass = compare_matrices(h_C_naive, h_C_cublas, M, N, EPSILON);
            }
            res_cublas.algo = "GPU-cuBLAS";
            res_cublas.time_ms = cublas_ms;
            res_cublas.gflops = calc_gflops(M, N, K, cublas_ms);
            res_cublas.speedup = run_cpu ? (cpu_time_ms / cublas_ms) : -1.0;
            res_cublas.verified = cublas_pass;
        } else {
            res_cublas.algo = "GPU-cuBLAS";
            res_cublas.time_ms = -1.0;
            res_cublas.gflops = -1.0;
            res_cublas.speedup = -1.0;
            res_cublas.verified = false;
        }
        print_result(res_cublas);

        // ---------------------------------------------------------
        // 资源释放
        // ---------------------------------------------------------
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_naive);
        free(h_C_tiled);
        free(h_C_padded);
        free(h_C_coarse);
        free(h_C_cublas);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        printf("\n");
    }

    return 0;
}
