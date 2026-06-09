#include <cstdio>
#include <cstdlib>
#include "utils.h"
#include "gemm_kernels.cuh"

// ---------------------------------------------------------
// 测试矩阵维度：方阵 N × N
// 分别测试 1024, 2048, 4096 三种规模，覆盖小、中、大三种计算负载
// ---------------------------------------------------------
const int TEST_DIMS[] = {1024, 2048, 4096};
const int NUM_TESTS = sizeof(TEST_DIMS) / sizeof(int);

/**
 * @brief 单次实验结果结构体
 * 用于收集某一维度下某一算法的性能数据，便于统一格式化输出
 */
struct Result {
    int N;              // 矩阵维度 (方阵)
    const char* algo;   // 算法名称字符串
    double time_ms;     // 执行时间 (毫秒)
    double gflops;      // 理论峰值 GFLOPS
    double speedup;     // 相对于 CPU 基线的加速比 (-1 表示 CPU 未运行，无意义)
    bool verified;      // 结果校验是否通过
};

/**
 * @brief 打印 Markdown 表格表头
 * 该格式可直接复制到技术报告或 Markdown 文档中
 */
void print_header() {
    printf("| 矩阵大小 | 算法类型 | 执行时间(ms) | GFLOPS | 加速比(vs CPU) | 结果校验 |\n");
    printf("|---|---|---|---|---|---|\n");
}

/**
 * @brief 打印单行结果到 Markdown 表格
 * @param r 结果结构体引用
 */
void print_result(const Result& r) {
    printf("| %d | %s | %.3f | %.2f | ", r.N, r.algo, r.time_ms, r.gflops);
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
 * 1. 对每个测试维度 N，在 Host 端生成随机矩阵 A 和 B。
 * 2. 将 A、B 拷贝到 Device 端（一次拷贝，供所有 GPU 版本复用）。
 * 3. 依次执行 CPU 基线、GPU Naive、GPU Tiled 三个版本。
 * 4. 使用 utils.h 中的 CpuTimer / GpuTimer 进行精确计时。
 * 5. 将 GPU 结果拷贝回 Host，调用 compare_matrices() 与 CPU 黄金标准进行比对。
 * 6. 计算 GFLOPS 与加速比，以 Markdown 表格形式输出。
 * 7. 释放所有 Host / Device 内存。
 *
 * 关于 CPU 基线的取舍：
 * - N = 1024, 2048：计算量适中，执行一次 CPU 基线以获取加速比。
 * - N = 4096：计算量达 68.7 GMAC (137 GFLOP)，单线程 CPU 可能耗时数十秒，
 *   因此跳过 CPU 基线。此时 GPU Tiled 的结果将与 GPU Naive 的结果进行交叉验证。
 */
int main() {
    print_header();

    for (int t = 0; t < NUM_TESTS; ++t) {
        int N = TEST_DIMS[t];
        int M = N, K = N;
        size_t matrix_bytes = static_cast<size_t>(N) * N * sizeof(float);

        // ---------------------------------------------------------
        // Host 内存分配：页锁定内存 (Pageable) 即可满足需求
        // h_C_cpu 作为黄金标准；h_C_naive 和 h_C_tiled 接收 GPU 结果
        // ---------------------------------------------------------
        float* h_A = (float*)malloc(matrix_bytes);
        float* h_B = (float*)malloc(matrix_bytes);
        float* h_C_cpu = (float*)malloc(matrix_bytes);
        float* h_C_naive = (float*)malloc(matrix_bytes);
        float* h_C_tiled = (float*)malloc(matrix_bytes);

        if (!h_A || !h_B || !h_C_cpu || !h_C_naive || !h_C_tiled) {
            fprintf(stderr, "[Error] Host 内存分配失败 (N=%d)\n", N);
            return EXIT_FAILURE;
        }

        // 初始化输入矩阵，数值范围 [-1.0, 1.0]
        random_matrix_init(h_A, N, N);
        random_matrix_init(h_B, N, N);

        // ---------------------------------------------------------
        // Device 内存分配与数据拷贝
        // 输入矩阵只需 H2D 一次，所有 GPU Kernel 共享 d_A 和 d_B
        // ---------------------------------------------------------
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        cudaMalloc((void**)&d_A, matrix_bytes);
        cudaMalloc((void**)&d_B, matrix_bytes);
        cudaMalloc((void**)&d_C, matrix_bytes);

        cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice);

        // 用于记录当前维度下 CPU 的执行时间，供后续计算加速比
        double cpu_time_ms = 0.0;
        // 所有维度均执行 CPU 基线，用于获取完整的加速比对比
        bool run_cpu = true;

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
            res_cpu.N = N;
            res_cpu.algo = "CPU-Baseline";
            res_cpu.time_ms = cpu_time_ms;
            res_cpu.gflops = calc_gflops(M, N, K, cpu_time_ms);
            res_cpu.speedup = 1.0;   // 基线加速比定义为 1.0x
            res_cpu.verified = true; // CPU 作为黄金标准，默认正确
            print_result(res_cpu);
        } else {
            printf("// [Note] N=%d 的 CPU 基线执行时间过长，已跳过。\n", N);
        }

        // ---------------------------------------------------------
        // 2. GPU Naive (Global Memory 直接访问)
        // ---------------------------------------------------------
        GpuTimer gpu_timer;

        // Warm-up：第一次 Kernel 启动可能包含 CUDA Context 初始化、JIT 缓存等额外开销，
        // 不计入正式计时，以排除冷启动噪声，保证测量稳定性。
        gemm_gpu_naive(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        // 正式计时
        gpu_timer.start();
        gemm_gpu_naive(d_A, d_B, d_C, M, N, K);
        gpu_timer.stop();
        float naive_ms = gpu_timer.elapsed_ms();

        // 将结果从 Device 拷贝回 Host，用于后续校验
        cudaMemcpy(h_C_naive, d_C, matrix_bytes, cudaMemcpyDeviceToHost);

        bool naive_pass = true;
        if (run_cpu) {
            // 与 CPU 黄金标准进行比对，误差阈值 1e-4
            naive_pass = compare_matrices(h_C_cpu, h_C_naive, N, N, EPSILON);
        }

        Result res_naive;
        res_naive.N = N;
        res_naive.algo = "GPU-Naive";
        res_naive.time_ms = naive_ms;
        res_naive.gflops = calc_gflops(M, N, K, naive_ms);
        res_naive.speedup = run_cpu ? (cpu_time_ms / naive_ms) : -1.0;
        res_naive.verified = naive_pass;
        print_result(res_naive);

        // ---------------------------------------------------------
        // 3. GPU Tiled (Shared Memory 分块优化)
        // ---------------------------------------------------------
        // Warm-up
        gemm_gpu_tiled(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        // 正式计时
        gpu_timer.start();
        gemm_gpu_tiled(d_A, d_B, d_C, M, N, K);
        gpu_timer.stop();
        float tiled_ms = gpu_timer.elapsed_ms();

        cudaMemcpy(h_C_tiled, d_C, matrix_bytes, cudaMemcpyDeviceToHost);

        bool tiled_pass = true;
        if (run_cpu) {
            // 与 CPU 黄金标准比对
            tiled_pass = compare_matrices(h_C_cpu, h_C_tiled, N, N, EPSILON);
        } else {
            // CPU 被跳过时，退而求其次：与 GPU-Naive 进行交叉验证
            // 若二者结果一致，则可高度置信 Tiled 实现的正确性
            tiled_pass = compare_matrices(h_C_naive, h_C_tiled, N, N, EPSILON);
        }

        Result res_tiled;
        res_tiled.N = N;
        res_tiled.algo = "GPU-Tiled";
        res_tiled.time_ms = tiled_ms;
        res_tiled.gflops = calc_gflops(M, N, K, tiled_ms);
        res_tiled.speedup = run_cpu ? (cpu_time_ms / tiled_ms) : -1.0;
        res_tiled.verified = tiled_pass;
        print_result(res_tiled);

        // ---------------------------------------------------------
        // 资源释放：严格配对 malloc/free 与 cudaMalloc/cudaFree，防止内存泄漏
        // ---------------------------------------------------------
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_naive);
        free(h_C_tiled);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // 每个维度测试结束后空一行，提升可读性
        printf("\n");
    }

    return 0;
}
