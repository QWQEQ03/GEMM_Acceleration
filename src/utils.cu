#include "utils.h"
#include <algorithm>

// ---------------------------------------------------------
// 矩阵随机初始化 (Host 端)
// 使用 Mersenne Twister 引擎与均匀分布，数值范围 [-1.0, 1.0]
// 固定种子 (seed = 42) 以确保每次运行初始矩阵相同，便于结果复现与比对
// ---------------------------------------------------------
void random_matrix_init(float* mat, int rows, int cols) {
    // Mersenne Twister 19937 是一种高质量的伪随机数生成器
    std::mt19937 gen(42);
    // uniform_real_distribution 生成 [a, b) 区间的浮点数，这里近似 [-1.0, 1.0]
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    const int total_elements = rows * cols;
    for (int i = 0; i < total_elements; ++i) {
        mat[i] = dis(gen);
    }
}

// ---------------------------------------------------------
// 矩阵结果比对 (Host 端)
// 同时检查绝对误差与相对误差，对接近零的元素更稳健
// 当发现不一致时，打印前 5 处具体数值，方便调试
// ---------------------------------------------------------
bool compare_matrices(const float* A, const float* B, int rows, int cols, float epsilon) {
    bool match = true;
    double max_err = 0.0;
    int err_count = 0;
    const int total_elements = rows * cols;

    for (int i = 0; i < total_elements; ++i) {
        float diff = std::fabs(A[i] - B[i]);

        // 复合容差判定逻辑：
        // 当参考值 |ref| 较大时，允许一定的相对误差；当参考值接近 0 时，严格按绝对误差控制。
        // tolerance = epsilon * max(1.0, |ref|)
        // 例如 epsilon = 1e-4：
        //   - 若 ref = 0.002，tolerance = 1e-4 * 1.0 = 1e-4（只看绝对误差）
        //   - 若 ref = 100.0，tolerance = 1e-4 * 100 = 0.01（相当于相对误差 1e-4）
        // 这样避免了对接近零的元素过度敏感，同时又保证了大数值的精度要求。
        float tolerance = epsilon * std::max(1.0f, std::fabs(A[i]));

        if (diff > tolerance) {
            if (err_count < 5) {
                float rel_diff = diff / (std::fabs(A[i]) + 1e-6f);
                printf("[Mismatch] index %d: ref=%.6f, got=%.6f, abs_err=%.6f, tolerance=%.6f, rel_err=%.6f\n",
                       i, A[i], B[i], diff, tolerance, rel_diff);
            }
            match = false;
            err_count++;
        }
        if (diff > max_err) {
            max_err = diff;
        }
    }

    if (!match) {
        printf("[Compare] FAILED: total %d errors, max absolute error = %.6f\n", err_count, max_err);
    } else {
        printf("[Compare] PASSED: max absolute error = %.6f (threshold %.6f)\n", max_err, epsilon);
    }
    return match;
}

// ---------------------------------------------------------
// CPU 计时器实现
// 封装 std::chrono::high_resolution_clock，提供毫秒级精度
// ---------------------------------------------------------
CpuTimer::CpuTimer() : running_(false) {}

void CpuTimer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    running_ = true;
}

void CpuTimer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    running_ = false;
}

double CpuTimer::elapsed_ms() const {
    // 若计时器仍在运行，计算到当前时刻的耗时；否则计算到 stop 时刻的耗时
    auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    return std::chrono::duration<double, std::milli>(end - start_time_).count();
}

// ---------------------------------------------------------
// GPU 计时器实现
// 利用 CUDA Event 的异步特性，精确捕获 Kernel 在 GPU 上的执行区间
// cudaEventRecord 在指定的流中插入事件标记，cudaEventSynchronize 等待事件完成
// ---------------------------------------------------------
GpuTimer::GpuTimer() : started_(false) {
    // 创建 CUDA 事件对象，默认标志为 cudaEventDefault
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

GpuTimer::~GpuTimer() {
    // 销毁 CUDA 事件对象，释放关联资源
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void GpuTimer::start() {
    // 在默认流 (stream 0) 中记录开始事件
    cudaEventRecord(start_event_, 0);
    started_ = true;
}

void GpuTimer::stop() {
    // 在默认流中记录结束事件，并立即同步等待该事件完成，确保后续读取时间准确
    cudaEventRecord(stop_event_, 0);
    cudaEventSynchronize(stop_event_);
    started_ = false;
}

float GpuTimer::elapsed_ms() const {
    float elapsed = 0.0f;
    // 计算两个事件之间的时间差，结果单位为毫秒 (ms)
    cudaEventElapsedTime(&elapsed, start_event_, stop_event_);
    return elapsed;
}
