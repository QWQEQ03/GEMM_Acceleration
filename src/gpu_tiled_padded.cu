#include "gemm_kernels.cuh"
#include "utils.h"
#include <cuda_runtime.h>

// ---------------------------------------------------------
// BLOCK_SIZE: 与 Tiled 版本保持一致，便于横向对比
// ---------------------------------------------------------
#define BLOCK_SIZE 16

/**
 * @brief CUDA Kernel: 基于 Shared Memory 的分块矩阵乘法 (Padding 版本)
 *
 * 核心差异：
 * - 假设 M, N, K 已被 Host 端填充至 BLOCK_SIZE 的整数倍 (padM, padN, padK)。
 * - Kernel 内部不做任何边界检查 (row < M, col < N 等 if 判断全部移除)。
 * - 这消除了 Warp Divergence，使得所有线程执行路径完全一致，
 *   特别适合 M/N/K 不是 BLOCK_SIZE 整数倍的非方阵场景。
 *
 * 代价：
 * - Host 端需要额外分配 pad 后的内存，并用 cudaMemcpy2D / memset 进行数据搬运。
 * - 当 pad 尺寸略大于原始尺寸时，有效计算占比 (M*N*K)/(padM*padN*padK) 会略低于 100%。
 */
__global__ void gemm_tiled_padded_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int padN, int padK) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = padK / BLOCK_SIZE;  // 整除，无需向上取整

    for (int tile = 0; tile < numTiles; ++tile) {
        // 协作加载 A 的 TILE（无边界检查）
        int aRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        int aCol = tile * BLOCK_SIZE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = A[aRow * padK + aCol];

        // 协作加载 B 的 TILE（无边界检查）
        int bRow = tile * BLOCK_SIZE + threadIdx.y;
        int bCol = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        sB[threadIdx.y][threadIdx.x] = B[bRow * padN + bCol];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 无边界判断，直接写回
    C[row * padN + col] = sum;
}

/**
 * @brief Host 端封装：启动 Tiled-Padded GEMM Kernel
 *
 * 流程：
 * 1. 计算 padM, padN, padK（向上对齐到 BLOCK_SIZE）。
 * 2. 分配 pad 后的设备内存 d_Ap, d_Bp, d_Cp。
 * 3. 将原始 A/B 数据通过 cudaMemcpy2D 拷贝到 pad 内存的有效区域，其余部分自动为 0。
 * 4. 启动无边界检查的 Kernel。
 * 5. 将 d_Cp 的有效区域通过 cudaMemcpy2D 拷贝回 d_C（原始尺寸）。
 * 6. 释放临时 pad 内存。
 *
 * 注意：计时范围从 pad 内存分配开始，到结果拷贝回 d_C 结束，
 *       以完整反映该策略的实际端到端开销。
 */
void gemm_gpu_tiled_padded(const float* A, const float* B, float* C,
                           int M, int N, int K,
                           int padM, int padN, int padK) {
    // 分配 pad 后的设备内存
    size_t bytes_Ap = static_cast<size_t>(padM) * padK * sizeof(float);
    size_t bytes_Bp = static_cast<size_t>(padK) * padN * sizeof(float);
    size_t bytes_Cp = static_cast<size_t>(padM) * padN * sizeof(float);

    float *d_Ap = nullptr, *d_Bp = nullptr, *d_Cp = nullptr;
    cudaMalloc((void**)&d_Ap, bytes_Ap);
    cudaMalloc((void**)&d_Bp, bytes_Bp);
    cudaMalloc((void**)&d_Cp, bytes_Cp);

    // 清零（确保 pad 区域为 0，不影响乘加结果）
    cudaMemset(d_Ap, 0, bytes_Ap);
    cudaMemset(d_Bp, 0, bytes_Bp);
    cudaMemset(d_Cp, 0, bytes_Cp);

    // 使用 cudaMemcpy2D 将有效数据拷贝到 pad 内存的左上角
    // A: M 行 × K 列 → padM 行 × padK 列
    cudaMemcpy2D(d_Ap, padK * sizeof(float), A, K * sizeof(float),
                 K * sizeof(float), M, cudaMemcpyDeviceToDevice);
    // B: K 行 × N 列 → padK 行 × padN 列
    cudaMemcpy2D(d_Bp, padN * sizeof(float), B, N * sizeof(float),
                 N * sizeof(float), K, cudaMemcpyDeviceToDevice);

    // 启动 Kernel
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(padN / BLOCK_SIZE, padM / BLOCK_SIZE);
    gemm_tiled_padded_kernel<<<grid, block>>>(d_Ap, d_Bp, d_Cp, padN, padK);
    cudaDeviceSynchronize();

    // 将 pad 结果的有效区域拷贝回原始输出 C (Device -> Device)
    cudaMemcpy2D(C, N * sizeof(float), d_Cp, padN * sizeof(float),
                 N * sizeof(float), M, cudaMemcpyDeviceToDevice);

    // 释放临时内存
    cudaFree(d_Ap);
    cudaFree(d_Bp);
    cudaFree(d_Cp);
}

// ---------------------------------------------------------
// cuBLAS 封装（动态加载，避免在 WSL2 等兼容性问题环境中导致
// 整个 CUDA 运行时初始化失败）
// ---------------------------------------------------------
#include <dlfcn.h>

typedef int (*cublasCreate_t)(void**);
typedef int (*cublasDestroy_t)(void*);
typedef int (*cublasSgemm_t)(void*, int, int, int, int, int,
                             const float*, const float*, int,
                             const float*, int,
                             const float*, float*, int);

/**
 * @brief Host 端封装：调用 cuBLAS 进行 GEMM（动态加载版）
 *
 * 在部分环境（如 WSL2 + CUDA 12.0）中，静态链接 cuBLAS 会导致 CUDA 运行时
 * 初始化失败（cudaGetDeviceCount 返回 0）。因此采用 dlopen 动态加载策略：
 * - 若 libcublas.so 加载失败，则标记 C 为 NaN，由调用方跳过计时。
 * - 若加载成功，则正常调用 cublasSgemm。
 *
 * 行优先转列优先的参数技巧同前。
 */
void gemm_gpu_cublas(const float* A, const float* B, float* C, int M, int N, int K) {
    void* handle_cublas = dlopen("libcublas.so.12", RTLD_LAZY);
    if (!handle_cublas) {
        // 动态库未找到或无法加载
        cudaMemset(C, 0xFF, static_cast<size_t>(M) * N * sizeof(float));
        return;
    }

    cublasCreate_t   my_cublasCreate   = (cublasCreate_t)dlsym(handle_cublas, "cublasCreate_v2");
    cublasDestroy_t  my_cublasDestroy  = (cublasDestroy_t)dlsym(handle_cublas, "cublasDestroy_v2");
    cublasSgemm_t    my_cublasSgemm    = (cublasSgemm_t)dlsym(handle_cublas, "cublasSgemm_v2");

    if (!my_cublasCreate || !my_cublasDestroy || !my_cublasSgemm) {
        cudaMemset(C, 0xFF, static_cast<size_t>(M) * N * sizeof(float));
        dlclose(handle_cublas);
        return;
    }

    void* handle = nullptr;
    int stat = my_cublasCreate(&handle);
    if (stat != 0) {
        // cuBLAS 初始化失败（如驱动不兼容）
        cudaMemset(C, 0xFF, static_cast<size_t>(M) * N * sizeof(float));
        dlclose(handle_cublas);
        return;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // CUBLAS_OP_N = 0
    my_cublasSgemm(handle, 0, 0,
                   N, M, K,
                   &alpha,
                   B, N,
                   A, K,
                   &beta,
                   C, N);

    cudaDeviceSynchronize();
    my_cublasDestroy(handle);
    dlclose(handle_cublas);
}
