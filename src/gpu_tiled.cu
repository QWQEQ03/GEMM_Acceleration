#include "gemm_kernels.cuh"
#include "utils.h"
#include <cuda_runtime.h>

// ---------------------------------------------------------
// BLOCK_SIZE: 定义每个 CUDA Block 的维度 (BLOCK_SIZE × BLOCK_SIZE)
// 每个 Block 负责计算 C 矩阵中一个 BLOCK_SIZE × BLOCK_SIZE 的子块。
// 选择 16 的原因：
//   1. 一个 Block 含 16×16 = 256 个线程，恰好等于一个 Warp 的整数倍 (8 Warps)，
//      有利于 Warp Scheduler 的调度与 hiding latency。
//   2. Shared Memory 占用：sA + sB = 2 × 16 × 16 × 4B = 2KB，远低于 SM 的 48KB/100KB 上限，
//      允许单个 SM 上同时驻留多个 Block，提升占用率 (Occupancy)。
//   3. 若 GPU 共享内存更大，可尝试 32，但需权衡寄存器压力。
// ---------------------------------------------------------
#define BLOCK_SIZE 16

/**
 * @brief CUDA Kernel: 基于 Shared Memory 的分块矩阵乘法 (Tiled GEMM)
 *
 * 核心优化原理：
 * 朴素实现中，每个线程计算 C 的一个元素时，需要从 Global Memory 读取 A 的整行 (K 个元素)
 * 和 B 的整列 (K 个元素)。由于 Global Memory 延迟高、带宽相对低，这成为性能瓶颈。
 *
 * 分块 (Tiling) 策略：
 * 1. 将 K 维度切分为多个宽度为 BLOCK_SIZE 的 TILE。
 * 2. 每个 CUDA Block 负责计算 C 的一个子块 (BLOCK_SIZE × BLOCK_SIZE)。
 * 3. Block 内所有线程协作，将当前 TILE 对应的 A 子块和 B 子块从 Global Memory
 *    一次性加载到快速的 Shared Memory 中。
 * 4. 加载完毕后调用 __syncthreads() 保证数据可见性。
 * 5. 每个线程随后从 Shared Memory 中反复读取数据进行乘加运算。
 * 6. 处理完当前 TILE 后，再次 __syncthreads()，然后加载下一对 TILE，如此循环。
 *
 * 为什么 Shared Memory 能加速？
 * - Global Memory 访问延迟约 400-800 个时钟周期；Shared Memory 仅约 20-30 个周期。
 * - 一个 TILE 加载到 Shared Memory 后，Block 内的 BLOCK_SIZE 个线程可以复用该数据
 *   BLOCK_SIZE 次，大幅降低了对 Global Memory 的总访问次数。
 * - 从计算访存比 (Arithmetic Intensity) 角度看，每个浮点运算对应的 Global Memory 访问量
 *   从 O(1) 降低到 O(1/BLOCK_SIZE)，显著提升 Roofline Model 下的性能上限。
 *
 * 访存合并 (Coalesced Access)：
 * - 在加载 A 的子块时，同一 Warp 内相邻线程的 threadIdx.x 相差 1，
 *   对应 Global Memory 地址 aRow * K + aCol 也连续递增，满足合并访问条件，
 *   使得一次内存事务 (Memory Transaction) 可为 32 个线程服务。
 * - 加载 B 的子块同理，bRow * N + bCol 的地址随 threadIdx.x 连续递增，同样是合并访问。
 *
 * __syncthreads() 的必要性：
 * - 第一次：加载 TILE 后，必须等待 Block 内所有线程完成写入 Shared Memory，
 *   否则先到达的线程可能读到其他线程尚未写完的脏数据。
 * - 第二次：计算完当前 TILE 后，必须等待所有线程完成读取 Shared Memory，
 *   否则下一轮循环开始覆盖 sA/sB 时，慢线程可能还在读取旧数据。
 */
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    // ---------------------------------------------------------
    // Shared Memory 声明：每个 Block 独占一份，生命周期与 Block 相同
    // sA 存储从 A 加载的 TILE (BLOCK_SIZE × BLOCK_SIZE)
    // sB 存储从 B 加载的 TILE (BLOCK_SIZE × BLOCK_SIZE)
    // ---------------------------------------------------------
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // 当前线程在全局 C 矩阵中负责的行和列索引
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // 寄存器变量 sum：累加当前线程对应的 C[row][col] 结果
    // 存放在寄存器中，避免中间结果写回内存，是性能关键之一
    float sum = 0.0f;

    // 计算 K 维度需要划分为多少个 TILE
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ---------------------------------------------------------
    // 主循环：遍历 K 维度的所有 TILE
    // ---------------------------------------------------------
    for (int tile = 0; tile < numTiles; ++tile) {
        // ================================================
        // 阶段一：协作加载 (Collaborative Loading)
        // Block 内所有线程分工合作，将 A 和 B 的当前 TILE 从 Global Memory
        // 加载到 Shared Memory。每个线程负责加载一个元素。
        // ================================================

        // A 的子块坐标：
        // 行范围由 blockIdx.y 决定，列范围由当前 tile 决定
        int aRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        int aCol = tile * BLOCK_SIZE + threadIdx.x;

        // 边界检查：当 M 或 K 不是 BLOCK_SIZE 整数倍时防止越界
        if (aRow < M && aCol < K) {
            // 行优先存储：A[aRow][aCol] 的线性索引为 aRow * K + aCol
            sA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            // 越界位置填充 0，保证后续乘加不影响结果
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // B 的子块坐标：
        // 行范围由当前 tile 决定，列范围由 blockIdx.x 决定
        int bRow = tile * BLOCK_SIZE + threadIdx.y;
        int bCol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        if (bRow < K && bCol < N) {
            // 行优先存储：B[bRow][bCol] 的线性索引为 bRow * N + bCol
            sB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ================================================
        // 阶段二：线程同步 (Synchronization)
        // __syncthreads() 是 Block 级别的栅栏同步 (Barrier)。
        // 必须确保 Block 内全部 256 个线程都已完成 sA/sB 的写入，
        // 否则后续计算阶段中，某些线程可能读取到其他线程尚未加载的数据。
        // ================================================
        __syncthreads();

        // ================================================
        // 阶段三：基于 Shared Memory 的乘加计算
        // 此时数据已驻留在低延迟的 Shared Memory 中。
        // 每个线程独立完成其对应 C 元素的局部内积。
        // sA[threadIdx.y][k] 对应 A 的 row 行、当前 TILE 的第 k 列
        // sB[k][threadIdx.x] 对应 B 的 col 列、当前 TILE 的第 k 行
        // 注意：循环次数固定为 BLOCK_SIZE，有利于编译器展开和指令调度
        // ================================================
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // ================================================
        // 阶段四：再次同步，防止数据覆盖 (Race Condition)
        // 下一个 tile 循环迭代会重新向 sA 和 sB 写入新数据。
        // 若不加 __syncthreads()，先到达下一轮加载阶段的线程会覆盖 Shared Memory，
        // 而慢线程可能仍在阶段三读取旧数据，导致计算错误。
        // ================================================
        __syncthreads();
    }

    // ---------------------------------------------------------
    // 阶段五：写回 Global Memory
    // 仅对矩阵有效范围内的线程执行写操作，避免越界污染内存
    // ---------------------------------------------------------
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Host 端封装：启动 Tiled GEMM Kernel
 *
 * 负责配置线程网格 (Grid / Block) 并调用 Kernel。
 * Grid 维度由输出矩阵 C 的 M 和 N 决定，向上取整以覆盖所有元素。
 */
void gemm_gpu_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block 维度：二维，每个 Block 含 BLOCK_SIZE × BLOCK_SIZE 个线程
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Grid 维度：二维，x 方向覆盖 N 列，y 方向覆盖 M 行
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE 实现了整数向上取整
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动 Kernel，<<< >>> 为 CUDA 执行配置语法
    gemm_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);

    // cudaDeviceSynchronize() 阻塞 Host，直到当前设备上所有先前启动的 Kernel 完成。
    // 这对于在 Host 端进行准确的 GPU 计时或立即读取结果至关重要。
    cudaDeviceSynchronize();
}

// =============================================================
// 线程粗化 (Thread Coarsening) 扩展
// =============================================================

/**
 * @brief CUDA Kernel: 共享内存分块 + 线程粗化 GEMM
 *
 * 粗化策略（列方向 Coarsening Factor = 2）：
 * 标准 Tiled 版本中，一个 Block (BLOCK_SIZE × BLOCK_SIZE) 负责计算 BLOCK_SIZE × BLOCK_SIZE
 * 的输出子块。每个线程只算 1 个元素。
 *
 * 粗化版本中，每个线程负责计算**同行相邻的 2 个元素**（col 和 col+BLOCK_SIZE）。
 * 因此，一个 Block 实际负责计算的输出区域变为 BLOCK_SIZE 行 × (2*BLOCK_SIZE) 列。
 * Grid 的 x 维度相应减半：grid.x = (N + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE)。
 *
 * 为什么粗化能提升性能？
 * 1. 分摊 Shared Memory 加载开销：
 *    加载 A 的子块到 sA 的代价被分摊到了 2 个输出元素的计算上。
 *    每个线程从 sA 读取一次 a_val，但用它计算了 2 个独立的内积（sum0 和 sum1）。
 * 2. 更好地掩盖访存延迟：
 *    增加寄存器累加器数量（sum0, sum1）提升了指令级并行 (ILP)，
 *    使 Warp Scheduler 有更多独立指令可调度，从而更好地隐藏 Shared Memory 的读取延迟。
 * 3. 减少 Block 数量和调度开销：
 *    Grid 规模缩小为原来的一半（x 方向），减少了 Kernel 启动和 Block 调度的固定开销。
 *
 * Shared Memory 布局变化：
 * - sA 保持不变：[BLOCK_SIZE][BLOCK_SIZE]
 * - sB 加宽一倍：[BLOCK_SIZE][2*BLOCK_SIZE]，以容纳同时加载的 2 列 B 元素。
 * - 每个 Block 的 Shared Memory 总量从 2KB 增加到 3KB（BLOCK_SIZE=16 时），
 *   仍在 SM 限制内，对 Occupancy 影响极小。
 *
 * 加载阶段：
 * - 每个线程仍需加载 1 个 A 元素（因为 A 的列维度未变）。
 * - 每个线程需要加载 2 个 B 元素（因为 B 的列范围扩展了一倍），
 *   分别存入 sB[ty][tx] 和 sB[ty][tx+BLOCK_SIZE]。
 */
__global__ void gemm_tiled_coarse_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    // sA 保持不变，sB 宽度加倍以覆盖 2 倍列数
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][2 * BLOCK_SIZE];

    // 当前线程负责的第 1 个元素的全局坐标
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * (2 * BLOCK_SIZE) + threadIdx.x;

    // 2 个寄存器累加器：分别累加同行相邻的 2 个输出元素
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        // -------------------------------------------------
        // 阶段一：协作加载 A（每个线程 1 个元素）
        // -------------------------------------------------
        int aRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        int aCol = tile * BLOCK_SIZE + threadIdx.x;

        if (aRow < M && aCol < K) {
            sA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // -------------------------------------------------
        // 阶段一（续）：协作加载 B（每个线程 2 个元素）
        // 因为输出子块在列方向扩展了 2 倍，B 的加载范围也相应加倍
        // -------------------------------------------------
        int bRow = tile * BLOCK_SIZE + threadIdx.y;
        int bCol0 = blockIdx.x * (2 * BLOCK_SIZE) + threadIdx.x;
        int bCol1 = bCol0 + BLOCK_SIZE;

        if (bRow < K && bCol0 < N) {
            sB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol0];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < K && bCol1 < N) {
            sB[threadIdx.y][threadIdx.x + BLOCK_SIZE] = B[bRow * N + bCol1];
        } else {
            sB[threadIdx.y][threadIdx.x + BLOCK_SIZE] = 0.0f;
        }

        // -------------------------------------------------
        // 阶段二：同步，确保 Shared Memory 就绪
        // -------------------------------------------------
        __syncthreads();

        // -------------------------------------------------
        // 阶段三：基于 Shared Memory 的乘加计算（粗化版）
        // 每个线程同时计算 2 个输出元素：
        // sum0 对应 C[row][col]，sum1 对应 C[row][col+BLOCK_SIZE]
        // 同一个 a_val 被复用 2 次，减少了对 sA 的重复读取
        // -------------------------------------------------
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a_val = sA[threadIdx.y][k];
            sum0 += a_val * sB[k][threadIdx.x];
            sum1 += a_val * sB[k][threadIdx.x + BLOCK_SIZE];
        }

        // -------------------------------------------------
        // 阶段四：再次同步，防止下一轮加载覆盖数据
        // -------------------------------------------------
        __syncthreads();
    }

    // -------------------------------------------------
    // 阶段五：将 2 个累加结果写回 Global Memory
    // -------------------------------------------------
    if (row < M) {
        if (col < N) {
            C[row * N + col] = sum0;
        }
        if (col + BLOCK_SIZE < N) {
            C[row * N + col + BLOCK_SIZE] = sum1;
        }
    }
}

/**
 * @brief Host 端封装：启动 Tiled + Coarse GEMM Kernel
 *
 * Grid 维度调整：
 * - x 方向覆盖 N 列，但由于每个 Block 覆盖 2*BLOCK_SIZE 列，
 *   grid.x = (N + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE)
 * - y 方向不变，仍覆盖 M 行
 */
void gemm_gpu_tiled_coarse(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE),
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_tiled_coarse_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
