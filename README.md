# GEMM 并行加速研究

> 面向网络大模型注意力机制的通用矩阵乘法（GEMM）并行加速研究

本项目基于 NVIDIA CUDA 实现了多种通用矩阵乘法（GEMM）算法，涵盖 CPU 串行基线、GPU 朴素并行、GPU Shared Memory 分块（Tiled）优化、预填充对齐（Padded）、线程粗化（Coarsening）以及 cuBLAS 参考实现。通过系统性的性能基准测试，直观对比不同实现策略在计算吞吐量（GFLOPS）与加速比上的差异，为理解大模型核心算子的 GPU 优化提供可运行的教学与实验平台。

---

## 📁 项目结构

```
GEMM_Acceleration/
├── CMakeLists.txt              # CMake 构建配置
├── README.md                   # 本文件
├── .gitignore                  # Git 忽略规则
├── include/
│   ├── gemm_kernels.cuh        # CPU/GPU GEMM 接口声明
│   └── utils.h                 # 计时器、矩阵初始化、结果比对等工具
├── src/
│   ├── main.cu                 # 主程序：驱动基准测试并输出结果表格
│   ├── cpu_gemm.cpp            # CPU 串行 GEMM（缓存优化三重循环）
│   ├── gpu_naive.cu            # GPU 朴素 GEMM（Global Memory 直接访问）
│   ├── gpu_tiled.cu            # GPU 分块 GEMM（Shared Memory 缓存优化 + 线程粗化）
│   ├── gpu_tiled_padded.cu     # GPU 分块 + 预填充对齐 + cuBLAS 封装
│   └── utils.cu                # 工具函数实现（计时器、矩阵操作等）
├── scripts/
│   ├── parse_benchmark.py      # 解析 gemm_acc Markdown 输出 → CSV 文件
│   └── plot_results.py         # 读取 CSV → 可视化图表（柱状图/折线图）
├── results/                    # 输出目录（.gitignore 忽略）
└── build/                      # 构建输出目录（.gitignore 忽略）
```

---

## 🚀 快速开始

### 环境要求

- **操作系统**：Linux（推荐）或 Windows with WSL2
- **CMake**：≥ 3.18
- **CUDA Toolkit**：≥ 11.0（需支持 `sm_75` 及以上架构）
- **NVIDIA GPU**：Turing（RTX 20 系列）、Ampere（A100 / RTX 30 系列）或更新架构
- **C++ 编译器**：支持 C++17（如 GCC 7+、Clang 5+）

### 编译

```bash
# 创建并进入构建目录
mkdir -p build && cd build

# 生成构建系统
cmake ..

# 编译
make -j$(nproc)
```

编译完成后，可执行文件位于 `build/gemm_acc`。

> **提示**：`CMakeLists.txt` 中默认的 `CMAKE_CUDA_ARCHITECTURES` 设置为 `75 80 86`，请根据你的 GPU 架构进行调整。例如：
> - RTX 2080 Ti / T4 → `75`
> - A100 → `80`
> - RTX 3090 / 4090 → `86`
> - RTX 4090 / Ada → `89`
> - H100 → `90`

### 运行

```bash
./gemm_acc
```

程序会自动在以下矩阵维度（方阵 + 非方阵）下运行全部算法实现，并以 Markdown 表格形式输出性能数据，包括执行时间、GFLOPS、相对于 CPU 的加速比以及结果校验状态。

测试维度：
- **方阵**：`1024×1024×1024`、`2048×2048×2048`、`4096×4096×4096`
- **非方阵**（模拟注意力机制实际特征）：
  - `4096×4096×128`：长序列 × 头维度（Q×K^T 场景）
  - `4096×128×128`：扁平投影
  - `8192×8192×64`：超长序列

---

## 📈 数据解析与可视化

### 脚本说明

| 脚本 | 功能 | 依赖 |
|---|---|---|
| [`scripts/parse_benchmark.py`](scripts/parse_benchmark.py) | 解析 `gemm_acc` 的 Markdown 表格输出，保存为 CSV 文件 | Python 3 标准库 |
| [`scripts/plot_results.py`](scripts/plot_results.py) | 读取 CSV，绘制 GFLOPS / 执行时间 / 加速比图表 | `matplotlib` |

### 用法

```bash
# 编译（如尚未编译）
cd build && cmake .. && make -j$(nproc)

# 运行基准测试，解析并可视化
./gemm_acc 2>&1 | python3 ../scripts/parse_benchmark.py && python3 ../scripts/plot_results.py
```

### 输出文件

运行后在 `results/` 目录下生成：

| 文件 | 说明 |
|---|---|
| `benchmark_results.csv` | 结构化 CSV，含 M/N/K/算法/时间/GFLOPs/加速比/校验结果 |
| `benchmark_gflops_bar.png` | GFLOPS 分组柱状图（各算法按矩阵大小分组） |
| `benchmark_time_bar.png` | 执行时间分组柱状图 |
| `benchmark_gflops_line.png` | GFLOPS 折线图（GPU 算法随规模变化趋势） |
| `benchmark_speedup_line.png` | 加速比折线图（相对 CPU 的加速比） |

### CSV 格式

```
M,N,K,description,algorithm,time_ms,gflops,speedup,verified
1024,1024,1024,Square,CPU-Baseline,2181.281,0.98,1.0,PASS
1024,1024,1024,Square,GPU-Naive,3.703,579.96,589.09,PASS
...
```

---

## 🧮 算法说明

### 1. CPU-Baseline（串行基线）

- **文件**：[`src/cpu_gemm.cpp`](src/cpu_gemm.cpp)
- **核心思想**：经典三重循环，但采用 `i → k → j` 的循环顺序优化缓存局部性。
  - `B[k][j]` 和 `C[i][j]` 均为行优先连续访问，最大化 CPU L1/L2/L3 缓存命中率。
  - `A[i][k]` 在内层循环中提取为局部变量，避免重复索引计算。
- **用途**：作为正确性验证的“黄金标准”以及 GPU 加速比的性能基准。

### 2. GPU-Naive（朴素并行）

- **文件**：[`src/gpu_naive.cu`](src/gpu_naive.cu)
- **核心思想**：每个 CUDA 线程负责计算输出矩阵 `C` 中的一个元素。
- **性能瓶颈**：
  - `A[row][k]` 沿行连续读取，满足 Global Memory 合并访问。
  - **`B[k][col]` 沿列跨步访问（步长为 N）**，同一 Warp 内相邻线程的内存地址相距甚远，严重不满足合并访问条件，导致显存带宽利用率极低。
  - 所有数据均来自高延迟 Global Memory，无 Shared Memory 缓存，实际性能通常只有 GPU 理论峰值的 **1%~5%**。
- **用途**：展示最直观的并行映射思路，作为后续优化的对比基准。

### 3. GPU-Tiled（分块优化）

- **文件**：[`src/gpu_tiled.cu`](src/gpu_tiled.cu)
- **核心思想**：利用 CUDA **Shared Memory** 将 Global Memory 上的数据分块（Tile）加载到快速的片上缓存，大幅减少重复的显存访问。
- **优化细节**：
  - **分块策略**：将 K 维度切分为多个宽度为 `BLOCK_SIZE`（默认 16）的 TILE。
  - **协作加载**：Block 内所有线程协作，将当前 TILE 对应的 `A` 子块和 `B` 子块从 Global Memory 一次性加载到 Shared Memory。
  - **数据复用**：一个 TILE 加载到 Shared Memory 后，Block 内的 `BLOCK_SIZE` 个线程可以复用该数据 `BLOCK_SIZE` 次。
  - **访存合并**：加载 TILE 时，同一 Warp 内相邻线程访问的 Global Memory 地址连续递增，满足合并访问条件。
  - **同步原语**：通过 `__syncthreads()` 保证加载阶段与计算阶段的严格顺序，避免数据竞争。
- **性能收益**：相比 Naive 版本，Tiled 实现通常能获得 **数倍至数十倍** 的性能提升，且随着矩阵规模增大，收益愈发明显。

### 4. GPU-Tiled-Padded（预填充对齐）

- **文件**：[`src/gpu_tiled_padded.cu`](src/gpu_tiled_padded.cu)
- **核心思想**：在 Host 端将输入矩阵预填充（Padding）至 `BLOCK_SIZE` 的整数倍，使 Kernel 内部无需任何边界检查（`if (row < M)` 等分支全部消除）。
- **优化细节**：
  - **消除 Warp Divergence**：所有线程执行路径完全一致，无条件分支，提升指令流水线效率。
  - **协作加载**：与 Tiled 版本相同，利用 Shared Memory 分块加载。
  - **端到端开销**：计时包含 Host 端 Padding 内存分配、数据拷贝及 Kernel 执行完整流程。
- **适用场景**：M/N/K 不是 `BLOCK_SIZE` 整数倍的非方阵场景（如 `4096×4096×128`）。
- **性能收益**：在非对齐尺寸下，相比 Tiled 版本省去大量分支判断开销，通常有 **10%~20%** 的额外提升。

### 5. GPU-Tiled-Coarse（线程粗化）

- **文件**：[`src/gpu_tiled.cu`](src/gpu_tiled.cu)
- **核心思想**：在 Tiled 基础上进一步**线程粗化（Thread Coarsening）**，每个线程负责计算输出矩阵中**同行相邻的 2 个元素**（Coarsening Factor = 2，列方向）。
- **优化细节**：
  - **分摊 Shared Memory 加载开销**：加载 A 的子块代价被 2 个输出元素分摊，同一 `a_val` 被复用 2 次计算 `sum0` 和 `sum1`。
  - **增加寄存器累加器**：每个线程持有 `sum0` 和 `sum1` 两个寄存器累加器，提升指令级并行（ILP），使 Warp Scheduler 更好地掩盖访存延迟。
  - **减少 Grid 规模**：Grid x 维度缩减为原来的一半，降低 Kernel 启动和 Block 调度的固定开销。
  - **Shared Memory 布局**：sB 宽度加倍至 `2*BLOCK_SIZE`，以同时覆盖 2 列 B 元素；sA 保持不变。
- **性能收益**：相比 Tiled 版本，在扁平矩阵（如 `4096×128×128`）上表现尤为明显，通常有 **20%~40%** 的额外提升。

### 6. GPU-cuBLAS（官方库参考）

- **文件**：[`src/gpu_tiled_padded.cu`](src/gpu_tiled_padded.cu)
- **核心思想**：通过动态加载（`dlopen`）调用 NVIDIA 官方高度优化的 cuBLAS 库，作为手写 Kernel 的性能上限参考。
- **动态加载策略**：避免静态链接 cuBLAS 在部分环境（如 WSL2 + CUDA 12.x）下导致的 CUDA 运行时初始化失败问题。
- **用途**：作为性能对比的"天花板"，帮助分析手写 Kernel 与工业级优化库之间的差距来源。

---

## 📊 性能基准测试

程序在 [`src/main.cu`](src/main.cu) 中实现了完整的基准测试流程：

1. **矩阵生成**：使用固定种子（`42`）的随机数生成器初始化输入矩阵 `A` 和 `B`，保证实验可复现。
2. **数据拷贝**：输入矩阵从 Host 端到 Device 端仅拷贝一次，所有 GPU Kernel 共享同一份设备内存。
3. **预热（Warm-up）**：正式计时前额外执行一次 Kernel，排除 CUDA Context 初始化、JIT 缓存等冷启动噪声。
4. **精确计时**：
   - CPU 端使用 `std::chrono::high_resolution_clock`。
   - GPU 端使用 `cudaEventRecord`，精确测量 Kernel 实际执行耗时。
5. **结果校验**：
   - GPU 结果与 CPU 黄金标准进行逐元素比对，同时检查绝对误差与相对误差，阈值 `1e-4`。
   - 当 CPU 基线因规模过大被跳过时（如 4096），GPU-Tiled 的结果会与 GPU-Naive 进行交叉验证。
6. **指标计算**：自动计算 GFLOPS 与相对于 CPU 的加速比，并以 Markdown 表格形式输出。

### 输出示例

```markdown
| 矩阵大小 | 描述 | 算法类型 | 执行时间(ms) | GFLOPS | 加速比(vs CPU) | 结果校验 |
|---|---|---|---|---|---|---|
| 1024×1024×1024 | Square | CPU-Baseline | 523.456 | 4.10 | 1.00x | PASS |
| 1024×1024×1024 | Square | GPU-Naive | 12.345 | 173.82 | 42.40x | PASS |
| 1024×1024×1024 | Square | GPU-Tiled | 2.567 | 835.12 | 203.92x | PASS |
| 1024×1024×1024 | Square | GPU-Tiled-Padded | 2.345 | 912.45 | 223.15x | PASS |
| 1024×1024×1024 | Square | GPU-Tiled-Coarse | 1.890 | 1134.21 | 276.95x | PASS |
| 1024×1024×1024 | Square | GPU-cuBLAS | 0.823 | 2601.34 | 636.08x | PASS |
| 4096×4096×128 | Long-seq x head-dim | GPU-Tiled-Coarse | 4.521 | 1892.34 | - | PASS |
| 4096×128×128 | Flat projection | GPU-Tiled-Coarse | 0.892 | 7456.12 | - | PASS |
```

> **注意**：实际性能数据高度依赖于 GPU 型号、显存带宽、CUDA 驱动版本以及系统负载，以上仅为格式示例。

---

## ⚙️ 可配置参数

| 参数位置 | 参数名 | 默认值 | 说明 |
|---|---|---|---|
| `src/main.cu:10` | `TEST_CASES` | 见下方列表 | 测试用例数组（结构体：M/N/K/desc） |
| `include/utils.h:14` | `EPSILON` | `1e-4f` | 矩阵比对时的浮点误差容忍度 |
| `src/gpu_naive.cu:11` | `NAIVE_BLOCK_SIZE` | `16` | Naive Kernel 的 Block 维度 |
| `src/gpu_tiled.cu:15` | `BLOCK_SIZE` | `16` | Tiled / Coarse Kernel 的 Block / Tile 维度 |
| `CMakeLists.txt:26` | `CMAKE_CUDA_ARCHITECTURES` | `75 80 86` | 目标 GPU 架构代码生成 |

**`TEST_CASES` 默认配置（方阵 + 非方阵）：**

| M | N | K | 描述 | 模拟场景 |
|---|---|---|---|---|
| 1024 | 1024 | 1024 | Square | 方阵基线 |
| 2048 | 2048 | 2048 | Square | 方阵中等规模 |
| 4096 | 4096 | 4096 | Square | 方阵大规模 |
| 4096 | 4096 | 128 | Long-seq x head-dim | Q×K^T 注意力场景 |
| 4096 | 128 | 128 | Flat projection | 扁平投影层 |
| 8192 | 8192 | 64 | Ultra-long seq | 超长序列（可选）|

---

## 📝 扩展方向

本项目已实现分块优化（Tiled）、预填充对齐（Padded）、线程粗化（Coarsening）以及 cuBLAS 参考对比，现代 CUDA GEMM 优化还有更多进阶方向可供探索：

1. **寄存器分块（Register Tiling）**：在 Tiled 基础上，让每个线程负责计算一个更大的子矩阵（如 4×4 或 8×8），减少 Shared Memory 的读写次数，进一步提升计算强度。
2. **双缓冲（Double Buffering）**：利用多余的 Shared Memory 或寄存器空间，在计算当前 TILE 的同时预加载下一个 TILE，用计算掩盖访存延迟。
3. **Warp-Level 原语**：使用 `__shfl_sync`（Shuffle）或 `wmma`（Tensor Core）进行细粒度的数据交换和矩阵乘加加速。
4. **混合精度**：引入 FP16 / BF16 / INT8 等低精度计算，结合 Tensor Core 实现更高的吞吐量和能效比。

---

## 📄 许可证

本项目仅供学习与研究使用。
