# GEMM 并行加速研究

> 面向网络大模型注意力机制的通用矩阵乘法（GEMM）并行加速研究

本项目基于 NVIDIA CUDA 实现了多种通用矩阵乘法（GEMM）算法，涵盖 CPU 串行基线、GPU 朴素并行以及 GPU Shared Memory 分块（Tiled）优化三种实现。通过系统性的性能基准测试，直观对比不同实现策略在计算吞吐量（GFLOPS）与加速比上的差异，为理解大模型核心算子的 GPU 优化提供可运行的教学与实验平台。

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
│   ├── gpu_tiled.cu            # GPU 分块 GEMM（Shared Memory 缓存优化）
│   └── utils.cu                # 工具函数实现（计时器、矩阵操作等）
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

程序会自动在矩阵维度 **1024、2048、4096** 下运行三种实现（CPU-Baseline、GPU-Naive、GPU-Tiled），并以 Markdown 表格形式输出性能数据，包括执行时间、GFLOPS、相对于 CPU 的加速比以及结果校验状态。

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
| 矩阵大小 | 算法类型 | 执行时间(ms) | GFLOPS | 加速比(vs CPU) | 结果校验 |
|---|---|---|---|---|---|
| 1024 | CPU-Baseline | 523.456 | 4.10 | 1.00x | PASS |
| 1024 | GPU-Naive | 12.345 | 173.82 | 42.40x | PASS |
| 1024 | GPU-Tiled | 2.567 | 835.12 | 203.92x | PASS |
```

> **注意**：实际性能数据高度依赖于 GPU 型号、显存带宽、CUDA 驱动版本以及系统负载，以上仅为格式示例。

---

## ⚙️ 可配置参数

| 参数位置 | 参数名 | 默认值 | 说明 |
|---|---|---|---|
| `src/main.cu:10` | `TEST_DIMS` | `{1024, 2048, 4096}` | 测试的矩阵维度（方阵 N×N） |
| `include/utils.h:14` | `EPSILON` | `1e-4f` | 矩阵比对时的浮点误差容忍度 |
| `src/gpu_naive.cu:11` | `NAIVE_BLOCK_SIZE` | `16` | Naive Kernel 的 Block 维度 |
| `src/gpu_tiled.cu:15` | `BLOCK_SIZE` | `16` | Tiled Kernel 的 Block / Tile 维度 |
| `CMakeLists.txt:26` | `CMAKE_CUDA_ARCHITECTURES` | `75 80 86` | 目标 GPU 架构代码生成 |

---

## 📝 扩展方向

本项目采用分块优化作为核心优化手段，但现代 CUDA GEMM 优化还有更多进阶方向可供探索：

1. **寄存器分块（Register Tiling）**：在 Tiled 基础上，让每个线程负责计算一个更大的子矩阵（如 4×4 或 8×8），减少 Shared Memory 的读写次数，进一步提升计算强度。
2. **双缓冲（Double Buffering）**：利用多余的 Shared Memory 或寄存器空间，在计算当前 TILE 的同时预加载下一个 TILE，用计算掩盖访存延迟。
3. **Warp-Level 原语**：使用 `__shfl_sync`（Shuffle）或 `wmma`（Tensor Core）进行细粒度的数据交换和矩阵乘加加速。
4. ** cuBLAS 对比**：将手写 Kernel 与 NVIDIA 官方高度优化的 cuBLAS 库进行性能对比，分析差距来源。
5. **混合精度**：引入 FP16 / BF16 / INT8 等低精度计算，结合 Tensor Core 实现更高的吞吐量和能效比。

---

## 📄 许可证

本项目仅供学习与研究使用。
