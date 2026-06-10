cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------
# 项目名称与语言支持：同时启用 C++ 和 CUDA
# ---------------------------------------------------------
project(GEMM_Acceleration LANGUAGES CXX CUDA)

# ---------------------------------------------------------
# 设置 C++ 与 CUDA 语言标准
# ---------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# 查找 CUDA 工具包 (CMake 3.18+ 推荐方式)
# ---------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------
# 指定 CUDA 架构代码生成目标
# 75: Turing (RTX 20系列), 80: Ampere (A100/RTX 30), 86: Ampere (RTX 30高端)
# 请根据实际 GPU 架构增删，以生成最优机器码
# ---------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# ---------------------------------------------------------
# 收集所有源文件：.cpp (CPU 端) + .cu (CUDA 端)
# ---------------------------------------------------------
set(SOURCES
    src/main.cu
    src/utils.cu
    src/cpu_gemm.cpp
    src/gpu_naive.cu
    src/gpu_tiled.cu
    src/gpu_tiled_padded.cu
)

# ---------------------------------------------------------
# 定义可执行文件目标
# ---------------------------------------------------------
add_executable(gemm_acc ${SOURCES})

# ---------------------------------------------------------
# 添加头文件搜索路径 (include/)
# ---------------------------------------------------------
target_include_directories(gemm_acc PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ---------------------------------------------------------
# 链接 CUDA 运行时库 (cudart)
# ---------------------------------------------------------
target_link_libraries(gemm_acc PRIVATE
    CUDA::cudart
    dl
)

# ---------------------------------------------------------
# 设置可执行文件输出目录
# ---------------------------------------------------------
set_target_properties(gemm_acc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
