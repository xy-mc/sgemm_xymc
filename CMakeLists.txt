cmake_minimum_required(VERSION 3.8)
project(sgemm_optimization CUDA CXX)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 86)  # 根据你的GPU调整

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加CUDA包含目录
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 添加可执行文件
add_executable(sgemm_benchmark 
    src/main.cu
    src/sgemm_v0_global_memory.cu
    # src/sgemm_v1_shared_memory.cu
    # src/sgemm_v2_tiling.cu
    # src/sgemm_v3_vectorized.cu
    # src/sgemm_v4_register_blocking.cu
    src/sgemm_cublas.cu
    src/utils.cu
)

# 链接CUDA库
target_link_libraries(sgemm_benchmark cublas cudart) 