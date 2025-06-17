cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sgemm_optimization CUDA CXX)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 86)  # 根据你的GPU调整

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 限制每个线程使用的寄存器数量为128
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=128")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86 -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# 设置调试标志
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif()

# 添加头文件目录
include_directories(include)

# 添加源文件
set(SOURCES
    src/main.cu
    src/sgemm_v0_global_memory.cu
    src/sgemm_v1_shared_memory.cu
    src/sgemm_v2_tiling.cu
    src/sgemm_v3_vectorized.cu
    src/sgemm_v4_register.cu
    src/sgemm_v5_transpose.cu
    src/sgemm_v6_double_buffer.cu
    src/sgemm_v7_bank_conflict.cu
    src/sgemm_v8_tensor_core.cu
    src/sgemm_v9_tensor_core_bank_conflict.cu
    src/sgemm_v10_tensor_core_vectorized.cu
    src/sgemm_v11_tensor_core_cp_async.cu
    src/sgemm_v12_tensor_core_mma.cu
    src/sgemm_v13_tensor_core_mma_swizzle.cu
    src/sgemm_v14_tensor_core_mma_kStage.cu
    src/sgemm_cublas.cu
    src/sgemm_cublas_tensor_core.cu
    src/utils.cu
)

# 创建主程序
add_executable(sgemm_benchmark ${SOURCES})

# 创建测试程序
add_executable(test_sgemm src/test_sgemm.cu src/sgemm_v14_tensor_core_mma_kStage.cu)

# 链接CUDA库
target_link_libraries(sgemm_benchmark cublas cudart)
target_link_libraries(test_sgemm cublas cudart) 