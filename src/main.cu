#include "../include/sgemm_common.h"
#include <iostream>
#include <vector>

// 声明不同版本的SGEMM实现
void sgemm_v0_global_memory(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v1_shared_memory(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v2_tiling(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v3_vectorized(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v4_register_blocking(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_cublas(float* C, const float* A, const float* B, const MatrixDims& dims);

int main() {
    // 设置矩阵维度
    std::vector<MatrixDims> test_cases = {
        {1024, 1024, 1024},  // 小矩阵
        {2048, 2048, 2048},  // 中等矩阵
        {4096, 4096, 4096}   // 大矩阵
    };

    // 测试每个维度
    for (const auto& dims : test_cases) {
        std::cout << "\nTesting matrix dimensions: " 
                  << dims.M << "x" << dims.K << " * " 
                  << dims.K << "x" << dims.N << "\n"
                  << "========================================\n";

        // 运行所有版本的SGEMM
        std::vector<PerformanceResult> results;
        
        results.push_back(runPerformanceTest(sgemm_v0_global_memory, dims, 100, "Global Memory"));
        // results.push_back(runPerformanceTest(sgemm_v1_shared_memory, dims, 100, "Shared Memory"));
        // results.push_back(runPerformanceTest(sgemm_v2_tiling, dims, 100, "Tiling"));
        // results.push_back(runPerformanceTest(sgemm_v3_vectorized, dims, 100, "Vectorized"));
        // results.push_back(runPerformanceTest(sgemm_v4_register_blocking, dims, 100, "Register Blocking"));
        results.push_back(runPerformanceTest(sgemm_cublas, dims, 100, "cuBLAS"));

        // 打印所有结果
        for (const auto& result : results) {
            printPerformanceResult(result);
        }
    }

    return 0;
} 