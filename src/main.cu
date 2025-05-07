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
        {128, 128, 128},
        // {1024, 1024, 1024},
        // {2048, 2048, 2048},
        // {4096, 4096, 4096}
    };

    // 测试每个维度
    for (const auto& dims : test_cases) {
        std::cout << "\nTesting matrix dimensions: " 
                  << dims.M << "x" << dims.K << " * " 
                  << dims.K << "x" << dims.N << "\n"
                  << "========================================\n";

        // 创建并初始化矩阵数据
        MatrixData data(dims);
        data.initialize();
        data.copyToDevice();

        // 运行所有版本的SGEMM
        std::vector<PerformanceResult> results;
        
        results.push_back(runPerformanceTest(sgemm_v0_global_memory, data, 100, "Global Memory"));
        results.push_back(runPerformanceTest(sgemm_v1_shared_memory, data, 100, "Shared Memory"));
        // results.push_back(runPerformanceTest(sgemm_v2_tiling, data, 100, "Tiling"));
        // results.push_back(runPerformanceTest(sgemm_v3_vectorized, data, 100, "Vectorized"));
        // results.push_back(runPerformanceTest(sgemm_v4_register_blocking, data, 100, "Register Blocking"));
        results.push_back(runPerformanceTest(sgemm_cublas, data, 100, "cuBLAS"));

        // 打印所有结果
        for (const auto& result : results) {
            printPerformanceResult(result);
        }

        // 进行误差测试
        std::cout << "\nError Analysis (compared with cuBLAS):\n"
                  << "========================================\n";
        
        std::vector<ErrorResult> error_results;
        error_results.push_back(runErrorTest(sgemm_v0_global_memory, data, "Global Memory"));
        error_results.push_back(runErrorTest(sgemm_v1_shared_memory, data, "Shared Memory"));
        // error_results.push_back(runErrorTest(sgemm_v2_tiling, data, "Tiling"));
        // error_results.push_back(runErrorTest(sgemm_v3_vectorized, data, "Vectorized"));
        // error_results.push_back(runErrorTest(sgemm_v4_register_blocking, data, "Register Blocking"));

        for (const auto& result : error_results) {
            printErrorResult(result);
        }
    }

    return 0;
} 