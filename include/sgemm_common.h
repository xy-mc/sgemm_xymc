#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <cmath>

// 性能测试结果结构
struct PerformanceResult {
    float gflops;
    float bandwidth;
    float time_ms;
    std::string version;
};

// 误差测试结果结构
struct ErrorResult {
    float max_abs_error;
    float mean_abs_error;
    float max_rel_error;
    float mean_rel_error;
    std::string version;
};

// 矩阵维度结构
struct MatrixDims {
    int M;  // 矩阵A的行数
    int N;  // 矩阵B的列数
    int K;  // 矩阵A的列数/矩阵B的行数
};

// 初始化矩阵
void initializeMatrices(float* A, float* B, float* C, const MatrixDims& dims);

// 验证结果
bool verifyResults(float* C1, float* C2, const MatrixDims& dims, float tolerance = 1e-5);

// 计算理论性能
float calculateTheoreticalGflops(const MatrixDims& dims, float time_ms);

// 运行性能测试
PerformanceResult runPerformanceTest(
    void (*sgemm_func)(float*, const float*, const float*, const MatrixDims&),
    const MatrixDims& dims,
    int num_iterations = 100,
    const std::string& version = "unknown"
);

// 运行误差测试
ErrorResult runErrorTest(
    void (*sgemm_func)(float*, const float*, const float*, const MatrixDims&),
    const MatrixDims& dims,
    const std::string& version = "unknown"
);

// 打印性能结果
void printPerformanceResult(const PerformanceResult& result);

// 打印误差结果
void printErrorResult(const ErrorResult& result); 