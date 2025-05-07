#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <cmath>

// 性能测试结果结构
struct PerformanceResult {
    float gflops;
    float error;
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
    int M, N, K;
};

// 内存管理结构体
struct MatrixData {
    float *d_A, *d_B, *d_C;  // 设备内存
    float *h_A, *h_B, *h_C;  // 主机内存
    size_t size_A, size_B, size_C;  // 内存大小
    MatrixDims dims;  // 矩阵维度

    MatrixData(const MatrixDims& dims);
    ~MatrixData();
    void initialize();  // 初始化矩阵数据
    void copyToDevice();  // 复制数据到设备
    void copyToHost();  // 复制数据到主机
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
    MatrixData& data,
    int num_iterations,
    const std::string& version);

// 运行误差测试
ErrorResult runErrorTest(
    void (*sgemm_func)(float*, const float*, const float*, const MatrixDims&),
    MatrixData& data,
    const std::string& version);

// 打印性能结果
void printPerformanceResult(const PerformanceResult& result);

// 打印误差结果
void printErrorResult(const ErrorResult& result); 