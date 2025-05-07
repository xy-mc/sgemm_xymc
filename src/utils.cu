#include "../include/sgemm_common.h"
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

void initializeMatrices(float* A, float* B, float* C, const MatrixDims& dims) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // 初始化矩阵A
    for (int i = 0; i < dims.M * dims.K; ++i) {
        A[i] = dis(gen);
    }

    // 初始化矩阵B
    for (int i = 0; i < dims.K * dims.N; ++i) {
        B[i] = dis(gen);
    }

    // 初始化矩阵C为0
    for (int i = 0; i < dims.M * dims.N; ++i) {
        C[i] = 0.0f;
    }
}

bool verifyResults(float* C1, float* C2, const MatrixDims& dims, float tolerance) {
    for (int i = 0; i < dims.M * dims.N; ++i) {
        if (std::abs(C1[i] - C2[i]) > tolerance) {
            std::cout << "Verification failed at index " << i 
                      << ": C1=" << C1[i] << ", C2=" << C2[i] << std::endl;
            return false;
        }
    }
    return true;
}

float calculateTheoreticalGflops(const MatrixDims& dims, float time_ms) {
    // 计算浮点运算次数：2 * M * N * K (每个元素需要K次乘法和K-1次加法)
    float flops = 2.0f * dims.M * dims.N * dims.K;
    // 转换为GFLOPS
    return (flops / (time_ms * 1e-3f)) / 1e9f;
}

PerformanceResult runPerformanceTest(
    void (*sgemm_func)(float*, const float*, const float*, const MatrixDims&),
    const MatrixDims& dims,
    int num_iterations,
    const std::string& version) {
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    size_t size_A = dims.M * dims.K * sizeof(float);
    size_t size_B = dims.K * dims.N * sizeof(float);
    size_t size_C = dims.M * dims.N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 分配主机内存并初始化
    float *h_A = new float[dims.M * dims.K];
    float *h_B = new float[dims.K * dims.N];
    float *h_C = new float[dims.M * dims.N];

    initializeMatrices(h_A, h_B, h_C, dims);

    // 复制数据到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 预热
    sgemm_func(d_C, d_A, d_B, dims);
    cudaDeviceSynchronize();

    // 性能测试
    float total_time = 0.0f;
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        sgemm_func(d_C, d_A, d_B, dims);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::milli>(end - start).count();
        total_time += time;
    }

    float avg_time = total_time / num_iterations;
    float gflops = calculateTheoreticalGflops(dims, avg_time);

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return {gflops, 0.0f, avg_time, version};
}

void printPerformanceResult(const PerformanceResult& result) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Version: " << result.version << "\n"
              << "Time: " << result.time_ms << " ms\n"
              << "Performance: " << result.gflops << " GFLOPS\n"
              << "----------------------------------------\n";
} 