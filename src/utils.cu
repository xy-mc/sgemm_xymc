#include "../include/sgemm_common.h"
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// 声明cuBLAS版本的SGEMM
void sgemm_cublas(float* C, const float* A, const float* B, const MatrixDims& dims);

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

ErrorResult runErrorTest(
    void (*sgemm_func)(float*, const float*, const float*, const MatrixDims&),
    const MatrixDims& dims,
    const std::string& version) {
    
    // 分配设备内存
    float *d_A, *d_B, *d_C, *d_C_cublas;
    size_t size_A = dims.M * dims.K * sizeof(float);
    size_t size_B = dims.K * dims.N * sizeof(float);
    size_t size_C = dims.M * dims.N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_C_cublas, size_C);

    // 分配主机内存并初始化
    float *h_A = new float[dims.M * dims.K];
    float *h_B = new float[dims.K * dims.N];
    float *h_C = new float[dims.M * dims.N];
    float *h_C_cublas = new float[dims.M * dims.N];

    initializeMatrices(h_A, h_B, h_C, dims);

    // 复制数据到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 运行自定义SGEMM
    sgemm_func(d_C, d_A, d_B, dims);
    
    // 运行cuBLAS SGEMM
    sgemm_cublas(d_C_cublas, d_A, d_B, dims);

    // 复制结果回主机
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas, d_C_cublas, size_C, cudaMemcpyDeviceToHost);

    // 计算误差
    std::vector<float> abs_errors(dims.M * dims.N);
    std::vector<float> rel_errors(dims.M * dims.N);
    
    for (int i = 0; i < dims.M * dims.N; ++i) {
        float abs_error = std::abs(h_C[i] - h_C_cublas[i]);
        abs_errors[i] = abs_error;
        
        // 避免除以零
        if (std::abs(h_C_cublas[i]) > 1e-6) {
            rel_errors[i] = abs_error / std::abs(h_C_cublas[i]);
        } else {
            rel_errors[i] = 0.0f;
        }
    }

    // 计算统计值
    float max_abs_error = *std::max_element(abs_errors.begin(), abs_errors.end());
    float mean_abs_error = std::accumulate(abs_errors.begin(), abs_errors.end(), 0.0f) / abs_errors.size();
    float max_rel_error = *std::max_element(rel_errors.begin(), rel_errors.end());
    float mean_rel_error = std::accumulate(rel_errors.begin(), rel_errors.end(), 0.0f) / rel_errors.size();

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_cublas);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cublas;

    return {max_abs_error, mean_abs_error, max_rel_error, mean_rel_error, version};
}

void printErrorResult(const ErrorResult& result) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Version: " << result.version << "\n"
              << "Max Absolute Error: " << result.max_abs_error << "\n"
              << "Mean Absolute Error: " << result.mean_abs_error << "\n"
              << "Max Relative Error: " << result.max_rel_error << "\n"
              << "Mean Relative Error: " << result.mean_rel_error << "\n"
              << "----------------------------------------\n";
} 