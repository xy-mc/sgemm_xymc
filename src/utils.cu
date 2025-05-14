#include "../include/sgemm_common.h"
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// 声明cuBLAS版本的SGEMM
void sgemm_cublas(float* C, const float* A, const float* B, const MatrixDims& dims);

// MatrixData 构造函数
MatrixData::MatrixData(const MatrixDims& dims) : dims(dims) {
    size_A = dims.M * dims.K * sizeof(float);
    size_B = dims.K * dims.N * sizeof(float);
    size_C = dims.M * dims.N * sizeof(float);

    // 分配设备内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 分配主机内存
    h_A = new float[dims.M * dims.K];
    h_B = new float[dims.K * dims.N];
    h_C = new float[dims.M * dims.N];
}

// MatrixData 析构函数
MatrixData::~MatrixData() {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

// 初始化矩阵数据
void MatrixData::initialize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // 初始化矩阵A和B
    for (int i = 0; i < dims.M * dims.K; ++i) {
        h_A[i] = dis(gen);
    }
    for (int i = 0; i < dims.K * dims.N; ++i) {
        h_B[i] = dis(gen);
    }
    // 初始化矩阵C为0
    std::fill(h_C, h_C + dims.M * dims.N, 0.0f);
}

// 复制数据到设备
void MatrixData::copyToDevice() {
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
}

// 复制数据到主机
void MatrixData::copyToHost() {
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
}

float calculateTheoreticalGflops(const MatrixDims& dims, float time_ms) {
    float flops = 2.0f * dims.M * dims.N * dims.K;
    return (flops / (time_ms * 1e-3f)) / 1e9f;
}

PerformanceResult runPerformanceTest(
    void (*sgemm_func)(float*, const float*, const float*, const MatrixDims&),
    MatrixData& data,
    int num_iterations,
    const std::string& version) {
    
    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    sgemm_func(data.d_C, data.d_A, data.d_B, data.dims);
    cudaDeviceSynchronize();

    // 性能测试
    float total_time = 0.0f;
    for (int i = 0; i < num_iterations; ++i) {
        cudaEventRecord(start);
        sgemm_func(data.d_C, data.d_A, data.d_B, data.dims);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time = 0.0f;
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }

    // 清理CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float avg_time = total_time / num_iterations;
    float gflops = calculateTheoreticalGflops(data.dims, avg_time);

    return {gflops, 0.0f, avg_time, version};
}

void printPerformanceResult(const PerformanceResult& result) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Version: " << result.version << "\n"
              << "Time: " << result.time_ms << " ms\n"
              << "Performance: " << result.gflops << " GFLOPS\n"
              << "----------------------------------------\n";
}

ErrorResult runErrorTest(
    void (*sgemm_func)(float*, const float*, const float*, const MatrixDims&),
    MatrixData& data,
    const std::string& version) {
    
    // 分配额外的设备内存用于cuBLAS结果
    float *d_C_cublas;
    cudaMalloc(&d_C_cublas, data.size_C);
    float *h_C_cublas = new float[data.dims.M * data.dims.N];

    // 运行自定义SGEMM
    sgemm_func(data.d_C, data.d_A, data.d_B, data.dims);
    
    // 运行cuBLAS SGEMM
    sgemm_cublas(d_C_cublas, data.d_A, data.d_B, data.dims);

    // 复制结果回主机
    data.copyToHost();
    cudaMemcpy(h_C_cublas, d_C_cublas, data.size_C, cudaMemcpyDeviceToHost);

    // 计算误差
    std::vector<float> abs_errors(data.dims.M * data.dims.N);
    std::vector<float> rel_errors(data.dims.M * data.dims.N);
    
    for (int i = 0; i < data.dims.M * data.dims.N; ++i) {
        float abs_error = std::abs(data.h_C[i] - h_C_cublas[i]);
        abs_errors[i] = abs_error;
        
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
    cudaFree(d_C_cublas);
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