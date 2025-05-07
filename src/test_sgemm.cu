#include "../include/sgemm_common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// 声明要测试的函数
void sgemm_v1_shared_memory(float* C, const float* A, const float* B, const MatrixDims& dims);

int main() {
    // 设置测试矩阵维度
    MatrixDims dims = {512, 512, 512};
    
    // 分配主机内存
    float *h_A = new float[dims.M * dims.K];
    float *h_B = new float[dims.K * dims.N];
    float *h_C = new float[dims.M * dims.N];
    
    // 初始化输入矩阵
    for (int i = 0; i < dims.M * dims.K; i++) {
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < dims.K * dims.N; i++) {
        h_B[i] = 1.0f;
    }
    for (int i = 0; i < dims.M * dims.N; i++) {
        h_C[i] = 0.0f;
    }
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dims.M * dims.K * sizeof(float));
    cudaMalloc(&d_B, dims.K * dims.N * sizeof(float));
    cudaMalloc(&d_C, dims.M * dims.N * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_A, h_A, dims.M * dims.K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, dims.K * dims.N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, dims.M * dims.N * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Starting SGEMM test...\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", dims.M, dims.N, dims.K);
    
    // 运行SGEMM
    sgemm_v1_shared_memory(d_C, d_A, d_B, dims);
    
    // 检查错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // 同步设备
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_C, d_C, dims.M * dims.N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < dims.M * dims.N; i++) {
        if (h_C[i] != dims.K) {  // 因为所有输入都是1，结果应该是K
            printf("Error at position %d: expected %d, got %f\n", i, dims.K, h_C[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Test passed!\n");
    }
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return 0;
} 