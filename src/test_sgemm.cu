#include "../include/sgemm_common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// 声明要测试的函数
void sgemm_v9_tensor_core_bank_conflict(float* C, const float* A, const float* B, const MatrixDims& dims);

int main() {
    printf("=== Starting CUDA SGEMM Test ===\n");
    
    // 设置测试矩阵维度
    MatrixDims dims = {1024, 1024, 1024};
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", dims.M, dims.N, dims.K);
    
    // 分配主机内存
    printf("Allocating host memory...\n");
    float *h_A = new float[dims.M * dims.K];
    float *h_B = new float[dims.K * dims.N];
    float *h_C = new float[dims.M * dims.N];
    
    // 初始化输入矩阵
    printf("Initializing input matrices...\n");
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
    printf("Allocating device memory...\n");
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dims.M * dims.K * sizeof(float));
    cudaMalloc(&d_B, dims.K * dims.N * sizeof(float));
    cudaMalloc(&d_C, dims.M * dims.N * sizeof(float));
    
    // 复制数据到设备
    printf("Copying data to device...\n");
    cudaMemcpy(d_A, h_A, dims.M * dims.K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, dims.K * dims.N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, dims.M * dims.N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 同步设备
    cudaDeviceSynchronize();
    printf("Device synchronized after memory operations\n");
    
    printf("About to launch SGEMM kernel...\n");
    // 运行SGEMM
    sgemm_v9_tensor_core_bank_conflict(d_C, d_A, d_B, dims);
    
    // 检查错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("Kernel launched, waiting for completion...\n");
    // 同步设备
    cudaDeviceSynchronize();
    printf("Device synchronized after kernel execution\n");
    
    // 复制结果回主机
    printf("Copying results back to host...\n");
    cudaMemcpy(h_C, d_C, dims.M * dims.N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("Verifying results...\n");
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
    printf("Cleaning up...\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    printf("=== Test completed ===\n");
    return 0;
} 