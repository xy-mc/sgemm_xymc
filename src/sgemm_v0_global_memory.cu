#include "../include/sgemm_common.h"

__global__ void sgemm_kernel(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void sgemm_v0_global_memory(float* C, const float* A, const float* B, const MatrixDims& dims) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x,
        (dims.M + blockDim.y - 1) / blockDim.y
    );
    
    sgemm_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
} 