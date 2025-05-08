#include "../include/sgemm_common.h"
#define BLOCK_SIZE 16

__global__ void sgemm_shared_memory_kernel(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K) {
        
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    float sum = 0.0f;
    
    for (int s = 0; s < K; s += BLOCK_SIZE) {
        A_shared[ty][tx] = A[row * K + s + tx];
        B_shared[ty][tx] = B[(s + ty) * N + col];
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += A_shared[ty][k] * B_shared[k][tx];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

void sgemm_v1_shared_memory(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x,
        (dims.M + blockDim.y - 1) / blockDim.y
    );
    
    sgemm_shared_memory_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}