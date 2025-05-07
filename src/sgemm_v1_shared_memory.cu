#include "../include/sgemm_common.h"

template<const int BLOCK_SIZE, const int TILE_SIZE>
__global__ void sgemm_kernel(
    float* C,
    const float* A,
    const float* B,
    const int M, const int N, const int K) {
        
    __shared__ float A_shared[BLOCK_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // if (tx == 0 && ty == 0) {
    //     printf("tx: %d, ty: %d\n", tx, ty);
    // }
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int s = 0; s < K; s += blockDim.x) {
        if (row < M && col < N) {
            if (tx + s < K) {
                A_shared[ty][tx + s] = A[row * K + tx + s];
            }
            if (ty + s < K) {
                B_shared[ty + s][tx] = B[(ty + s) * N + col];
            }
        }
    }

    __syncthreads();

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A_shared[ty][k] * B_shared[k][tx];
        }
        C[row * N + col] = sum;
    }
}

void sgemm_v1_shared_memory(float* C, const float* A, const float* B, const MatrixDims& dims) {
    constexpr int BLOCK_SIZE = 16;
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x,
        (dims.M + blockDim.y - 1) / blockDim.y
    );
    
    // 根据不同的K值选择不同的模板实例化
    if (dims.K == 128) {
        // printf("dims.K: %d\n", dims.K);
        sgemm_kernel<BLOCK_SIZE, 128><<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    } 
}