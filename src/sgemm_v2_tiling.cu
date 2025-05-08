#include "../include/sgemm_common.h"

#define BLOCK_SIZE 16
#define TILE_SIZE 4

__global__ void sgemm_tiling_kernel(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K) {
    
    constexpr int STEP = BLOCK_SIZE * TILE_SIZE;

    __shared__ float A_shared[STEP][STEP];
    __shared__ float B_shared[STEP][STEP];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum[TILE_SIZE][TILE_SIZE] = {0.0f};
    
    for (int s = 0; s < K; s += STEP) {

        for (int i = 0; i < TILE_SIZE; i++) {

            for (int j = 0; j < TILE_SIZE; j++) {

                A_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 
                                                        A[(blockIdx.y * STEP + ty + i * BLOCK_SIZE) * K + tx + j * BLOCK_SIZE + s];

                B_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 
                                                        B[(ty + i * BLOCK_SIZE + s) * N + blockIdx.x * STEP + tx + j * BLOCK_SIZE];
            }
        }
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {

            for (int j = 0; j < TILE_SIZE; j++) {

                for (int k = 0; k < STEP; k++) {
                    sum[i][j] += A_shared[ty + i * BLOCK_SIZE][k] * B_shared[k][tx + j * BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TILE_SIZE; i++) {

        for (int j = 0; j < TILE_SIZE; j++) {

            C[(blockIdx.y * STEP + ty + i * BLOCK_SIZE) * N + blockIdx.x * STEP + tx + j * BLOCK_SIZE] = sum[i][j];
        }
    }
}

void sgemm_v2_tiling(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x / TILE_SIZE,
        (dims.M + blockDim.y - 1) / blockDim.y / TILE_SIZE
    );
    
    sgemm_tiling_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}