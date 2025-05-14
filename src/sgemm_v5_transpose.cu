#include "../include/sgemm_common.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CONST_FLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])

#define BLOCK_SIZE 16
#define TILE_SIZE 4

// constexpr int STEP = BLOCK_SIZE * TILE_SIZE;

__global__ void sgemm_transpose_kernel(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K) {
    
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;

    constexpr int TM = BM / BLOCK_SIZE;
    constexpr int TN = BN / BLOCK_SIZE;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float A_shared[BK][BM];
    __shared__ float B_shared[BK][BN];
    
    float sum[TM][TN] = {0.0f};
    
    const float *A_start = A + by * BM * K;
    const float *B_start = B + bx * BN;
    
    const int sy_k = (ty * BLOCK_SIZE + tx) * 4 / BK;
    const int sx_k = (ty * BLOCK_SIZE + tx) * 4 % BK;

    const int sy_n = (ty * BLOCK_SIZE + tx) * 4 / BN;
    const int sx_n = (ty * BLOCK_SIZE + tx) * 4 % BN;

    float global_to_reg_A[4];
    // float global_to_reg_B[4];
    float reg_A[TM];
    float reg_B[TN];

    #pragma unroll
    for (int s = 0; s < K; s += BK) {

        FLOAT4(global_to_reg_A[0]) = CONST_FLOAT4(A_start[OFFSET(sy_k, sx_k + s, K)]);

        A_shared[sx_k][sy_k] = global_to_reg_A[0];
        A_shared[sx_k + 1][sy_k] = global_to_reg_A[1];
        A_shared[sx_k + 2][sy_k] = global_to_reg_A[2];
        A_shared[sx_k + 3][sy_k] = global_to_reg_A[3];
        
        // FLOAT4(A_shared[sy_k][sx_k])
        //         = CONST_FLOAT4(A_start[OFFSET(sy_k, sx_k + s, K)]);

        FLOAT4(B_shared[sy_n][sx_n])
                = CONST_FLOAT4(B_start[OFFSET(sy_n + s, sx_n, N)]);
            
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {

            FLOAT4(reg_A[0]) = FLOAT4(A_shared[k][ty * TM / 2]);
            FLOAT4(reg_A[4]) = FLOAT4(A_shared[k][ty * TM / 2 + BM / 2]);

            FLOAT4(reg_B[0]) = FLOAT4(B_shared[k][tx * TN / 2]);
            FLOAT4(reg_B[4]) = FLOAT4(B_shared[k][tx * TN / 2 + BN / 2]);

            for (int i = 0; i < TM; i++) {
                // reg_A[i] = A_shared[ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    // sum[i][j] += A_shared[ty + i * BLOCK_SIZE][k] * B_shared[k][tx + j * BLOCK_SIZE];
                    sum[i][j] += reg_A[i] * reg_B[j];
                    
                }
            }
        }
        __syncthreads();
    }

    float *C_start = C + by * BM * N + bx * BN;

    // #pragma unroll
    // for (int i = 0; i < TM; i++) {

    //     for (int j = 0; j < TN; j += 4) {

    //         FLOAT4(C_start[OFFSET(ty * TM + i, tx * TN + j, N)])
    //                 = FLOAT4(sum[i][j]);
                    
    //     }
    // }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {

        FLOAT4(C_start[OFFSET(ty * TM / 2 + i, tx * TN / 2, N)])
                = FLOAT4(sum[i][0]);

        FLOAT4(C_start[OFFSET(ty * TM / 2 + i, tx * TN / 2 + BN / 2, N)])
                = FLOAT4(sum[i][TN / 2]);

        FLOAT4(C_start[OFFSET(ty * TM / 2 + i + BM / 2, tx * TN / 2, N)])
                = FLOAT4(sum[i + TM / 2][0]);

        FLOAT4(C_start[OFFSET(ty * TM / 2 + i + BM / 2, tx * TN / 2 + BN / 2, N)])
                = FLOAT4(sum[i + TM / 2][TN / 2]);
    }
}

void sgemm_v5_transpose(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x / 8,
        (dims.M + blockDim.y - 1) / blockDim.y / 8
    );
    
    sgemm_transpose_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}