#include "../include/sgemm_common.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CONST_FLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])

#define BLOCK_SIZE 16
#define TILE_SIZE 4

#define padding_A 0
#define padding_B 0

static __device__ 
uint32_t swizzle_A(uint32_t y, uint32_t x) {
    x >>= 2;
    return ((1 & x) << 4) ^ y;
}
// constexpr int STEP = BLOCK_SIZE * TILE_SIZE;

__global__ void sgemm_v7_1_bank_conflict_finished_kernel(
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

    __shared__ float A_shared[2][BK][BM + padding_A];
    __shared__ float B_shared[2][BK][BN + padding_B];
    
    float sum[TM][TN] = {0.0f};
    
    const float *A_start = A + by * BM * K;
    const float *B_start = B + bx * BN;
    
    const int sy_k = (ty * BLOCK_SIZE + tx) * 4 / BK;
    const int sx_k = (ty * BLOCK_SIZE + tx) * 4 % BK;

    const int sy_n = (ty * BLOCK_SIZE + tx) * 4 / BN;
    const int sx_n = (ty * BLOCK_SIZE + tx) * 4 % BN;

    float global_to_reg_A[4];
    float global_to_reg_B[4];

    float reg_A[TM];
    float reg_B[TN];

    {
        FLOAT4(global_to_reg_A[0]) = CONST_FLOAT4(A_start[OFFSET(sy_k, sx_k, K)]);
        FLOAT4(global_to_reg_B[0]) = CONST_FLOAT4(B_start[OFFSET(sy_n, sx_n, N)]);

        int swizzled_sy_k = swizzle_A(sy_k, sx_k);

        A_shared[0][sx_k][swizzled_sy_k] = global_to_reg_A[0];
        A_shared[0][sx_k + 1][swizzled_sy_k] = global_to_reg_A[1];
        A_shared[0][sx_k + 2][swizzled_sy_k] = global_to_reg_A[2];
        A_shared[0][sx_k + 3][swizzled_sy_k] = global_to_reg_A[3];

        FLOAT4(B_shared[0][sy_n][sx_n]) = FLOAT4(global_to_reg_B[0]);
        
        __syncthreads();
    }
    
    int choice = 0;

    #pragma unroll
    for (int s = BK; s < K; s += BK) {

        FLOAT4(global_to_reg_A[0]) = CONST_FLOAT4(A_start[OFFSET(sy_k, sx_k + s, K)]);
        FLOAT4(global_to_reg_B[0]) = CONST_FLOAT4(B_start[OFFSET(sy_n + s, sx_n, N)]);

        #pragma unroll
        for (int k = 0; k < BK; k++) {

            FLOAT4(reg_A[0]) = FLOAT4(A_shared[choice][k][swizzle_A(ty * TM / 2, k)]);
            FLOAT4(reg_A[4]) = FLOAT4(A_shared[choice][k][swizzle_A(ty * TM / 2 + BM / 2, k)]);

            FLOAT4(reg_B[0]) = FLOAT4(B_shared[choice][k][tx * TN / 2]);
            FLOAT4(reg_B[4]) = FLOAT4(B_shared[choice][k][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                   
                    sum[i][j] += reg_A[i] * reg_B[j];
                    
                }
            }
        }

        choice ^= 1;

        int swizzled_sy_k = swizzle_A(sy_k, sx_k);

        A_shared[choice][sx_k][swizzled_sy_k] = global_to_reg_A[0];
        A_shared[choice][sx_k + 1][swizzled_sy_k] = global_to_reg_A[1];
        A_shared[choice][sx_k + 2][swizzled_sy_k] = global_to_reg_A[2];
        A_shared[choice][sx_k + 3][swizzled_sy_k] = global_to_reg_A[3];

        FLOAT4(B_shared[choice][sy_n][sx_n]) = FLOAT4(global_to_reg_B[0]);
                
        __syncthreads();
    }

    {
        #pragma unroll
        for (int k = 0; k < BK; k++) {

            FLOAT4(reg_A[0]) = FLOAT4(A_shared[choice][k][swizzle_A(ty * TM / 2, k)]);
            FLOAT4(reg_A[4]) = FLOAT4(A_shared[choice][k][swizzle_A(ty * TM / 2 + BM / 2, k)]);

            FLOAT4(reg_B[0]) = FLOAT4(B_shared[choice][k][tx * TN / 2]);
            FLOAT4(reg_B[4]) = FLOAT4(B_shared[choice][k][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                   
                    sum[i][j] += reg_A[i] * reg_B[j];
                    
                }
            }
        }
    }

    float *C_start = C + by * BM * N + bx * BN;

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

void sgemm_v7_1_bank_conflict_finished(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x / 8,
        (dims.M + blockDim.y - 1) / blockDim.y / 8
    );
    
    sgemm_v7_1_bank_conflict_finished_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}