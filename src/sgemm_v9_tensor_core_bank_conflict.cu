#include "../include/sgemm_common.h"
#include <mma.h>
using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CONST_FLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])

#define BLOCK_SIZE 16
#define TILE_SIZE 4

#define padding_A 4
#define padding_B 8

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

// constexpr int Warp_M = 2;
// constexpr int Warp_N = 4;

// constexpr int STEP = BLOCK_SIZE * TILE_SIZE;

__global__ void sgemm_tensor_core_bank_conflict_kernel(
    float* C,
    const float* A, 
    const float* B,
    int M, int N, int K) {

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;

    // constexpr int TM = BM / BLOCK_SIZE;
    // constexpr int TN = BN / BLOCK_SIZE;

    // const int bx = blockIdx.x;
    // const int by = blockIdx.y;

    const int tx = threadIdx.x;
    // const int ty = threadIdx.y;

    const int warp_id = tx >> 5;
    const int lane_id = tx & 31;

    const int tx_A = lane_id & 3;
    const int ty_A = lane_id >> 2;

    const int tx_B = lane_id & 7;
    const int ty_B = lane_id >> 3;

    __shared__ float A_shared[2][BM][BK + padding_A];
    __shared__ float B_shared[2][BK][BN + padding_B];
    
    // float sum[TM][TN] = {0.0f};
    
    const float *A_start = A + blockIdx.y * BM * K;
    const float *B_start = B + blockIdx.x * BN;
    
    // const int sy_k = (ty * BLOCK_SIZE + tx) * 4 / BK;
    // const int sx_k = (ty * BLOCK_SIZE + tx) * 4 % BK;

    // const int sy_n = (ty * BLOCK_SIZE + tx) * 4 / BN;
    // const int sx_n = (ty * BLOCK_SIZE + tx) * 4 % BN;

    // float global_to_reg_A[4];
    // float global_to_reg_B[4];

    // float reg_A[TM];
    // float reg_B[TN];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::precision::tf32, wmma::row_major> a_frag;

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::precision::tf32, wmma::row_major> b_frag;
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    int choice = 0;
    
    {
        A_shared[0][warp_id * 16 + ty_A][tx_A] = A_start[OFFSET(warp_id * 16 + ty_A, tx_A, K)];
        A_shared[0][warp_id * 16 + ty_A][tx_A + 4] = A_start[OFFSET(warp_id * 16 + ty_A, tx_A + 4, K)];
        A_shared[0][warp_id * 16 + ty_A + 8][tx_A] = A_start[OFFSET(warp_id * 16 + ty_A + 8, tx_A, K)];
        A_shared[0][warp_id * 16 + ty_A + 8][tx_A + 4] = A_start[OFFSET(warp_id * 16 + ty_A + 8, tx_A + 4, K)];
        // FLOAT4(A_shared[0][sy_k][sx_k])
        //         = CONST_FLOAT4(A_start[OFFSET(sy_k, sx_k, K)]);

        // FLOAT4(B_shared[0][sy_n][sx_n])
        //         = CONST_FLOAT4(B_start[OFFSET(sy_n, sx_n, N)]);
        B_shared[0][ty_B][warp_id * 16 + tx_B] = B_start[OFFSET(ty_B, warp_id * 16 + tx_B, N)];
        B_shared[0][ty_B][warp_id * 16 + tx_B + 8] = B_start[OFFSET(ty_B, warp_id * 16 + tx_B + 8, N)];
        B_shared[0][ty_B + 4][warp_id * 16 + tx_B] = B_start[OFFSET(ty_B + 4, warp_id * 16 + tx_B, N)];
        B_shared[0][ty_B + 4][warp_id * 16 + tx_B + 8] = B_start[OFFSET(ty_B + 4, warp_id * 16 + tx_B + 8, N)];

        __syncthreads();

        wmma::load_matrix_sync(b_frag, &B_shared[0][0][warp_id * 16], BN + padding_B);
    }
    // printf("1\n");
    #pragma unroll
    for (int s = BK; s < K; s += BK) {

        #pragma unroll
        for (int i = 0; i < 8; i++) {

            wmma::load_matrix_sync(a_frag, &A_shared[choice][16 * i][0], BK + padding_A);

            wmma::mma_sync(c_frag[i], a_frag, b_frag, c_frag[i]);
        }
            
        choice ^= 1;

        A_shared[choice][warp_id * 16 + ty_A][tx_A] = A_start[OFFSET(warp_id * 16 + ty_A, tx_A + s, K)];
        A_shared[choice][warp_id * 16 + ty_A][tx_A + 4] = A_start[OFFSET(warp_id * 16 + ty_A, tx_A + 4 + s, K)];
        A_shared[choice][warp_id * 16 + ty_A + 8][tx_A] = A_start[OFFSET(warp_id * 16 + ty_A + 8, tx_A + s, K)];
        A_shared[choice][warp_id * 16 + ty_A + 8][tx_A + 4] = A_start[OFFSET(warp_id * 16 + ty_A + 8, tx_A + 4 + s, K)];
        // FLOAT4(A_shared[choice][sy_k][sx_k])
        //         = CONST_FLOAT4(A_start[OFFSET(sy_k, sx_k + s, K)]);    // xymc

        // FLOAT4(B_shared[choice][sy_n][sx_n])
        //         = CONST_FLOAT4(B_start[OFFSET(sy_n + s, sx_n, N)]);
        B_shared[choice][ty_B][warp_id * 16 + tx_B] = B_start[OFFSET(ty_B + s, warp_id * 16 + tx_B, N)];
        B_shared[choice][ty_B][warp_id * 16 + tx_B + 8] = B_start[OFFSET(ty_B + s, warp_id * 16 + tx_B + 8, N)];
        B_shared[choice][ty_B + 4][warp_id * 16 + tx_B] = B_start[OFFSET(ty_B + 4 + s, warp_id * 16 + tx_B, N)];
        B_shared[choice][ty_B + 4][warp_id * 16 + tx_B + 8] = B_start[OFFSET(ty_B + 4 + s, warp_id * 16 + tx_B + 8, N)];

        __syncthreads();

        wmma::load_matrix_sync(b_frag, &B_shared[choice][0][warp_id * 16], BN + padding_B);
    }

    {

        // wmma::load_matrix_sync(b_frag, &B_shared[choice][0][warp_id * 16], BN + 4);

        #pragma unroll
        for (int i = 0; i < 8; i++) {

            wmma::load_matrix_sync(a_frag, &A_shared[choice][16 * i][0], BK + padding_A);

            wmma::mma_sync(c_frag[i], a_frag, b_frag, c_frag[i]);
        }
            
    }

    float *C_start = C + blockIdx.y * BM * N + blockIdx.x * BN;

    #pragma unroll
    for (int i = 0; i < 8; i++) {

        wmma::store_matrix_sync(&C_start[OFFSET(16 * i, 16 * warp_id, N)], c_frag[i], N, wmma::mem_row_major);

    }
    
    
}

void sgemm_v9_tensor_core_bank_conflict(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);

    dim3 gridDim(
        (dims.N + BLOCK_SIZE - 1) / BLOCK_SIZE / 8,
        (dims.M + BLOCK_SIZE - 1) / BLOCK_SIZE / 8
    );
    
    sgemm_tensor_core_bank_conflict_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}