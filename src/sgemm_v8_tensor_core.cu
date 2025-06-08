#include "../include/sgemm_common.h"
#include <mma.h>
using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CONST_FLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])

#define BLOCK_SIZE 16
#define TILE_SIZE 4

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

#define padding_A 4
#define padding_B 8

// constexpr int Warp_M = 2;
// constexpr int Warp_N = 4;

// constexpr int STEP = BLOCK_SIZE * TILE_SIZE;

__global__ void sgemm_tensor_core_kernel(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K) {

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;

    // constexpr int TM = BM / BLOCK_SIZE;
    // constexpr int TN = BN / BLOCK_SIZE;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int warp_id = (ty * BLOCK_SIZE + tx) / 32;
    // const int lane_id = (tx * BLOCK_SIZE + ty) % 32;

    __shared__ float A_shared[2][BM][BK + padding_A];
    __shared__ float B_shared[2][BK][BN + padding_B];
    
    // float sum[TM][TN] = {0.0f};
    
    const float *A_start = A + by * BM * K;
    const float *B_start = B + bx * BN;
    
    const int sy_k = (ty * BLOCK_SIZE + tx) * 4 / BK;
    const int sx_k = (ty * BLOCK_SIZE + tx) * 4 % BK;

    const int sy_n = (ty * BLOCK_SIZE + tx) * 4 / BN;
    const int sx_n = (ty * BLOCK_SIZE + tx) * 4 % BN;

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

    {
        FLOAT4(A_shared[0][sy_k][sx_k])
                = CONST_FLOAT4(A_start[OFFSET(sy_k, sx_k, K)]);

        FLOAT4(B_shared[0][sy_n][sx_n])
                = CONST_FLOAT4(B_start[OFFSET(sy_n, sx_n, N)]);
        
        __syncthreads();
    }

    int choice = 0;

    #pragma unroll
    for (int s = BK; s < K; s += BK) {

        wmma::load_matrix_sync(b_frag, &B_shared[choice][0][warp_id * 16], BN + padding_B);

        #pragma unroll
        for (int i = 0; i < 8; i++) {

            wmma::load_matrix_sync(a_frag, &A_shared[choice][16 * i][0], BK + padding_A);

            wmma::mma_sync(c_frag[i], a_frag, b_frag, c_frag[i]);
        }
            
        choice ^= 1;

        FLOAT4(A_shared[choice][sy_k][sx_k])
                = CONST_FLOAT4(A_start[OFFSET(sy_k, sx_k + s, K)]);

        FLOAT4(B_shared[choice][sy_n][sx_n])
                = CONST_FLOAT4(B_start[OFFSET(sy_n + s, sx_n, N)]);
                
        __syncthreads();
    }

    {

        wmma::load_matrix_sync(b_frag, &B_shared[choice][0][warp_id * 16], BN + padding_B);

        #pragma unroll
        for (int i = 0; i < 8; i++) {

            wmma::load_matrix_sync(a_frag, &A_shared[choice][16 * i][0], BK + padding_A);

            wmma::mma_sync(c_frag[i], a_frag, b_frag, c_frag[i]);
        }
            
    }

    float *C_start = C + by * BM * N + bx * BN;

    #pragma unroll
    for (int i = 0; i < 8; i++) {

        wmma::store_matrix_sync(&C_start[OFFSET(16 * i, 16 * warp_id, N)], c_frag[i], N, wmma::mem_row_major);

    }
    
    
}

void sgemm_v8_tensor_core(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x / 8,
        (dims.M + blockDim.y - 1) / blockDim.y / 8
    );
    
    sgemm_tensor_core_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}