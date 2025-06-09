#include "../include/sgemm_common.h"
#include <mma.h>
using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CONST_FLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])

#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))

#define CP_ASYNC_CG(smem_ptr, gmem_ptr, Bytes) \
    asm volatile ( \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" \
        :: "r"(smem_ptr), "l"(gmem_ptr), "n"(Bytes) \
    );

#define CP_ASYNC_ALL(smem_ptr, gmem_ptr, Bytes) \
    asm volatile ( \
        "cp.async.ca.shared.global [%0], [%1], %2;\n" \
        :: "r"(smem_ptr), "l"(gmem_ptr), "n"(Bytes) \
    );
// 提交当前 group
#define CP_ASYNC_COMMIT_GROUP() \
    asm volatile ("cp.async.commit_group;\n" ::);

#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define BLOCK_SIZE 16
#define TILE_SIZE 4

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

#define padding_A 0
#define padding_B 0

constexpr int Warp_M = 2;
constexpr int Warp_N = 4;

// constexpr int STEP = BLOCK_SIZE * TILE_SIZE;

__global__ void sgemm_tensor_core_cp_async_kernel(
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

    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
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
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[Warp_M][Warp_N];

    #pragma unroll
    for (int i = 0; i < Warp_M; i++) {
        for (int j = 0; j < Warp_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    
    {
        uint32_t smem_ptr_A = __cvta_generic_to_shared(&A_shared[0][sy_k][sx_k]);

        uint32_t smem_ptr_B = __cvta_generic_to_shared(&B_shared[0][sy_n][sx_n]);

        CP_ASYNC_CG(smem_ptr_A, &A_start[OFFSET(sy_k, sx_k, K)], 16);

        CP_ASYNC_CG(smem_ptr_B, &B_start[OFFSET(sy_n, sx_n, N)], 16);

        CP_ASYNC_COMMIT_GROUP();

        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    int choice = 1;

    #pragma unroll
    for (int s = BK; s < K; s += BK) {

        uint32_t smem_ptr_A = __cvta_generic_to_shared(&A_shared[choice][sy_k][sx_k]);

        uint32_t smem_ptr_B = __cvta_generic_to_shared(&B_shared[choice][sy_n][sx_n]);

        CP_ASYNC_CG(smem_ptr_A, &A_start[OFFSET(sy_k, sx_k + s, K)], 16);

        CP_ASYNC_CG(smem_ptr_B, &B_start[OFFSET(sy_n + s, sx_n, N)], 16);

        CP_ASYNC_COMMIT_GROUP();

        choice ^= 1;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, 
        nvcuda::wmma::precision::tf32, wmma::row_major> a_frag[Warp_M];

        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, 
        nvcuda::wmma::precision::tf32, wmma::row_major> b_frag[Warp_N];

        #pragma unroll
        for (int i = 0; i < Warp_M; i++) {

            wmma::load_matrix_sync(a_frag[i], &A_shared[choice][16 * (i + warp_m * Warp_M)][0], BK + padding_A);

        }

        #pragma unroll
        for (int i = 0; i < Warp_N; i++) {

            wmma::load_matrix_sync(b_frag[i], &B_shared[choice][0][16 * (i + warp_n * Warp_N)], BN + padding_B);

        }

        #pragma unroll
        for (int i = 0; i < Warp_M; i++) {
            
            #pragma unroll
            for (int j = 0; j < Warp_N; j++) {

                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);

            }

        }

        CP_ASYNC_WAIT_GROUP(0);
                
        __syncthreads();
    }

    choice ^= 1;

    {

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, 
        nvcuda::wmma::precision::tf32, wmma::row_major> a_frag[Warp_M];

        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, 
        nvcuda::wmma::precision::tf32, wmma::row_major> b_frag[Warp_N];

        #pragma unroll
        for (int i = 0; i < Warp_N; i++) {

            wmma::load_matrix_sync(b_frag[i], &B_shared[choice][0][16 * (i + warp_n * Warp_N)], BN + padding_B);

        }

        #pragma unroll
        for (int i = 0; i < Warp_M; i++) {

            wmma::load_matrix_sync(a_frag[i], &A_shared[choice][16 * (i + warp_m * Warp_M)][0], BK + padding_A);
      
        }

        #pragma unroll
        for (int i = 0; i < Warp_M; i++) {

            #pragma unroll
            for (int j = 0; j < Warp_N; j++) {

                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);

            }

        }
            
    }

    float *C_start = C + by * BM * N + bx * BN + warp_m * Warp_M * 16 * N + warp_n * Warp_N * 16;

    #pragma unroll
    for (int i = 0; i < Warp_M; i++) {

        #pragma unroll
        for (int j = 0; j < Warp_N; j++) {

            wmma::store_matrix_sync(&C_start[OFFSET(16 * i, 16 * j, N)], c_frag[i][j], N, wmma::mem_row_major);

        }

    }
    
    
}

void sgemm_v11_tensor_core_cp_async(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x / 8,
        (dims.M + blockDim.y - 1) / blockDim.y / 8
    );
    
    sgemm_tensor_core_cp_async_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}