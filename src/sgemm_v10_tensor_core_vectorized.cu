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

#define padding_A 0
#define padding_B 0

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

// constexpr int Warp_M = 2;
// constexpr int Warp_N = 4;

// constexpr int STEP = BLOCK_SIZE * TILE_SIZE;

__global__ void sgemm_tensor_core_vectorized_kernel(
    float* C,
    const float* A, 
    const float* B,
    int M, int N, int K) {

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;

    // const int bx = blockIdx.x;
    // const int by = blockIdx.y;

    const int tx = threadIdx.x;
    // const int ty = threadIdx.y;

    const int warp_id = (tx) / 32;
    const int lane_id = (tx) % 32;     

    const int tx_A = lane_id % 4;
    const int ty_A = lane_id / 4;

    // const int tx_B = lane_id & 7;
    // const int ty_B = lane_id >> 3;

    __shared__ float A_shared[2][BM][BK + padding_A];
    __shared__ float B_shared[2][BK][BN + padding_B];
    
    // float sum[TM][TN] = {0.0f};
    
    const float *A_start = A + blockIdx.y * BM * K;
    const float *B_start = B + blockIdx.x * BN;
    
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::precision::tf32, wmma::row_major> a_frag[2][8];

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::precision::tf32, wmma::row_major> b_frag[2];
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    int choice = 0;
    
    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {

            uint32_t smem_ptr_A = __cvta_generic_to_shared(&A_shared[choice][warp_id * 16 + i * 8 + ty_A][tx_A * 4]);

            uint32_t smem_ptr_B = __cvta_generic_to_shared(&B_shared[choice][warp_id * 2 + i][lane_id * 4]);

            CP_ASYNC_CG(smem_ptr_A, &A_start[OFFSET(warp_id * 16 + i * 8 + ty_A, tx_A * 4, K)], 16);

            CP_ASYNC_CG(smem_ptr_B, &B_start[OFFSET(warp_id * 2 + i, lane_id * 4, N)], 16);
        }

        CP_ASYNC_COMMIT_GROUP();

        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }
    
    choice ^= 1;

    #pragma unroll
    for (int s = BK; s < K; s += BK) {

        for (int i = 0; i < 2; i++) {
            
            uint32_t smem_ptr_A = __cvta_generic_to_shared(&A_shared[choice][warp_id * 16 + i * 8 + ty_A][tx_A * 4]);
            
            uint32_t smem_ptr_B = __cvta_generic_to_shared(&B_shared[choice][warp_id * 2 + i][lane_id * 4]);

            CP_ASYNC_CG(smem_ptr_A, &A_start[OFFSET(warp_id * 16 + i * 8 + ty_A, tx_A * 4 + s, K)], 16);

            CP_ASYNC_CG(smem_ptr_B, &B_start[OFFSET(warp_id * 2 + i + s, lane_id * 4, N)], 16);
        }

        CP_ASYNC_COMMIT_GROUP();

        choice ^= 1;

        #pragma unroll
        for (int i = 0; i < 2; i++) {

            #pragma unroll
            for (int j = 0; j < 8; j++) {

                wmma::load_matrix_sync(a_frag[i][j], &A_shared[choice][16 * j][8 * i], BK + padding_A);
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {

            wmma::load_matrix_sync(b_frag[i], &B_shared[choice][8 * i][warp_id * 16], BN + padding_B);
        
        }


        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {

                wmma::mma_sync(c_frag[j], a_frag[i][j], b_frag[i], c_frag[j]);
            }
        }
           
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    choice ^= 1;
    
    {

        #pragma unroll
        for (int i = 0; i < 2; i++) {

            #pragma unroll
            for (int j = 0; j < 8; j++) {

                wmma::load_matrix_sync(a_frag[i][j], &A_shared[choice][16 * j][8 * i], BK + padding_A);
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {

            wmma::load_matrix_sync(b_frag[i], &B_shared[choice][8 * i][warp_id * 16], BN + padding_B);
        
        }


        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {

                wmma::mma_sync(c_frag[j], a_frag[i][j], b_frag[i], c_frag[j]);
            }
        }
            
    }

    float *C_start = C + blockIdx.y * BM * N + blockIdx.x * BN;

    #pragma unroll
    for (int i = 0; i < 8; i++) {

        wmma::store_matrix_sync(&C_start[OFFSET(16 * i, 16 * warp_id, N)], c_frag[i], N, wmma::mem_row_major);

    }
    
    
}

void sgemm_v10_tensor_core_vectorized(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);

    dim3 gridDim(
        (dims.N + BLOCK_SIZE - 1) / BLOCK_SIZE / 8,
        (dims.M + BLOCK_SIZE - 1) / BLOCK_SIZE / 8
    );
    
    // cudaFuncSetAttribute(
    //     sgemm_tensor_core_vectorized_kernel,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize,
    //     98304 // 64KB
    // );
    
    sgemm_tensor_core_vectorized_kernel<<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}