#include "../include/sgemm_common.h"
#include "../include/PTX.h"
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

#define BLOCK_SIZE 16

#define padding_A 0
#define padding_B 0

constexpr int Warp_M = 4;
constexpr int Warp_N = 4;

static __device__ 
uint32_t swizzle_A(uint32_t y, uint32_t x) {
    x >>= 2;
    return (((y & 7) >> 2) ^ x) << 2;
}

static __device__ 
uint32_t swizzle_B(uint32_t y, uint32_t x) {
    x >>= 3;
    return ((y & 3) ^ x) << 3;
}

template<const int Stage_num>
__global__ void sgemm_tensor_core_mma_kStage_kernel(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K) {

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8; 

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int warp_id = (ty * BLOCK_SIZE + tx) / 32;
    const int lane_id = (ty * BLOCK_SIZE + tx) % 32; 

    const int warp_m = warp_id / 4; // 0,1
    const int warp_n = warp_id % 4; // 0,1,2,3

    const int NUM_K_TILES = (K + BK - 1) / BK;

    __shared__ float A_shared[Stage_num][BM][BK + padding_A];
    __shared__ float B_shared[Stage_num][BK][BN + padding_B];
    
    uint32_t RA[Warp_M][4];

    uint32_t RB[Warp_N][2];

    float RC[Warp_M][Warp_N][4];

    #pragma unroll
    for (int i = 0; i < Warp_M; i++) {
        for (int j = 0; j < Warp_N; j++) {
            for (int k = 0; k < 4; k++) {
                RC[i][j][k] = 0.0f;
            }
        }
    }
    
    const float *A_start = A + by * BM * K;
    const float *B_start = B + bx * BN;
    
    const int sy_k = (ty * BLOCK_SIZE + tx) * 4 / BK;
    const int sx_k = (ty * BLOCK_SIZE + tx) * 4 % BK;

    const int sy_n = (ty * BLOCK_SIZE + tx) * 4 / BN;
    const int sx_n = (ty * BLOCK_SIZE + tx) * 4 % BN;
    
    #pragma unroll
    for (int k = 0; k < Stage_num - 1; k++) {

        uint32_t smem_ptr_A = __cvta_generic_to_shared(&A_shared[k][sy_k][swizzle_A(sy_k, sx_k)]);

        uint32_t smem_ptr_B = __cvta_generic_to_shared(&B_shared[k][sy_n][swizzle_B(sy_n, sx_n) + (sx_n % 8)]);

        CP_ASYNC_CG(smem_ptr_A, &A_start[OFFSET(sy_k, sx_k + k * BK, K)], 16);

        CP_ASYNC_CG(smem_ptr_B, &B_start[OFFSET(sy_n + k * BK, sx_n, N)], 16);

        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(Stage_num - 2);
    __syncthreads();

    // int reg_store_idx = 0;
    // int reg_load_idx = 1;
    int smem_reg = 0;

    // {
    //     #pragma unroll
    //     for (int i = 0; i < Warp_M; i++) {
            
    //         int smem_regA_addr_y = lane_id % 16;
    //         int smem_regA_addr_x = lane_id / 16 * 4;

    //         uint32_t smem_reg_A = 
    //             __cvta_generic_to_shared(&A_shared[smem_reg][16 * (i + warp_m * Warp_M) + smem_regA_addr_y]
    //                                                 [swizzle_A(smem_regA_addr_y, smem_regA_addr_x)]);
            
    //         LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
    //                     RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], smem_reg_A);

    //     }

    //     #pragma unroll
    //     for (int i = 0; i < Warp_N; i++) {

    //         int smem_regB_addr_y = lane_id % 4;
    //         int smem_regB_addr_x = 8 * (i + warp_n * Warp_N);

    //         RB[reg_store_idx][i][0] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y]
    //                                 [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];

    //         RB[reg_store_idx][i][1] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y + 4]
    //                                 [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];
    //     }
    // }

    #pragma unroll
    for (int k = Stage_num - 1; k < NUM_K_TILES; k++) {
        
        #pragma unroll
        for (int i = 0; i < Warp_M; i++) {
            
            int smem_regA_addr_y = lane_id % 16;
            int smem_regA_addr_x = lane_id / 16 * 4;

            uint32_t smem_reg_A = 
                __cvta_generic_to_shared(&A_shared[smem_reg][16 * (i + warp_m * Warp_M) + smem_regA_addr_y]
                                                    [swizzle_A(smem_regA_addr_y, smem_regA_addr_x)]);
            
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smem_reg_A);

        }

        #pragma unroll
        for (int i = 0; i < Warp_N; i++) {

            int smem_regB_addr_y = lane_id % 4;
            int smem_regB_addr_x = 8 * (i + warp_n * Warp_N);

            RB[i][0] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y]
                                    [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];

            RB[i][1] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y + 4]
                                    [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];
        }

        #pragma unroll
        for (int i = 0; i < Warp_M; i++) {
            
            #pragma unroll
            for (int j = 0; j < Warp_N; j++) {

                SMMA1688(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], 
                    RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                    RB[j][0], RB[j][1], 
                    RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);

            }

        }

        int global_smem = k % Stage_num;

        uint32_t smem_ptr_A = 
                __cvta_generic_to_shared(&A_shared[global_smem][sy_k][swizzle_A(sy_k, sx_k)]);

        uint32_t smem_ptr_B = 
                __cvta_generic_to_shared(&B_shared[global_smem][sy_n][swizzle_B(sy_n, sx_n) + (sx_n % 8)]);

        CP_ASYNC_CG(smem_ptr_A, &A_start[OFFSET(sy_k, sx_k + k * BK, K)], 16);

        CP_ASYNC_CG(smem_ptr_B, &B_start[OFFSET(sy_n + k * BK, sx_n, N)], 16);

        CP_ASYNC_COMMIT_GROUP();

        // #pragma unroll
        // for (int i = 0; i < Warp_M; i++) {
            
        //     int smem_regA_addr_y = lane_id % 16;
        //     int smem_regA_addr_x = lane_id / 16 * 4;

        //     uint32_t smem_reg_A = 
        //         __cvta_generic_to_shared(&A_shared[smem_reg][16 * (i + warp_m * Warp_M) + smem_regA_addr_y]
        //                                             [swizzle_A(smem_regA_addr_y, smem_regA_addr_x)]);
            
        //     LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smem_reg_A);

        // }

        // #pragma unroll
        // for (int i = 0; i < Warp_N; i++) {

        //     int smem_regB_addr_y = lane_id % 4;
        //     int smem_regB_addr_x = 8 * (i + warp_n * Warp_N);

        //     RB[i][0] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y]
        //                             [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];

        //     RB[i][1] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y + 4]
        //                             [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];
        // }

        // #pragma unroll
        // for (int i = 0; i < Warp_M; i++) {
            
        //     #pragma unroll
        //     for (int j = 0; j < Warp_N; j++) {

        //         SMMA1688(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], 
        //             RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
        //             RB[j][0], RB[j][1], 
        //             RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);

        //     }

        // }

        smem_reg = (smem_reg + 1) % Stage_num;

        CP_ASYNC_WAIT_GROUP(Stage_num - 2);
        __syncthreads();
    }

    if constexpr ((Stage_num - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k < Stage_num - 1; k++) {

        // reg_store_idx ^= 1;
        // reg_load_idx ^= 1;
        

        #pragma unroll
        for (int i = 0; i < Warp_M; i++) {
            
            int smem_regA_addr_y = lane_id % 16;
            int smem_regA_addr_x = lane_id / 16 * 4;

            uint32_t smem_reg_A = 
                __cvta_generic_to_shared(&A_shared[smem_reg][16 * (i + warp_m * Warp_M) + smem_regA_addr_y]
                                                    [swizzle_A(smem_regA_addr_y, smem_regA_addr_x)]);
            
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smem_reg_A);

            #pragma unroll
            for (int j = 0; j < Warp_N; j++) {

                int smem_regB_addr_y = lane_id % 4;
                int smem_regB_addr_x = 8 * (i + warp_n * Warp_N);

                RB[i][0] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y]
                                        [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];

                RB[i][1] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y + 4]
                                        [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];
                                    
            }

        }

        #pragma unroll
        for (int i = 0; i < Warp_N; i++) {

            int smem_regB_addr_y = lane_id % 4;
            int smem_regB_addr_x = 8 * (i + warp_n * Warp_N);

            RB[i][0] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y]
                                    [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];

            RB[i][1] = (uint32_t&)B_shared[smem_reg][smem_regB_addr_y + 4]
                                    [swizzle_B(smem_regB_addr_y, smem_regB_addr_x) + lane_id / 4];
                                    
        }

        #pragma unroll
        for (int i = 0; i < Warp_M; i++) {
            
            #pragma unroll
            for (int j = 0; j < Warp_N; j++) {

                SMMA1688(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], 
                    RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                    RB[j][0], RB[j][1], 
                    RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);

            }
        }

        smem_reg = (smem_reg + 1) % Stage_num;
    }

    float *C_start = C + by * BM * N + bx * BN + warp_m * Warp_M * 16 * N + warp_n * Warp_N * 8;

    #pragma unroll
    for (int i = 0; i < Warp_M; i++) {

        #pragma unroll
        for (int j = 0; j < Warp_N; j++) {
            
            LDST64BITS(C_start[OFFSET(16 * i + lane_id / 4, 8 * j + (lane_id % 4) * 2, N)]) 
                        = LDST64BITS(RC[i][j][0]);

            LDST64BITS(C_start[OFFSET(16 * i + lane_id / 4 + 8, 8 * j + (lane_id % 4) * 2, N)]) 
                        = LDST64BITS(RC[i][j][2]);

        }

    }
    
    
}

void sgemm_v14_tensor_core_mma_kStage(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(
        (dims.N + blockDim.x - 1) / blockDim.x / 8,
        (dims.M + blockDim.y - 1) / blockDim.y / 8
    );
    
    sgemm_tensor_core_mma_kStage_kernel<4><<<gridDim, blockDim>>>(C, A, B, dims.M, dims.N, dims.K);
    
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}