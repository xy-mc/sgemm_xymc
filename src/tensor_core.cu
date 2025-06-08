#include "../include/sgemm_common.h"
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
// Support A and B matrix with row-major inorder to compare with the kernels
// using CUDA Cores in sgemm.cu and sgemm_async.cu. also need flag when
// compiling.

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

__global__ void f32x4_tf32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = wmma::__float_to_tf32(reg_x.x);
    reg_y.y = wmma::__float_to_tf32(reg_x.y);
    reg_y.z = wmma::__float_to_tf32(reg_x.z);
    reg_y.w = wmma::__float_to_tf32(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}
// stage2/3/4 (stage2=double buffers+copy async)
// 1. When using shared memory exceeds 48 KB, dynamic shared memory needs to be
// used, i.e., declare a block of dynamic shared memory with extern shared half
// smem[];. When calling the kernel, the size of the dynamic shared memory needs
// to be specified, and smem addressing should be used in a one-dimensional
// array manner.
// 2. Improve L2 Cache locality (Thread Block Swizzle):
// https://zhuanlan.zhihu.com/p/555339335
// 3. __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void
sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel(const float *A, const float *B, float *C,
                                                 int M, int N, int K) {
  // 256 threads(8 warps) per block.
  // const int bx = blockIdx.x;
  // BLOCK_SWIZZLE 0/1 控制是否使用 block swizzle
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K;                             // 8
  __shared__ float s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];

  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int warp_m = warp_id / 2;      // 0,1,2,3
  const int warp_n = warp_id % 2;      // 0,1

  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2;                // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // col 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32;       // row 0~7
  int load_smem_b_n = (tid % 32) * 4; // col 0,4,...,124,...
  // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数
  // 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {         // 0, 1
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        __cvta_generic_to_shared(&s_a[k][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        __cvta_generic_to_shared(&s_b[k][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    // load stage 2, k start from 2
    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];

// compute stage 0
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0],
                             BK + A_PAD);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n],
                             BN + B_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  // processing last (K_STAGE-1) k iters.
  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0],
                               BK + A_PAD);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n],
                               BN + B_PAD);
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

// finally, store back to C matrix.
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}

#define BLOCK_SIZE 16

void sgemm_tensor_example(float* C, const float* A, const float* B, const MatrixDims& dims) {
    
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
    
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<16, 16, 8, 4, 2, 2, 4, 0, 0, 2, 0><<<gridDim, blockDim>>>(A, B, C, dims.M, dims.N, dims.K);
        
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}