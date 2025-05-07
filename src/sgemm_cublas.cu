#include "../include/sgemm_common.h"

// 全局 handle，只创建一次
static cublasHandle_t handle = nullptr;

void sgemm_cublas(float* C, const float* A, const float* B, const MatrixDims& dims) {
    // 如果 handle 不存在，创建它
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS使用列主序，所以需要转置矩阵
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dims.N, dims.M, dims.K,
                &alpha,
                B, dims.N,
                A, dims.K,
                &beta,
                C, dims.N);
}

// 清理函数，在程序结束时调用
void cleanup_cublas() {
    if (handle != nullptr) {
        cublasDestroy(handle);
        handle = nullptr;
    }
} 