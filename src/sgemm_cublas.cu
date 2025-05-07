#include "../include/sgemm_common.h"

void sgemm_cublas(float* C, const float* A, const float* B, const MatrixDims& dims) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
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
    
    cublasDestroy(handle);
} 