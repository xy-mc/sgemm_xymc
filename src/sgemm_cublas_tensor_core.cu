#include "../include/sgemm_common.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// 全局 handle，只创建一次
static cublasHandle_t handle_tc = nullptr;

void sgemm_cublas_tensorcore(float* C, const float* A, const float* B, const MatrixDims& dims) {
    if (handle_tc == nullptr) {
        cublasCreate(&handle_tc);
        // 启用TF32
        cublasSetMathMode(handle_tc, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    int m = dims.M, n = dims.N, k = dims.K;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS使用列主序，所以需要转置矩阵
    cublasSgemm(
        handle_tc,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, n,
        A, k,
        &beta,
        C, n
    );
}

// 清理函数，在程序结束时调用
void cleanup_cublas_tensorcore() {
    if (handle_tc != nullptr) {
        cublasDestroy(handle_tc);
        handle_tc = nullptr;
    }
}