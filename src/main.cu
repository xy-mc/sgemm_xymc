#include "../include/sgemm_common.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

#define num_test 1
// 声明不同版本的SGEMM实现
void sgemm_v0_global_memory(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v1_shared_memory(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v2_tiling(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v3_vectorized(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v4_register(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v5_transpose(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v6_double_buffer(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v7_bank_conflict(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v8_tensor_core(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v9_tensor_core_bank_conflict(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v10_tensor_core_vectorized(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_v11_tensor_core_cp_async(float* C, const float* A, const float* B, const MatrixDims& dims);

void sgemm_tensor_example(float* C, const float* A, const float* B, const MatrixDims& dims);

void sgemm_cublas(float* C, const float* A, const float* B, const MatrixDims& dims);
void sgemm_cublas_tensorcore(float* C, const float* A, const float* B, const MatrixDims& dims);

// 运行cuBLAS并保存结果
void run_cublas_and_save_result(MatrixData& data) {
    // 创建临时数组来存储cuBLAS结果
    float* cublas_result;
    cudaMalloc(&cublas_result, data.size_C);
    
    // 运行cuBLAS
    sgemm_cublas(cublas_result, data.d_A, data.d_B, data.dims);
    
    // 将结果复制回主机
    cudaMemcpy(data.h_C, cublas_result, data.size_C, cudaMemcpyDeviceToHost);
    
    // 清理临时数组
    cudaFree(cublas_result);
}

// 将矩阵结果写入文件
void write_matrix_to_file(const std::string& filename, const float* matrix, int M, int N) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            outfile << matrix[i * N + j] << " ";
        }
        outfile << "\n";
    }
    outfile.close();
}

void write_matrix_to_file_tile(const std::string& filename, const float* matrix, int M, int N, int tile_M = 16, int tile_N = 16) {
    std::ofstream fout(filename);
    int num_tile_rows = (M + tile_M - 1) / tile_M;
    int num_tile_cols = (N + tile_N - 1) / tile_N;
    for (int tile_i = 0; tile_i < num_tile_rows; ++tile_i) {
        for (int tile_j = 0; tile_j < num_tile_cols; ++tile_j) {
            fout << "Tile (" << tile_i << "," << tile_j << "):\n";
            for (int inner_i = 0; inner_i < tile_M; ++inner_i) {
                int global_i = tile_i * tile_M + inner_i;
                if (global_i >= M) break;
                for (int inner_j = 0; inner_j < tile_N; ++inner_j) {
                    int global_j = tile_j * tile_N + inner_j;
                    if (global_j >= N) break;
                    fout << matrix[global_i * N + global_j] << " ";
                }
                fout << "\n";
            }
            fout << "\n"; // tile之间空一行
        }
    }
    fout.close();
}

int main() {
    // 设置矩阵维度
    std::vector<MatrixDims> test_cases = {
        // {128, 128, 128},
        // {1024, 1024, 1024},
        {2048, 2048, 2048},
        // {4096, 4096, 4096},
        // {8192, 8192, 8192},
        // {16384, 16384, 16384}
    };

    // 测试每个维度
    for (const auto& dims : test_cases) {
        std::cout << "\nTesting matrix dimensions: " 
                  << dims.M << "x" << dims.K << " * " 
                  << dims.K << "x" << dims.N << "\n"
                  << "========================================\n";

        // 创建并初始化矩阵数据
        MatrixData data(dims);
        data.initialize();
        data.copyToDevice();

        // 运行所有版本的SGEMM
        std::vector<PerformanceResult> results;
        
        // results.push_back(runPerformanceTest(sgemm_v0_global_memory, data, num_test, "Global Memory"));
        // results.push_back(runPerformanceTest(sgemm_v1_shared_memory, data, num_test, "Shared Memory"));
        // results.push_back(runPerformanceTest(sgemm_v2_tiling, data, num_test, "Tiling"));
        // results.push_back(runPerformanceTest(sgemm_v3_vectorized, data, num_test, "Vectorized"));
        // results.push_back(runPerformanceTest(sgemm_v4_register, data, num_test, "Register"));
        // results.push_back(runPerformanceTest(sgemm_v5_transpose, data, num_test, "Transpose"));
        // results.push_back(runPerformanceTest(sgemm_v6_double_buffer, data, num_test, "Double Buffer"));
        results.push_back(runPerformanceTest(sgemm_v7_bank_conflict, data, num_test, "Bank Conflict"));
        // results.push_back(runPerformanceTest(sgemm_v8_tensor_core, data, num_test, "Tensor Core"));
        // results.push_back(runPerformanceTest(sgemm_v9_tensor_core_bank_conflict, data, num_test, "Tensor Core Bank Conflict"));
        results.push_back(runPerformanceTest(sgemm_v10_tensor_core_vectorized, data, num_test, "Tensor Core Vectorized"));
        results.push_back(runPerformanceTest(sgemm_v11_tensor_core_cp_async, data, num_test, "Tensor Core cp.async"));
        // results.push_back(runPerformanceTest(sgemm_tensor_example, data, num_test, "Tensor Core Example"));
        results.push_back(runPerformanceTest(sgemm_cublas, data, num_test, "cuBLAS"));
        results.push_back(runPerformanceTest(sgemm_cublas_tensorcore, data, num_test, "cuBLAS Tensor Core"));
        // 打印所有结果
        for (const auto& result : results) {
            printPerformanceResult(result);
        }

        // 进行误差测试
        std::cout << "\nError Analysis (compared with cuBLAS):\n"
                  << "========================================\n";
        
        std::vector<ErrorResult> error_results;
        // error_results.push_back(runErrorTest(sgemm_v0_global_memory, data, "Global Memory"));
        // error_results.push_back(runErrorTest(sgemm_v1_shared_memory, data, "Shared Memory"));
        // error_results.push_back(runErrorTest(sgemm_v2_tiling, data, "Tiling"));
        // error_results.push_back(runErrorTest(sgemm_v3_vectorized, data, "Vectorized"));
        // error_results.push_back(runErrorTest(sgemm_v4_register, data, "Register"));
        // error_results.push_back(runErrorTest(sgemm_v5_transpose, data, "Transpose"));
        // error_results.push_back(runErrorTest(sgemm_v6_double_buffer, data, "Double Buffer"));
        // error_results.push_back(runErrorTest(sgemm_v7_bank_conflict, data, "Bank Conflict"));
        // error_results.push_back(runErrorTest(sgemm_v8_tensor_core, data, "Tensor Core"));
        // error_results.push_back(runErrorTest(sgemm_v9_tensor_core_bank_conflict, data, "Tensor Core Bank Conflict"));
        // error_results.push_back(runErrorTest(sgemm_v10_tensor_core_vectorized, data, "Tensor Core Vectorized"));
        // error_results.push_back(runErrorTest(sgemm_v11_tensor_core_cp_async, data, "Tensor Core cp.async"));
        for (const auto& result : error_results) {
            printErrorResult(result);
        }

        // 输出最后一个计算结果到文件
        // data.copyToHost();
        // write_matrix_to_file_tile("last_result_" + std::to_string(dims.M) + "x" + std::to_string(dims.N) + ".txt", data.h_C, dims.M, dims.N, 16, 16);
        // // std::string last_result_file = "last_result_" + std::to_string(dims.M) + "x" + std::to_string(dims.N) + ".txt";
        // // write_matrix_to_file(last_result_file, data.h_C, dims.M, dims.N);
        // // std::cout << "\nLast computation results written to: " << last_result_file << std::endl;

        // // // 运行cuBLAS并输出结果到文件
        // run_cublas_and_save_result(data);
        // // std::string cublas_result_file = "cublas_result_" + std::to_string(dims.M) + "x" + std::to_string(dims.N) + ".txt";
        // // write_matrix_to_file(cublas_result_file, data.h_C, dims.M, dims.N);
        // // std::cout << "cuBLAS results written to: " << cublas_result_file << std::endl;
        // write_matrix_to_file_tile("cublas_result_" + std::to_string(dims.M) + "x" + std::to_string(dims.N) + ".txt", data.h_C, dims.M, dims.N, 16, 16);
    }

    return 0;
} 