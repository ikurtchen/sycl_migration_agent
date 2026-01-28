#pragma once

#include <gtest/gtest.h>
#include "test_utils.h"
#include <cuda_runtime.h>
#include <string>
#include <memory>

template<typename T>
class BaseCudaTest : public ::testing::Test {
protected:
    virtual void SetUp() override {
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        ASSERT_GT(device_count, 0) << "No CUDA devices available";

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceReset());
    }

    virtual void TearDown() override {
        CUDA_CHECK(cudaDeviceReset());
    }

    void allocateMemory(T** d_ptr, T** h_ptr, size_t count) {
        size_t size = count * sizeof(T);

        // Host memory
        *h_ptr = new T[count];

        // Device memory
        CUDA_CHECK(cudaMalloc(d_ptr, size));
    }

    // Overload for allocating device memory with existing host pointer
    void allocateMemory(T* d_ptr, T* h_ptr, size_t count) {
        size_t size = count * sizeof(T);

        // Allocate device memory only (host memory already allocated)
        CUDA_CHECK(cudaMalloc(d_ptr, size));
    }

    void freeMemory(T* d_ptr, T* h_ptr) {
        delete[] h_ptr;
        CUDA_CHECK(cudaFree(d_ptr));
    }

    void copyToDevice(T* d_ptr, const T* h_ptr, size_t count) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyFromDevice(T* h_ptr, const T* d_ptr, size_t count) {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void saveInput(const char* filename, const T* data, size_t count) {
        std::string filepath = std::string("cuda_outputs/") + filename;
        saveData(filepath.c_str(), data, count);
    }

    void saveOutput(const char* filename, const T* data, size_t count) {
        std::string filepath = std::string("cuda_outputs/") + filename;
        saveData(filepath.c_str(), data, count);
    }

   BenchmarkResult benchmarkKernel(std::function<void()> kernel_func,
                                 size_t elements_per_iteration,
                                 const std::string& kernel_name,
                                 int warmup_iterations = 10,
                                 int benchmark_iterations = 100) {
        BenchmarkResult result;
        result.kernel_name = kernel_name;
        result.num_iterations = benchmark_iterations;
        result.data_size_bytes = elements_per_iteration * sizeof(T);

        // Warmup
        for (int i = 0; i < warmup_iterations; ++i) {
            kernel_func();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        CudaTimer timer;
        timer.start();

        for (int i = 0; i < benchmark_iterations; ++i) {
            kernel_func();
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();

        result.avg_time_ms = timer.elapsed_ms() / benchmark_iterations;

        // Calculate GFLOPS (assuming 2 FLOPs per element for simple operations)
        double flops_per_iteration = 2.0 * elements_per_iteration;
        double seconds_per_iteration = result.avg_time_ms / 1000.0;
        result.gflops = flops_per_iteration / (seconds_per_iteration * 1e9);

        return result;
    }

private:
    int device_count;
};

// Specialization for FP16 tests
class FP16Test : public BaseCudaTest<half> {
protected:
    void SetUp() override {
        BaseCudaTest<half>::SetUp();

        // Check FP16 support
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        ASSERT_GE(prop.major, 6) << "FP16 requirescompute capability >= 6.0";
    }
};