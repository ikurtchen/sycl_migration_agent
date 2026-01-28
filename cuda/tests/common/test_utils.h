#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>

template<typename T>
void saveData(const char* filename, const T* data, size_t count) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    file.close();
}

template<typename T>
void loadData(const char* filename, T* data, size_t count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(data), count * sizeof(T));
    file.close();
}

template<typename T>
void initializeRandom(T* data, size_t count, T min_val = static_cast<T>(-1.0),
                     T max_val = static_cast<T>(1.0)) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(static_cast<float>(min_val), static_cast<float>(max_val));

    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, half>) {
            data[i] = __float2half(dis(gen));
        } else {
            data[i] = static_cast<T>(dis(gen));
        }
    }
}

template<typename T>
void initializeConstant(T* data, size_t count, T value) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = value;
    }
}

struct BenchmarkResult {
    std::string kernel_name;
    double avg_time_ms;
    double gflops;
    size_t data_size_bytes;
    int num_iterations;

    void saveToJson(const char* filename) const {
        std::ofstream file(filename);
        file << "{\n";
        file << "  \"kernel_name\": \"" << kernel_name << "\",\n";
        file << "  \"avg_time_ms\": " << avg_time_ms << ",\n";
        file << "  \"gflops\": " << gflops << ",\n";
        file << "  \"data_size_bytes\": " << data_size_bytes << ",\n";
        file << "  \"num_iterations\": " << num_iterations << "\n";
        file << "}\n";
        file.close();
    }
};

class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;

public:
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event, 0);
    }

    void stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
    }

    float elapsed_ms() {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }
};

template<typename T>
bool compareArrays(const T* a, const T* b, size_t count, T tolerance = static_cast<T>(1e-5)) {
    for (size_t i = 0; i < count; ++i) {
        T diff = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
        if constexpr (std::is_same_v<T, half>) {
            if (__half2float(diff) > __half2float(tolerance)) {
                std::cerr << "Mismatch at index " << i << ": "
                          << __half2float(a[i]) << " vs " << __half2float(b[i])
                          << " (diff: " << __half2float(diff) << ")" << std::endl;
                return false;
            }
        } else {
            if (diff > tolerance) {
                std::cerr << "Mismatch at index " << i << ": "
                          << a[i] << " vs " << b[i]
                          << " (diff: " << diff << ")" << std::endl;
                return false;
            }
        }
    }
    return true;
}

inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)