#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void add_vectors_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        return; \
    } \
} while(0)

void printSomeValues(const float* data, int n, const char* name) {
    std::cout << name << " first 5 values: ";
    for (int i = 0; i < std::min(5, n); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    const int n = 1024;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // Allocate host memory
    h_a = new float[n];
    h_b = new float[n];
    h_c = new float[n];

    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f * i;
        h_c[i] = 0.0f;  // Initialize to zero to see if kernel writes
    }

    std::cout << "=== Initial state ===" << std::endl;
    printSomeValues(h_a, n, "h_a");
    printSomeValues(h_b, n, "h_b");
    printSomeValues(h_c, n, "h_c");

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    // Copy to device
    std::cout << "\n=== Copying to device ===" << std::endl;
    CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, n * sizeof(float)));  // Clear output buffer

    // Launch kernel
    std::cout << "=== Launching kernel ===" << std::endl;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    std::cout << "Grid size: " << grid_size << ", Block size: " << block_size << std::endl;

    add_vectors_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::cout << "=== Copying back results ===" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    std::cout << "\n=== Results ===" << std::endl;
    printSomeValues(h_c, n, "h_c");

    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-5f) {
            if (errors < 5) {
                std::cout << "Error at index " << i << ": expected " << expected
                          << ", got " << h_c[i] << std::endl;
            }
            errors++;
        }
    }

    std::cout << "\nTotal errors: " << errors << " out of " << n << " elements" << std::endl;

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}