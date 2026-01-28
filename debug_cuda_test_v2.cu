#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void add_vectors_kernel_debug(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Debug: have first thread print something
    if (idx == 0) {
        printf("Kernel is running! Thread 0 active.\n");
        printf("First few values from device: a[0]=%f, b[0]=%f\n", a[idx], b[idx]);
    }

    if (idx < n) {
        c[idx] = a[idx] + b[idx];

        // Debug: print first few results
        if (idx < 5) {
            printf("Device: idx=%d, a=%f, b=%f, c=%f\n", idx, a[idx], b[idx], c[idx]);
        }
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
    const int n = 32;  // Smaller size for debugging
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // Check CUDA capabilities
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    // Allocate host memory
    h_a = new float[n];
    h_b = new float[n];
    h_c = new float[n];

    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f * i;
        h_c[i] = -999.0f;  // Initialize to impossible value
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
    CUDA_CHECK(cudaMemset(d_c, 0, n * sizeof(float)));

    // Check for any pending CUDA errors before kernel launch
    cudaError_t pre_error = cudaGetLastError();
    if (pre_error != cudaSuccess) {
        std::cerr << "Pre-launch error: " << cudaGetErrorString(pre_error) << std::endl;
    }

    // Launch kernel with stream for better error checking
    std::cout << "=== Launching kernel ===" << std::endl;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    std::cout << "Grid size: " << grid_size << ", Block size: " << block_size << std::endl;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    add_vectors_kernel_debug<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, n);

    // Check for launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(launch_error) << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

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