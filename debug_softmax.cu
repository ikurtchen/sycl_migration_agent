#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Simple single-block softmax kernel
__global__ void simple_softmax_1d(float* input, int size) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;

    // Phase 1: Find maximum value
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        max_val = max(max_val, input[i]);
    }
    shared_data[tid] = max_val;
    __syncthreads();

    // Reduce within block to get global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared_data[0];

    // Phase 2: Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        sum_exp += expf(input[i] - max_val);
    }
    shared_data[tid] = sum_exp;
    __syncthreads();

    // Reduce within block to get global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_data[0];

    // Phase 3: Apply softmax
    for (int i = tid; i < size; i += blockDim.x) {
        float exp_val = expf(input[i] - max_val);
        input[i] = exp_val / sum_exp;
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

int main() {
    const int size = 10;
    float *d_input, *h_input;

    // Allocate host memory and initialize
    h_input = new float[size];
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i;
    }

    std::cout << "Input: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    // Allocate device memory and copy
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int block_size = 256;
    int grid_size = 1;
    std::cout << "Launching kernel with grid_size=" << grid_size << ", block_size=" << block_size << std::endl;

    simple_softmax_1d<<<grid_size, block_size>>>(d_input, size);

    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(launch_error) << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_input, d_input, size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Output: ";
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        std::cout << h_input[i] << " ";
        sum += h_input[i];
    }
    std::cout << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    // Cleanup
    delete[] h_input;
    CUDA_CHECK(cudaFree(d_input));

    return 0;
}