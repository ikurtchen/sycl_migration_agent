#include "vectorAdd_kernel.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

/**
 * Host function to launch the CUDA kernel
 *
 * This function properly separates CUDA kernel launch code from C++ test code.
 * The CUDA kernel (__global__ function) can only be called from .cu files,
 * while this host function can be called from both .cu and .cpp files.
 */
cudaError_t launchVectorAdd(const float *A, const float *B, float *C, int numElements)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Validate input parameters
    if (A == nullptr || B == nullptr || C == nullptr || numElements <= 0) {
        fprintf(stderr, "Invalid parameters passed to launchVectorAdd!\n");
        return cudaErrorInvalidValue;
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching vectorAdd kernel: %d blocks, %d threads per block, %d elements\n",
           blocksPerGrid, threadsPerBlock, numElements);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        return err;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %s after launching vectorAdd!\n", cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}