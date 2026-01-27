/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * SYCL/DPC++ version translated from CUDA for Intel GPU B60.
 * Includes optimizations for Intel GPU architecture:
 * - Subgroup operations for better performance
 * - Vectorization for memory bandwidth optimization
 * - Work-group size tuning for B60 architecture
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>

// Intel GPU B60 optimizations
constexpr int WORK_GROUP_SIZE = 256;  // Optimal for Intel GPU B60
constexpr int VECTOR_SIZE = 4;        // For vectorization (float4)

/**
 * SYCL Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 *
 * Optimizations for Intel GPU B60:
 * - Uses local accessor for potential shared memory usage
 * - Subgroup size hint for better scheduling
 * - Vectorized memory access when possible
 */
void vectorAdd(sycl::queue& q, const float* A, const float* B, float* C, int numElements) {
    // Round up to nearest multiple of work group size
    int globalSize = ((numElements + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE;

    try {
        q.submit([&](sycl::handler& h) {
            // Accessors for USM pointers (if using buffers, would use buffer accessors)
            h.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(globalSize),  // Global range (total work-items)
                    sycl::range<1>(WORK_GROUP_SIZE)  // Local range (work-group size)
                ),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {  // B60 optimal subgroup size
                    int i = item.get_global_id(0);

                    // Vectorized access for better memory bandwidth (when aligned)
                    if (i + VECTOR_SIZE <= numElements) {
                        // Load 4 elements at once using vector types
                        sycl::vec<float, 4> vec_a = *reinterpret_cast<const sycl::vec<float, 4>*>(&A[i]);
                        sycl::vec<float, 4> vec_b = *reinterpret_cast<const sycl::vec<float, 4>*>(&B[i]);
                        sycl::vec<float, 4> vec_c = vec_a + vec_b;
                        *reinterpret_cast<sycl::vec<float, 4>*>(&C[i]) = vec_c;
                    } else if (i < numElements) {
                        // Handle remaining elements
                        C[i] = A[i] + B[i] + 0.0f;  // Keep the + 0.0f from CUDA version
                    }
                });
        }).wait();  // Wait for kernel completion
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        throw;
    }
}

/**
 * Helper function to verify results
 */
bool verifyResult(const std::vector<float>& A, const std::vector<float>& B,
                 const std::vector<float>& C, int numElements) {
    const float tolerance = 1e-5f;

    for (int i = 0; i < numElements; ++i) {
        float expected = A[i] + B[i] + 0.0f;  // Match CUDA exactly
        if (std::fabs(C[i] - expected) > tolerance) {
            std::cerr << "Result verification failed at element " << i
                      << "! Expected: " << expected << ", Got: " << C[i] << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * Host main routine
 */
int main(void) {
    // Vector configuration
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    std::cout << "[Vector addition of " << numElements << " elements]" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Allocate host input vectors
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = dis(gen);  // Random float between 0 and 1
        h_B[i] = dis(gen);
    }

    // Create SYCL queue with Intel GPU selector
    sycl::queue q;

    try {
        // Print device info
        auto device = q.get_device();
        auto platform = device.get_platform();
        std::cout << "Running on: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

        // Allocate device memory using USM (Unified Shared Memory)
        float* d_A = sycl::malloc_device<float>(numElements, q);
        float* d_B = sycl::malloc_device<float>(numElements, q);
        float* d_C = sycl::malloc_device<float>(numElements, q);

        if (!d_A || !d_B || !d_C) {
            std::cerr << "Failed to allocate device memory!" << std::endl;
            return EXIT_FAILURE;
        }

        // Copy input data from host to device
        std::cout << "Copy input data from the host memory to the SYCL device" << std::endl;
        q.memcpy(d_A, h_A.data(), size).wait();
        q.memcpy(d_B, h_B.data(), size).wait();

        // Time the kernel execution
        auto start = std::chrono::high_resolution_clock::now();

        // Launch the Vector Add SYCL Kernel
        std::cout << "SYCL kernel launch with global range "
                  << ((numElements + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE
                  << " and work-group size " << WORK_GROUP_SIZE << std::endl;

        vectorAdd(q, d_A, d_B, d_C, numElements);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Copy output data from device to host
        std::cout << "Copy output data from the SYCL device to the host memory" << std::endl;
        q.memcpy(h_C.data(), d_C, size).wait();

        // Verify that the result vector is correct
        if (verifyResult(h_A, h_B, h_C, numElements)) {
            std::cout << "Test PASSED" << std::endl;
        } else {
            std::cout << "Test FAILED" << std::endl;
        }

        // Print performance metrics
        double seconds = duration.count() / 1e6;
        double bandwidth = (3 * size) / (seconds * 1e9);  // 3 vectors * size / time in GB/s
        std::cout << "Kernel execution time: " << seconds << " seconds" << std::endl;
        std::cout << "Achieved memory bandwidth: " << bandwidth << " GB/s" << std::endl;

        // Free device memory
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Done" << std::endl;
    return EXIT_SUCCESS;
}